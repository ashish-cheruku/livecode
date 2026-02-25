"""
SQL Pipeline — converts a natural language question into a database answer
via a three-step process:

  Step 1: LLM generates a SQL SELECT statement from the question + schema
  Step 2: Execute the SQL against SQLite (read-only, with safety validation)
  Step 3: LLM synthesizes a natural language answer from the raw SQL results

Public API:
    execute_sql_pipeline(question: str) -> dict

Return shape (matches models.schemas.Artifacts fields):
    {
        "answer":        str,                    # synthesized natural language answer
        "sql_query":     str | None,             # generated SQL (None if generation failed)
        "sql_results":   list[dict] | None,      # raw rows as list of dicts
        "sql_row_count": int | None,             # number of rows returned
        "error":         str | None,             # error message if pipeline partially failed
    }
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import time
from typing import Any, Optional

from openai import AsyncOpenAI

from agent.prompts import (
    build_sql_generation_system_prompt,
    build_sql_generation_user_message,
    build_sql_synthesis_system_prompt,
    build_sql_synthesis_user_message,
)
from config import settings
from db.sqlite_init import get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SQL generation: temperature=0 for deterministic, reproducible queries
_SQL_GEN_TEMPERATURE = 0.0
_SQL_GEN_MAX_TOKENS = 512  # A SELECT statement rarely exceeds 512 tokens

# SQL synthesis: slight temperature for more natural language variation
_SQL_SYNTH_TEMPERATURE = 0.1
_SQL_SYNTH_MAX_TOKENS = 1024  # Answer synthesis may need more tokens

# SQL safety: only allow SELECT statements (defense-in-depth)
_ALLOWED_SQL_PREFIXES = ("SELECT", "WITH")  # WITH for CTEs that start with SELECT

# Fallback answer when the pipeline encounters an unrecoverable error
_FALLBACK_ANSWER = (
    "I encountered an error while processing your question against the database. "
    "Please try rephrasing your question or check that you're asking about "
    "departments D_402 or D_517 for fiscal years 2024-2025."
)

# ---------------------------------------------------------------------------
# OpenAI client singleton
# ---------------------------------------------------------------------------

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    """
    Returns the module-level AsyncOpenAI client, creating it on first call.
    Reuses the same HTTP connection pool across all pipeline invocations.
    """
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


# ---------------------------------------------------------------------------
# Step 1: SQL Generation
# ---------------------------------------------------------------------------


def _clean_sql_output(raw_sql: str) -> str:
    """
    Cleans the raw LLM output to extract a pure SQL statement.

    Handles common LLM output patterns:
      - Markdown code fences: ```sql ... ``` or ``` ... ```
      - Leading/trailing whitespace and newlines
      - Explanatory text before or after the SQL
      - Trailing semicolons (preserved — SQLite handles them fine)
    """
    text = raw_sql.strip()

    # Remove markdown code fences (```sql ... ``` or ``` ... ```)
    text = re.sub(r"^```(?:sql)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # If the LLM added explanatory text before the SQL, find the first
    # SELECT or WITH keyword and take everything from there
    select_match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if select_match:
        text = text[select_match.start():]

    # Remove any trailing explanation after the semicolon
    semicolon_pos = text.find(";")
    if semicolon_pos != -1:
        text = text[: semicolon_pos + 1]

    return text.strip()


def _validate_sql_safety(sql: str) -> tuple[bool, str]:
    """
    Validates that the generated SQL is safe to execute (read-only).

    Checks:
      1. The statement starts with SELECT or WITH (for CTEs)
      2. Does not contain dangerous DML/DDL keywords

    Returns:
        (is_safe, error_message) — is_safe=True means the SQL passed validation.
    """
    if not sql:
        return False, "Generated SQL is empty."

    sql_upper = sql.upper().strip()

    # Must start with SELECT or WITH
    starts_with_allowed = any(
        sql_upper.startswith(prefix) for prefix in _ALLOWED_SQL_PREFIXES
    )
    if not starts_with_allowed:
        return False, (
            f"Generated SQL does not start with SELECT or WITH. "
            f"Got: {sql[:80]!r}. Only read-only SELECT statements are permitted."
        )

    # Block dangerous DML/DDL keywords (word-boundary anchors avoid false positives)
    dangerous_patterns = [
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bDELETE\b",
        r"\bDROP\b",
        r"\bCREATE\b",
        r"\bALTER\b",
        r"\bTRUNCATE\b",
        r"\bREPLACE\b",
        r"\bMERGE\b",
        r"\bEXEC\b",
        r"\bEXECUTE\b",
        r"\bATTACH\b",
        r"\bDETACH\b",
        r"\bPRAGMA\b",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, sql_upper):
            keyword = pattern.replace(r"\b", "")
            return False, (
                f"Generated SQL contains a forbidden keyword: {keyword}. "
                f"Only SELECT statements are permitted."
            )

    return True, ""


async def _generate_sql(question: str) -> tuple[str, Optional[str]]:
    """
    Step 1: Calls the LLM to generate a SQL query from the natural language question.

    Returns:
        (cleaned_sql, error_message)
        - On success: (valid SQL string, None)
        - On failure: ("", error description string)
    """
    logger.info("SQL generation: generating SQL for question: %r", question[:100])
    start = time.perf_counter()

    client = _get_client()

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": build_sql_generation_system_prompt(),
                },
                {
                    "role": "user",
                    "content": build_sql_generation_user_message(question),
                },
            ],
            temperature=_SQL_GEN_TEMPERATURE,
            max_tokens=_SQL_GEN_MAX_TOKENS,
        )
    except Exception as e:
        logger.error("SQL generation LLM call failed: %s", e, exc_info=True)
        return "", f"LLM call for SQL generation failed: {type(e).__name__}: {e}"

    elapsed_ms = (time.perf_counter() - start) * 1000
    raw_output = response.choices[0].message.content or ""

    logger.debug(
        "SQL generation raw LLM output (%.1fms): %r",
        elapsed_ms,
        raw_output[:300],
    )

    if not raw_output.strip():
        return "", "LLM returned an empty response for SQL generation."

    cleaned_sql = _clean_sql_output(raw_output)

    is_safe, safety_error = _validate_sql_safety(cleaned_sql)
    if not is_safe:
        logger.warning(
            "SQL safety validation failed: %s | raw_output=%r",
            safety_error,
            raw_output[:200],
        )
        return "", safety_error

    logger.info(
        "SQL generation complete (%.1fms): %r",
        elapsed_ms,
        cleaned_sql[:200],
    )
    return cleaned_sql, None


# ---------------------------------------------------------------------------
# Step 2: SQL Execution
# ---------------------------------------------------------------------------


def _execute_sql_sync(sql: str) -> tuple[list[dict[str, Any]], Optional[str]]:
    """
    Synchronous SQL execution against the SQLite database (read-only mode).

    Enforces settings.sql_max_rows to prevent oversized result sets from
    flooding the synthesis prompt's context window.

    Returns:
        (rows, error_message)
        - On success: (list of row dicts, None)
        - On failure: ([], error description string)
    """
    try:
        conn = get_connection(read_only=True)
    except Exception as e:
        logger.error("Failed to open SQLite connection: %s", e, exc_info=True)
        return [], f"Database connection failed: {type(e).__name__}: {e}"

    try:
        cursor = conn.execute(sql)
        # Fetch one extra to detect truncation without loading excess rows into memory
        raw_rows = cursor.fetchmany(settings.sql_max_rows + 1)

        truncated = len(raw_rows) > settings.sql_max_rows
        if truncated:
            raw_rows = raw_rows[: settings.sql_max_rows]
            logger.warning(
                "SQL result truncated to %d rows (sql_max_rows limit). SQL: %r",
                settings.sql_max_rows,
                sql[:100],
            )

        # sqlite3.Row objects must be converted to plain dicts for JSON serialization
        rows = [dict(row) for row in raw_rows]

        logger.info(
            "SQL execution: %d row(s) returned%s. SQL: %r",
            len(rows),
            " (TRUNCATED)" if truncated else "",
            sql[:100],
        )
        return rows, None

    except sqlite3.OperationalError as e:
        # Common causes: invalid column name, syntax error, table not found,
        # or write attempt on a read-only connection
        error_msg = f"SQL execution error (OperationalError): {e}"
        logger.warning("%s | SQL: %r", error_msg, sql[:200])
        return [], error_msg

    except sqlite3.DatabaseError as e:
        error_msg = f"SQL execution error (DatabaseError): {e}"
        logger.error("%s | SQL: %r", error_msg, sql[:200], exc_info=True)
        return [], error_msg

    except Exception as e:
        error_msg = f"Unexpected SQL execution error: {type(e).__name__}: {e}"
        logger.error("%s | SQL: %r", error_msg, sql[:200], exc_info=True)
        return [], error_msg

    finally:
        conn.close()


async def _execute_sql(sql: str) -> tuple[list[dict[str, Any]], Optional[str]]:
    """
    Async wrapper around the synchronous SQL execution.

    Runs the blocking SQLite call in a thread pool executor to avoid
    blocking the FastAPI event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _execute_sql_sync, sql)


# ---------------------------------------------------------------------------
# Step 3: Answer Synthesis
# ---------------------------------------------------------------------------


async def _synthesize_answer(
    question: str,
    sql_query: str,
    sql_results: list[dict[str, Any]],
    sql_error: Optional[str] = None,
) -> str:
    """
    Step 3: Calls the LLM to synthesize a natural language answer from SQL results.

    If SQL execution produced an error, the error context is included in the
    synthesis prompt so the LLM can return a helpful response rather than
    a raw exception message.

    Returns:
        Synthesized natural language answer string.
    """
    logger.info(
        "SQL synthesis: synthesizing answer from %d row(s)%s",
        len(sql_results),
        f" (with error: {sql_error[:60]})" if sql_error else "",
    )
    start = time.perf_counter()

    effective_sql = sql_query

    # When there was a SQL error, annotate the query string so the synthesis
    # prompt has full context about what went wrong
    if sql_error and not sql_results:
        effective_sql = (
            f"{sql_query}\n\n"
            f"-- NOTE: This query produced an error: {sql_error}"
        )

    user_message = build_sql_synthesis_user_message(
        question=question,
        sql_query=effective_sql,
        sql_results=sql_results,
    )

    # Append explicit error instructions so the LLM provides a helpful explanation
    if sql_error and not sql_results:
        user_message += (
            f"\n\nIMPORTANT: The SQL query above produced an error: {sql_error}\n"
            "Please explain to the user that the query could not be executed, "
            "describe what data was being sought, and suggest how they might "
            "rephrase their question. Be helpful and specific."
        )

    client = _get_client()

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": build_sql_synthesis_system_prompt(),
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            temperature=_SQL_SYNTH_TEMPERATURE,
            max_tokens=_SQL_SYNTH_MAX_TOKENS,
        )
    except Exception as e:
        logger.error("SQL synthesis LLM call failed: %s", e, exc_info=True)
        return _FALLBACK_ANSWER

    elapsed_ms = (time.perf_counter() - start) * 1000
    answer = (response.choices[0].message.content or "").strip()

    if not answer:
        logger.warning("SQL synthesis returned empty answer (%.1fms)", elapsed_ms)
        return _FALLBACK_ANSWER

    logger.info(
        "SQL synthesis complete (%.1fms): %r",
        elapsed_ms,
        answer[:150],
    )
    return answer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_sql_pipeline(question: str) -> dict[str, Any]:
    """
    Executes the full SQL pipeline for a natural language question.

    Three-step process:
      1. LLM generates a SQL SELECT statement from the question + schema
      2. Execute the SQL against SQLite in read-only mode
      3. LLM synthesizes a natural language answer from the raw results

    Args:
        question: The user's natural language question (pre-validated by Pydantic).

    Returns:
        A dict with keys: answer, sql_query, sql_results, sql_row_count, error.
        Matches models.schemas.Artifacts fields exactly.

    Notes:
        - Never raises an exception — all errors are caught and returned in
          the "error" field with a helpful "answer" string.
        - SQL generation failure: error returned, synthesis explains it.
        - SQL execution failure: bad SQL + error returned, synthesis explains it.
        - Empty results: SQL + empty list returned, synthesis explains no data found.
    """
    pipeline_start = time.perf_counter()
    logger.info("SQL pipeline starting for question: %r", question[:100])

    # ── Step 1: Generate SQL ─────────────────────────────────────────────────
    sql_query, gen_error = await _generate_sql(question)

    if gen_error:
        logger.warning("SQL generation failed: %s", gen_error)
        answer = await _synthesize_answer(
            question=question,
            sql_query="-- SQL generation failed",
            sql_results=[],
            sql_error=gen_error,
        )
        elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info("SQL pipeline complete (GENERATION FAILED) in %.1fms", elapsed_ms)
        return {
            "answer": answer,
            "sql_query": None,
            "sql_results": None,
            "sql_row_count": None,
            "error": gen_error,
        }

    # ── Step 2: Execute SQL ──────────────────────────────────────────────────
    sql_results, exec_error = await _execute_sql(sql_query)

    # ── Step 3: Synthesize Answer ────────────────────────────────────────────
    # Always attempt synthesis — even on execution error the LLM explains the issue
    answer = await _synthesize_answer(
        question=question,
        sql_query=sql_query,
        sql_results=sql_results,
        sql_error=exec_error,
    )

    elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
    row_count = len(sql_results) if sql_results is not None else None

    logger.info(
        "SQL pipeline complete in %.1fms | rows=%s | error=%s",
        elapsed_ms,
        row_count,
        exec_error[:60] if exec_error else None,
    )

    return {
        "answer": answer,
        "sql_query": sql_query,
        "sql_results": sql_results if sql_results is not None else [],
        "sql_row_count": row_count,
        "error": exec_error,
    }


# ---------------------------------------------------------------------------
# __main__ — manual testing without running the full FastAPI server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    TEST_QUESTIONS = [
        "What was the total revenue for department D_402 in fiscal year 2025?",
        "How did the EBITDA margin for D_402 change from Q3 to Q4 2025?",
        "What were the restructuring charges for both departments in Q3 2025?",
        "Which department had more full-time employees at the end of Q4 2025?",
        "Show me revenue, EBITDA, and headcount for all quarters in 2025.",
        "What was the average new bookings per quarter for D_517?",
    ]

    async def run_tests(questions: list[str]) -> None:
        print("\n" + "=" * 70)
        print("SQL PIPELINE TEST — calling OpenAI API + SQLite")
        print("=" * 70)

        for i, question in enumerate(questions, 1):
            print(f"\n{'─'*70}")
            print(f"[{i}/{len(questions)}] Question: {question}")
            print("─" * 70)

            try:
                result = await execute_sql_pipeline(question)

                print(f"SQL Query:   {result['sql_query']}")
                print(f"Row Count:   {result['sql_row_count']}")
                print(f"Raw Results: {json.dumps(result['sql_results'], indent=2)}")
                print(f"Error:       {result['error']}")
                print(f"\nAnswer:\n{result['answer']}")

            except Exception as e:
                print(f"💥 UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*70}")
        print("SQL Pipeline tests complete.")
        print("=" * 70)

    if len(sys.argv) > 1:
        single_question = " ".join(sys.argv[1:])

        async def run_single() -> None:
            print(f"\nQuestion: {single_question}\n")
            result = await execute_sql_pipeline(single_question)
            print(f"SQL:       {result['sql_query']}")
            print(f"Rows:      {result['sql_row_count']}")
            print(f"Results:   {json.dumps(result['sql_results'], indent=2)}")
            if result["error"]:
                print(f"Error:     {result['error']}")
            print(f"\nAnswer:\n{result['answer']}")

        asyncio.run(run_single())
    else:
        asyncio.run(run_tests(TEST_QUESTIONS))
