"""
Hybrid Pipeline — runs SQL and Vector pipelines concurrently and synthesizes
a unified answer that integrates quantitative data with qualitative context.

Three-step process:
  Step 1: Execute SQL pipeline AND Vector pipeline concurrently (asyncio.gather)
  Step 2: Merge artifacts from both pipelines
  Step 3: LLM synthesizes a unified answer from the combined context

Public API:
    execute_hybrid_pipeline(question: str) -> dict

Return shape (matches models.schemas.Artifacts fields — all fields populated):
    {
        "answer":           str,                    # unified synthesized answer
        "sql_query":        str | None,             # generated SQL
        "sql_results":      list[dict] | None,      # raw SQL rows
        "sql_row_count":    int | None,             # number of SQL rows
        "retrieved_chunks": list[str] | None,       # top-k document chunks
        "chunk_scores":     list[float] | None,     # cosine distance scores
        "chunk_metadata":   list[dict] | None,      # chunk metadata dicts
        "error":            str | None,             # combined error summary (None on full success)
    }

Graceful degradation:
    - If SQL pipeline fails entirely → synthesize from Vector results only
    - If Vector pipeline fails entirely → synthesize from SQL results only
    - If both fail → return fallback answer with combined error message
    - asyncio.gather uses return_exceptions=True so one failure never kills the other

Notes:
    - The hybrid synthesis LLM call receives raw artifacts (SQL rows + chunks),
      NOT the individual pipeline answers. This produces a truly integrated
      response rather than a concatenation of two separate answers.
    - All errors are caught and returned in the "error" field — this function
      never raises an exception to the caller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from openai import AsyncOpenAI

from agent.prompts import (
    build_hybrid_synthesis_system_prompt,
    build_hybrid_synthesis_user_message,
)
from agent.sql_pipeline import execute_sql_pipeline
from agent.vector_pipeline import execute_vector_pipeline
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hybrid synthesis LLM parameters
_SYNTH_TEMPERATURE = 0.1
_SYNTH_MAX_TOKENS = 1536  # Hybrid answers are typically longer — more context to integrate

# Fallback answer when both pipelines fail
_FALLBACK_ANSWER = (
    "I encountered errors in both the database and document search pipelines "
    "while processing your question. Please try again. If the issue persists, "
    "try asking a purely numerical question (e.g., 'What was D_402 revenue in Q4 2025?') "
    "or a purely strategic question (e.g., 'What is Project Phoenix?') to isolate "
    "which data source is unavailable."
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
# Step 1 & 2: Concurrent execution + artifact merging
# ---------------------------------------------------------------------------


async def _run_pipelines_concurrently(
    question: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Runs the SQL and Vector pipelines concurrently using asyncio.gather.

    Uses return_exceptions=True so that if one pipeline raises an unexpected
    exception (beyond its own internal error handling), the other pipeline's
    result is still captured and used.

    Args:
        question: The user's natural language question.

    Returns:
        (sql_result, vector_result) — dicts from each pipeline.
        If a pipeline raised an unexpected exception, a synthetic error dict
        matching that pipeline's return shape is returned instead.
    """
    logger.info(
        "Hybrid pipeline: launching SQL and Vector pipelines concurrently for: %r",
        question[:100],
    )
    start = time.perf_counter()

    # Both pipelines handle their own errors internally and never raise.
    # return_exceptions=True is a safety net for truly unexpected failures.
    results = await asyncio.gather(
        execute_sql_pipeline(question),
        execute_vector_pipeline(question),
        return_exceptions=True,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Hybrid pipeline: both pipelines completed in %.1fms",
        elapsed_ms,
    )

    sql_result, vector_result = results

    # Convert any unexpected exceptions into synthetic error dicts
    if isinstance(sql_result, BaseException):
        logger.error(
            "SQL pipeline raised an unexpected exception: %s: %s",
            type(sql_result).__name__,
            sql_result,
        )
        sql_result = {
            "answer": "",
            "sql_query": None,
            "sql_results": None,
            "sql_row_count": None,
            "error": f"SQL pipeline exception: {type(sql_result).__name__}: {sql_result}",
        }

    if isinstance(vector_result, BaseException):
        logger.error(
            "Vector pipeline raised an unexpected exception: %s: %s",
            type(vector_result).__name__,
            vector_result,
        )
        vector_result = {
            "answer": "",
            "retrieved_chunks": None,
            "chunk_scores": None,
            "chunk_metadata": None,
            "error": f"Vector pipeline exception: {type(vector_result).__name__}: {vector_result}",
        }

    sql_error = sql_result.get("error")
    vec_error = vector_result.get("error")

    if sql_error:
        logger.warning("SQL sub-pipeline reported error: %s", sql_error[:100])
    else:
        logger.info(
            "SQL sub-pipeline succeeded: %d row(s) returned",
            sql_result.get("sql_row_count") or 0,
        )

    if vec_error:
        logger.warning("Vector sub-pipeline reported error: %s", vec_error[:100])
    else:
        logger.info(
            "Vector sub-pipeline succeeded: %d chunk(s) retrieved",
            len(vector_result.get("retrieved_chunks") or []),
        )

    return sql_result, vector_result


def _merge_errors(
    sql_error: Optional[str],
    vector_error: Optional[str],
) -> Optional[str]:
    """
    Combines error messages from both pipelines into a single error string.

    Returns None if both pipelines succeeded. Returns a combined message if
    one or both pipelines had errors.

    Args:
        sql_error:    Error from the SQL pipeline (None if successful).
        vector_error: Error from the Vector pipeline (None if successful).

    Returns:
        Combined error string, or None if no errors occurred.
    """
    errors = []
    if sql_error:
        errors.append(f"SQL: {sql_error}")
    if vector_error:
        errors.append(f"Vector: {vector_error}")

    if not errors:
        return None
    return " | ".join(errors)


# ---------------------------------------------------------------------------
# Step 3: Hybrid Answer Synthesis
# ---------------------------------------------------------------------------


async def _synthesize_hybrid_answer(
    question: str,
    sql_result: dict[str, Any],
    vector_result: dict[str, Any],
) -> str:
    """
    Step 3: Calls the LLM to synthesize a unified answer from both pipeline results.

    Passes the raw SQL artifacts and Vector artifacts to the hybrid synthesis
    prompt — NOT the individual pipeline answers. This ensures the LLM produces
    a truly integrated response rather than a concatenation of two separate answers.

    Handles partial failure gracefully:
    - If SQL failed (no results), synthesis relies primarily on the vector chunks.
    - If Vector failed (no chunks), synthesis relies primarily on the SQL data.
    - If both failed, returns the fallback answer immediately (no LLM call).

    Returns:
        Synthesized natural language answer string.
    """
    sql_error = sql_result.get("error")
    vec_error = vector_result.get("error")

    sql_query: str = sql_result.get("sql_query") or ""
    sql_results: list[dict[str, Any]] = sql_result.get("sql_results") or []

    retrieved_chunks: list[str] = vector_result.get("retrieved_chunks") or []
    chunk_scores: list[float] = vector_result.get("chunk_scores") or []
    chunk_metadata: list[dict[str, Any]] = vector_result.get("chunk_metadata") or []

    sql_has_data = bool(sql_results)
    vec_has_data = bool(retrieved_chunks)

    # Both pipelines returned no data — skip LLM call entirely
    if not sql_has_data and not vec_has_data:
        logger.error(
            "Hybrid synthesis: both pipelines returned no data. "
            "SQL error: %s | Vector error: %s",
            sql_error,
            vec_error,
        )
        return _FALLBACK_ANSWER

    logger.info(
        "Hybrid synthesis: SQL rows=%d, Vector chunks=%d | "
        "SQL error=%s, Vector error=%s",
        len(sql_results),
        len(retrieved_chunks),
        bool(sql_error),
        bool(vec_error),
    )

    user_message = build_hybrid_synthesis_user_message(
        question=question,
        sql_query=sql_query if sql_query else "-- SQL query not available",
        sql_results=sql_results,
        retrieved_chunks=retrieved_chunks,
        chunk_scores=chunk_scores,
        chunk_metadata=chunk_metadata,
    )

    # Append partial failure context so the LLM can acknowledge the limitation
    partial_failure_notes = []
    if sql_error and not sql_has_data:
        partial_failure_notes.append(
            f"NOTE: The database query encountered an error ({sql_error[:100]}). "
            "The answer below is based on document context only."
        )
    if vec_error and not vec_has_data:
        partial_failure_notes.append(
            f"NOTE: The document search encountered an error ({vec_error[:100]}). "
            "The answer below is based on database data only."
        )

    if partial_failure_notes:
        user_message += "\n\n" + "\n".join(partial_failure_notes)

    start = time.perf_counter()
    client = _get_client()

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": build_hybrid_synthesis_system_prompt(),
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            temperature=_SYNTH_TEMPERATURE,
            max_tokens=_SYNTH_MAX_TOKENS,
        )
    except Exception as e:
        logger.error("Hybrid synthesis LLM call failed: %s", e, exc_info=True)
        # Degrade to concatenating individual pipeline answers
        sql_answer = sql_result.get("answer", "")
        vec_answer = vector_result.get("answer", "")
        if sql_answer and vec_answer:
            return (
                f"{sql_answer}\n\n"
                f"Additionally, from strategic documents: {vec_answer}"
            )
        return sql_answer or vec_answer or _FALLBACK_ANSWER

    elapsed_ms = (time.perf_counter() - start) * 1000
    answer = (response.choices[0].message.content or "").strip()

    if not answer:
        logger.warning("Hybrid synthesis returned empty answer (%.1fms)", elapsed_ms)
        sql_answer = sql_result.get("answer", "")
        vec_answer = vector_result.get("answer", "")
        return sql_answer or vec_answer or _FALLBACK_ANSWER

    logger.info(
        "Hybrid synthesis complete (%.1fms): %r",
        elapsed_ms,
        answer[:150],
    )
    return answer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_hybrid_pipeline(question: str) -> dict[str, Any]:
    """
    Executes the full Hybrid pipeline for a natural language question.

    Three-step process:
      1. Run SQL and Vector pipelines concurrently via asyncio.gather
      2. Collect and validate artifacts from both pipelines
      3. LLM synthesizes a unified answer from the combined raw artifacts

    Args:
        question: The user's natural language question (pre-validated by Pydantic).

    Returns:
        A dict with ALL artifact fields from both pipelines:
        {
            "answer":           str,          # unified synthesized answer
            "sql_query":        str | None,   # generated SQL
            "sql_results":      list | None,  # raw SQL rows as list of dicts
            "sql_row_count":    int | None,   # number of SQL rows returned
            "retrieved_chunks": list | None,  # text of top-k document chunks
            "chunk_scores":     list | None,  # cosine distance scores
            "chunk_metadata":   list | None,  # metadata dicts for each chunk
            "error":            str | None,   # combined error summary (None on full success)
        }

    Notes:
        - Never raises an exception — all errors are caught and returned in the
          "error" field with a helpful "answer" string.
        - SQL and Vector pipelines run concurrently — total latency ≈ max(sql_latency,
          vector_latency) rather than sql_latency + vector_latency.
        - If one pipeline fails, synthesis proceeds with the other pipeline's data.
        - If both pipelines fail, returns the fallback answer immediately (no LLM call).
        - The synthesis LLM call uses raw artifacts (not individual pipeline answers)
          to produce a truly integrated response.
    """
    pipeline_start = time.perf_counter()
    logger.info("Hybrid pipeline starting for question: %r", question[:100])

    # ── Steps 1 & 2: Concurrent execution + artifact collection ─────────────
    sql_result, vector_result = await _run_pipelines_concurrently(question)

    # ── Step 3: Unified synthesis ────────────────────────────────────────────
    answer = await _synthesize_hybrid_answer(
        question=question,
        sql_result=sql_result,
        vector_result=vector_result,
    )

    # ── Assemble combined return dict ────────────────────────────────────────
    combined_error = _merge_errors(
        sql_error=sql_result.get("error"),
        vector_error=vector_result.get("error"),
    )

    elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info(
        "Hybrid pipeline complete in %.1fms | sql_rows=%s | vec_chunks=%s | error=%s",
        elapsed_ms,
        sql_result.get("sql_row_count"),
        len(vector_result.get("retrieved_chunks") or []),
        combined_error[:80] if combined_error else None,
    )

    return {
        "answer": answer,
        # SQL pipeline artifacts
        "sql_query": sql_result.get("sql_query"),
        "sql_results": sql_result.get("sql_results"),
        "sql_row_count": sql_result.get("sql_row_count"),
        # Vector pipeline artifacts
        "retrieved_chunks": vector_result.get("retrieved_chunks"),
        "chunk_scores": vector_result.get("chunk_scores"),
        "chunk_metadata": vector_result.get("chunk_metadata"),
        # Combined error (None if both pipelines succeeded)
        "error": combined_error,
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
        (
            "Why did department D_402's EBITDA margin improve from Q3 to Q4 2025, "
            "and what strategic initiative drove it?"
        ),
        "Did Project Phoenix achieve its financial targets for D_402?",
        "How did the restructuring affect headcount in both departments, and what was the plan?",
        "What were the restructuring charges in Q3 2025 and why were they incurred?",
        "How did D_517's sales pipeline and bookings change in 2025, and what drove this?",
    ]

    async def run_tests(questions: list[str]) -> None:
        from db.sqlite_init import init_sqlite
        from db.vector_init import init_vector_store

        init_sqlite()
        init_vector_store()

        print("\n" + "=" * 70)
        print("HYBRID PIPELINE TEST — calling OpenAI API + SQLite + ChromaDB")
        print("=" * 70)

        for i, question in enumerate(questions, 1):
            print(f"\n{'─'*70}")
            print(f"[{i}/{len(questions)}] Question: {question}")
            print("─" * 70)

            try:
                result = await execute_hybrid_pipeline(question)

                print(f"SQL Query:      {result['sql_query']}")
                print(f"SQL Row Count:  {result['sql_row_count']}")
                print(
                    f"SQL Results:    "
                    f"{json.dumps(result['sql_results'], indent=2) if result['sql_results'] else 'None'}"
                )
                print(f"Chunks Retrieved: {len(result['retrieved_chunks'] or [])}")
                if result["retrieved_chunks"]:
                    for j, (chunk, score, meta) in enumerate(
                        zip(
                            result["retrieved_chunks"],
                            result["chunk_scores"],
                            result["chunk_metadata"],
                        ),
                        1,
                    ):
                        section = meta.get("section", "Unknown")
                        print(
                            f"  Chunk {j}: section='{section}' | "
                            f"distance={score:.4f} | "
                            f"preview='{chunk[:60]}...'"
                        )
                if result["error"]:
                    print(f"Error:          {result['error']}")
                print(f"\nAnswer:\n{result['answer']}")

            except Exception as e:
                print(f"💥 UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n{'='*70}")
        print("Hybrid Pipeline tests complete.")
        print("=" * 70)

    if len(sys.argv) > 1:
        single_question = " ".join(sys.argv[1:])

        async def run_single() -> None:
            from db.sqlite_init import init_sqlite
            from db.vector_init import init_vector_store

            init_sqlite()
            init_vector_store()

            print(f"\nQuestion: {single_question}\n")
            result = await execute_hybrid_pipeline(single_question)

            print(f"SQL Query:     {result['sql_query']}")
            print(f"SQL Rows:      {result['sql_row_count']}")
            print(
                f"SQL Results:   "
                f"{json.dumps(result['sql_results'], indent=2) if result['sql_results'] else 'None'}"
            )
            print(f"Chunks:        {len(result['retrieved_chunks'] or [])}")
            if result["retrieved_chunks"]:
                for j, (chunk, score, meta) in enumerate(
                    zip(
                        result["retrieved_chunks"],
                        result["chunk_scores"],
                        result["chunk_metadata"],
                    ),
                    1,
                ):
                    section = meta.get("section", "Unknown")
                    print(
                        f"  Chunk {j}: section='{section}' | "
                        f"distance={score:.4f} | "
                        f"preview='{chunk[:80]}...'"
                    )
            if result["error"]:
                print(f"Error:         {result['error']}")
            print(f"\nAnswer:\n{result['answer']}")

        asyncio.run(run_single())
    else:
        asyncio.run(run_tests(TEST_QUESTIONS))
