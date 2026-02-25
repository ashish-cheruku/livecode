"""
Query Router — classifies a natural language question into one of three
retrieval pipelines: SQL, VECTOR, or HYBRID.

Public API:
    route_query(question: str) -> str
        Returns exactly one of: "SQL", "VECTOR", "HYBRID"

Internal flow:
    1. Build system + user messages from agent/prompts.py
    2. Call OpenAI chat completions (temperature=0 for determinism)
    3. Parse the JSON response to extract the "route" field
    4. Apply multi-layer fallback if parsing fails
    5. Return the validated route string

Fallback chain (most → least preferred):
    1. Direct json.loads() on the raw LLM response
    2. Strip markdown code fences, then json.loads()
    3. Regex extraction of "route" field from anywhere in the response
    4. Keyword scanning (does the text contain SQL/VECTOR/HYBRID?)
    5. Default to "HYBRID" (safest: most informative, never loses information)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from agent.prompts import build_router_system_prompt, build_router_user_message
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ROUTES = frozenset({"SQL", "VECTOR", "HYBRID"})
DEFAULT_FALLBACK_ROUTE = "HYBRID"

# Router LLM parameters — temperature=0 for deterministic classification
_ROUTER_TEMPERATURE = 0.0
_ROUTER_MAX_TOKENS = 256  # Route + reasoning fits comfortably in 256 tokens

# ---------------------------------------------------------------------------
# Internal result type
# ---------------------------------------------------------------------------


@dataclass
class RouteResult:
    """
    Internal result from the router LLM call.

    Carries both the validated route string and the LLM's reasoning text
    for logging and observability. The public `route_query()` function
    returns only the `route` string.
    """

    route: str
    reasoning: str
    raw_response: str
    latency_ms: float
    fallback_used: bool = False
    fallback_method: Optional[str] = None


# ---------------------------------------------------------------------------
# OpenAI client (module-level singleton — reused across requests)
# ---------------------------------------------------------------------------

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    """
    Returns the module-level AsyncOpenAI client, creating it on first call.
    Using a singleton avoids re-initializing the HTTP connection pool on
    every request.
    """
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """
    Removes markdown code fences that LLMs sometimes wrap JSON in.

    Handles patterns like:
        ```json
        { ... }
        ```
    and:
        ```
        { ... }
        ```
    """
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Attempts to extract a JSON object from a string using multiple strategies.

    Strategy 1: Direct json.loads() — works when the LLM returns clean JSON.
    Strategy 2: Strip markdown fences, then json.loads().
    Strategy 3: Find the first {...} block via regex and parse it.

    Returns the parsed dict, or None if all strategies fail.
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown fences
    stripped = _strip_markdown_fences(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Regex — find first {...} block (handles extra text before/after JSON)
    json_pattern = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_pattern:
        try:
            return json.loads(json_pattern.group())
        except json.JSONDecodeError:
            pass

    return None


def _extract_route_from_json(parsed: dict) -> Optional[str]:
    """
    Extracts and validates the 'route' field from a parsed JSON dict.

    Handles case variations (e.g., "sql" → "SQL") and strips whitespace.
    Returns the normalized route string, or None if not found/invalid.
    """
    route_raw = parsed.get("route") or parsed.get("Route") or parsed.get("ROUTE")
    if route_raw is None:
        return None

    route_normalized = str(route_raw).strip().upper()
    if route_normalized in VALID_ROUTES:
        return route_normalized

    logger.warning(
        "Router returned unrecognized route value: %r (normalized: %r)",
        route_raw,
        route_normalized,
    )
    return None


def _extract_reasoning_from_json(parsed: dict) -> str:
    """
    Extracts the 'reasoning' field from a parsed JSON dict.
    Returns an empty string if not present.
    """
    return str(
        parsed.get("reasoning")
        or parsed.get("Reasoning")
        or parsed.get("REASONING")
        or ""
    ).strip()


def _keyword_scan_fallback(text: str) -> Optional[str]:
    """
    Last-resort fallback: scan the raw LLM response text for route keywords.

    Checks for "HYBRID" first (most specific), then "VECTOR", then "SQL"
    (most likely to appear as a false positive in other contexts).

    Returns the first matching route, or None if none found.
    """
    text_upper = text.upper()

    # Check in specificity order: HYBRID > VECTOR > SQL
    for route in ("HYBRID", "VECTOR", "SQL"):
        if route in text_upper:
            return route

    return None


def _parse_router_response(raw_response: str) -> tuple[str, str, bool, Optional[str]]:
    """
    Parses the raw LLM response string into (route, reasoning, fallback_used, fallback_method).

    Applies the full fallback chain:
      1. JSON extraction + route field validation
      2. Keyword scanning
      3. Default to HYBRID

    Returns:
        route:           Validated route string ("SQL", "VECTOR", or "HYBRID")
        reasoning:       LLM's reasoning text (empty string if unavailable)
        fallback_used:   True if any fallback was needed
        fallback_method: Description of which fallback was used (or None)
    """
    # ── Attempt 1: JSON parsing ──────────────────────────────────────────────
    parsed = _extract_json_object(raw_response)
    if parsed is not None:
        route = _extract_route_from_json(parsed)
        if route is not None:
            reasoning = _extract_reasoning_from_json(parsed)
            return route, reasoning, False, None

        # JSON parsed but route field missing or invalid
        logger.warning(
            "Router JSON parsed successfully but 'route' field is missing or invalid. "
            "Parsed keys: %s. Raw: %r",
            list(parsed.keys()),
            raw_response[:200],
        )

    # ── Attempt 2: Keyword scanning ──────────────────────────────────────────
    keyword_route = _keyword_scan_fallback(raw_response)
    if keyword_route is not None:
        logger.warning(
            "Router JSON parsing failed; used keyword scan fallback. "
            "Found route: %r in response: %r",
            keyword_route,
            raw_response[:200],
        )
        return keyword_route, "", True, "keyword_scan"

    # ── Attempt 3: Default fallback ──────────────────────────────────────────
    logger.error(
        "Router could not determine route from response: %r. "
        "Defaulting to %r.",
        raw_response[:200],
        DEFAULT_FALLBACK_ROUTE,
    )
    return DEFAULT_FALLBACK_ROUTE, "", True, "default_fallback"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def route_query(question: str) -> str:
    """
    Classifies a natural language question and returns the appropriate pipeline route.

    Args:
        question: The user's natural language question (already stripped/validated
                  by the Pydantic QueryRequest model).

    Returns:
        One of: "SQL", "VECTOR", "HYBRID"

    Raises:
        openai.APIError: If the OpenAI API call fails (network error, auth error, etc.)
        ValueError: If question is empty (should be caught upstream by Pydantic).

    Notes:
        - Uses temperature=0 for deterministic, consistent routing.
        - Applies a multi-layer fallback chain if the LLM returns malformed output.
        - Logs the routing decision and reasoning at INFO level for observability.
        - Never raises an exception due to parsing failure — always returns a valid route.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    start_time = time.perf_counter()

    system_prompt = build_router_system_prompt()
    user_message = build_router_user_message(question)

    logger.info("Routing question: %r", question[:100])

    client = _get_client()

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=_ROUTER_TEMPERATURE,
        max_tokens=_ROUTER_MAX_TOKENS,
        # Request JSON output mode for supported models — provides a strong hint
        # to the model to return valid JSON, but we still apply fallback parsing
        # because older model versions may not honour this parameter.
        response_format={"type": "json_object"},
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    raw_response = response.choices[0].message.content or ""

    logger.debug(
        "Router raw LLM response (%.1fms): %r",
        elapsed_ms,
        raw_response[:300],
    )

    route, reasoning, fallback_used, fallback_method = _parse_router_response(raw_response)

    result = RouteResult(
        route=route,
        reasoning=reasoning,
        raw_response=raw_response,
        latency_ms=elapsed_ms,
        fallback_used=fallback_used,
        fallback_method=fallback_method,
    )

    _log_route_result(result, question)

    return result.route


async def route_query_with_details(question: str) -> RouteResult:
    """
    Extended version of route_query that returns the full RouteResult dataclass.

    Useful for debugging, testing, and future observability features where
    the reasoning text and latency are needed alongside the route.

    Args:
        question: The user's natural language question.

    Returns:
        RouteResult with route, reasoning, raw_response, latency_ms, and fallback info.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    start_time = time.perf_counter()

    system_prompt = build_router_system_prompt()
    user_message = build_router_user_message(question)

    client = _get_client()

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=_ROUTER_TEMPERATURE,
        max_tokens=_ROUTER_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    raw_response = response.choices[0].message.content or ""

    route, reasoning, fallback_used, fallback_method = _parse_router_response(raw_response)

    result = RouteResult(
        route=route,
        reasoning=reasoning,
        raw_response=raw_response,
        latency_ms=elapsed_ms,
        fallback_used=fallback_used,
        fallback_method=fallback_method,
    )

    _log_route_result(result, question)

    return result


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _log_route_result(result: RouteResult, question: str) -> None:
    """Logs the routing decision at the appropriate log level."""
    if result.fallback_used:
        logger.warning(
            "Route decision (FALLBACK via %s) | route=%r | latency=%.1fms | question=%r",
            result.fallback_method,
            result.route,
            result.latency_ms,
            question[:80],
        )
    else:
        logger.info(
            "Route decision | route=%r | reasoning=%r | latency=%.1fms | question=%r",
            result.route,
            result.reasoning[:100] if result.reasoning else "",
            result.latency_ms,
            question[:80],
        )


# ---------------------------------------------------------------------------
# __main__ — manual testing without running the full FastAPI server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Test questions covering all three routes
    TEST_QUESTIONS = [
        # Expected: SQL
        ("What was the total revenue for department D_402 in fiscal year 2025?", "SQL"),
        ("How did EBITDA margins change from Q3 to Q4 2025 for both departments?", "SQL"),
        ("Which department had more contractors at the end of Q4 2024?", "SQL"),
        ("What were the restructuring charges in Q3 2025?", "SQL"),
        # Expected: VECTOR
        ("What is Project Phoenix and what are its expected outcomes?", "VECTOR"),
        ("What is the strategic rationale for the restructuring?", "VECTOR"),
        ("What risks did the company identify in the restructuring memo?", "VECTOR"),
        ("What is the go-to-market strategy for Enterprise Sales?", "VECTOR"),
        # Expected: HYBRID
        (
            "Why did department D_402's EBITDA margin improve from Q3 to Q4 2025, "
            "and what strategic initiative drove it?",
            "HYBRID",
        ),
        ("Did Project Phoenix achieve its financial targets?", "HYBRID"),
        ("How did the restructuring affect headcount and what was the plan?", "HYBRID"),
    ]

    async def run_tests() -> None:
        print("\n" + "=" * 70)
        print("ROUTER TEST — calling OpenAI API")
        print("=" * 70)

        correct = 0
        total = len(TEST_QUESTIONS)

        for question, expected in TEST_QUESTIONS:
            try:
                result = await route_query_with_details(question)
                status = "✅" if result.route == expected else "❌"
                if result.route == expected:
                    correct += 1
                print(
                    f"\n{status} Expected={expected:<7} Got={result.route:<7} "
                    f"({result.latency_ms:.0f}ms)"
                    + (" [FALLBACK]" if result.fallback_used else "")
                )
                print(f"   Q: {question[:80]}")
                if result.reasoning:
                    print(f"   R: {result.reasoning[:100]}")
            except Exception as e:
                print(f"\n💥 ERROR for question: {question[:60]}")
                print(f"   {type(e).__name__}: {e}")

        print(f"\n{'='*70}")
        print(f"Results: {correct}/{total} correct ({correct/total*100:.0f}%)")
        print("=" * 70)

    # Allow passing a single question as a CLI argument for quick testing
    if len(sys.argv) > 1:
        single_question = " ".join(sys.argv[1:])

        async def run_single() -> None:
            result = await route_query_with_details(single_question)
            print(f"\nQuestion: {single_question}")
            print(f"Route:    {result.route}")
            print(f"Reasoning: {result.reasoning}")
            print(f"Latency:  {result.latency_ms:.1f}ms")
            if result.fallback_used:
                print(f"⚠️  Fallback used: {result.fallback_method}")

        asyncio.run(run_single())
    else:
        asyncio.run(run_tests())
