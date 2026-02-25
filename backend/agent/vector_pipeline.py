"""
Vector Search Pipeline — retrieves semantically relevant document chunks
from ChromaDB and synthesizes a natural language answer.

Two-step process:
  Step 1: Query ChromaDB for top-k similar chunks using cosine similarity
  Step 2: LLM synthesizes a natural language answer from the retrieved chunks

Public API:
    execute_vector_pipeline(question: str) -> dict

Return shape (matches models.schemas.Artifacts fields):
    {
        "answer":           str,                   # synthesized natural language answer
        "retrieved_chunks": list[str] | None,      # text of top-k chunks (most relevant first)
        "chunk_scores":     list[float] | None,    # cosine distance scores (lower = more similar)
        "chunk_metadata":   list[dict] | None,     # metadata dicts for each chunk
        "error":            str | None,            # error message if pipeline partially failed
    }

Notes:
    - ChromaDB's Python client is synchronous; retrieval runs in a thread pool
      executor to avoid blocking the FastAPI async event loop.
    - The module-level collection cache avoids repeated PersistentClient
      initialization on every request.
    - All errors are caught and returned in the "error" field — this function
      never raises an exception to the caller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import chromadb
from openai import AsyncOpenAI

from agent.prompts import (
    build_vector_synthesis_system_prompt,
    build_vector_synthesis_user_message,
)
from config import settings
from db.vector_init import get_vector_collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Synthesis LLM parameters
_SYNTH_TEMPERATURE = 0.1
_SYNTH_MAX_TOKENS = 1024

# Cosine distance threshold above which we consider a chunk "low relevance".
# ChromaDB returns distances in [0, 2] for cosine space; practically [0, 1]
# for well-formed embeddings. Values > 0.6 indicate a weak semantic match.
_LOW_RELEVANCE_THRESHOLD = 0.6

# Fallback answer when the pipeline encounters an unrecoverable error
_FALLBACK_ANSWER = (
    "I encountered an error while searching the document store. "
    "Please try rephrasing your question. If you're asking about "
    "Project Phoenix, the restructuring initiative, financial targets, "
    "or strategic rationale, I should be able to help."
)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_openai_client: Optional[AsyncOpenAI] = None
_chroma_collection: Optional[chromadb.Collection] = None


def _get_openai_client() -> AsyncOpenAI:
    """
    Returns the module-level AsyncOpenAI client, creating it on first call.
    Reuses the same HTTP connection pool across all pipeline invocations.
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_collection() -> chromadb.Collection:
    """
    Returns the module-level ChromaDB collection, fetching it on first call.

    Caches the collection object to avoid repeated PersistentClient
    initialization and disk I/O on every request. The collection object
    is safe to reuse across requests — ChromaDB handles thread safety
    internally.

    Raises:
        RuntimeError: If the collection has not been initialized yet
                      (init_vector_store() must be called first, which
                      happens in the FastAPI lifespan handler in main.py).
    """
    global _chroma_collection
    if _chroma_collection is None:
        logger.debug("Fetching ChromaDB collection (first access — caching for reuse)")
        _chroma_collection = get_vector_collection()
        logger.info(
            "ChromaDB collection '%s' cached (%d documents)",
            _chroma_collection.name,
            _chroma_collection.count(),
        )
    return _chroma_collection


def invalidate_collection_cache() -> None:
    """
    Clears the cached ChromaDB collection reference.

    Call this if the collection is re-initialized (e.g., during testing
    or after a data reload). The next call to _get_collection() will
    re-fetch from disk.
    """
    global _chroma_collection
    _chroma_collection = None
    logger.info("ChromaDB collection cache invalidated.")


# ---------------------------------------------------------------------------
# Step 1: Vector Retrieval
# ---------------------------------------------------------------------------


def _query_chromadb_sync(
    question: str,
    n_results: int,
) -> tuple[list[str], list[float], list[dict[str, Any]], Optional[str]]:
    """
    Synchronous ChromaDB query — runs in a thread pool executor.

    Queries the collection using the question text. ChromaDB's
    OpenAIEmbeddingFunction automatically embeds the query text
    using the configured embedding model before performing the
    cosine similarity search.

    Args:
        question:  The user's natural language question.
        n_results: Number of top-k chunks to retrieve.

    Returns:
        (chunks, scores, metadata_list, error_message)
        - On success: (list of chunk texts, list of distances, list of metadata dicts, None)
        - On failure: ([], [], [], error description string)
    """
    try:
        collection = _get_collection()
    except Exception as e:
        error_msg = f"Failed to access ChromaDB collection: {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        return [], [], [], error_msg

    try:
        # Clamp n_results to the actual collection size — ChromaDB raises an error
        # if you request more results than documents in the collection
        actual_count = collection.count()
        effective_n = min(n_results, actual_count)

        if effective_n == 0:
            logger.warning("ChromaDB collection is empty — no chunks to retrieve.")
            return [], [], [], "The document store is empty. No chunks available for retrieval."

        if effective_n < n_results:
            logger.warning(
                "Requested %d chunks but collection only has %d — retrieving %d.",
                n_results,
                actual_count,
                effective_n,
            )

        results = collection.query(
            query_texts=[question],
            n_results=effective_n,
            include=["documents", "distances", "metadatas"],
        )

        # ChromaDB returns nested lists (one per query text); we sent one query
        # so we take index [0] from each result list
        chunks: list[str] = results["documents"][0]
        scores: list[float] = results["distances"][0]
        metadatas: list[dict[str, Any]] = results["metadatas"][0]

        # Defensive check: parallel arrays must have equal lengths
        if not (len(chunks) == len(scores) == len(metadatas)):
            error_msg = (
                f"ChromaDB returned mismatched array lengths: "
                f"chunks={len(chunks)}, scores={len(scores)}, metadatas={len(metadatas)}"
            )
            logger.error(error_msg)
            return [], [], [], error_msg

        if chunks:
            min_score = min(scores)
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            logger.info(
                "ChromaDB retrieved %d chunk(s) | distances: min=%.4f avg=%.4f max=%.4f",
                len(chunks),
                min_score,
                avg_score,
                max_score,
            )
            if min_score > _LOW_RELEVANCE_THRESHOLD:
                logger.warning(
                    "All retrieved chunks have high cosine distance (min=%.4f > %.2f). "
                    "The question may not be well-covered by the document store.",
                    min_score,
                    _LOW_RELEVANCE_THRESHOLD,
                )

        return chunks, scores, metadatas, None

    except Exception as e:
        error_msg = f"ChromaDB query failed: {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        return [], [], [], error_msg


async def _retrieve_chunks(
    question: str,
    n_results: int,
) -> tuple[list[str], list[float], list[dict[str, Any]], Optional[str]]:
    """
    Async wrapper around the synchronous ChromaDB query.

    Runs the blocking ChromaDB call in a thread pool executor to avoid
    blocking the FastAPI event loop.

    Returns:
        (chunks, scores, metadata_list, error_message) — same as _query_chromadb_sync.
    """
    logger.info(
        "Vector retrieval: querying ChromaDB for top-%d chunks | question: %r",
        n_results,
        question[:100],
    )
    start = time.perf_counter()

    loop = asyncio.get_event_loop()
    chunks, scores, metadatas, error = await loop.run_in_executor(
        None,
        _query_chromadb_sync,
        question,
        n_results,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Vector retrieval complete (%.1fms): %d chunk(s) retrieved",
        elapsed_ms,
        len(chunks),
    )

    return chunks, scores, metadatas, error


# ---------------------------------------------------------------------------
# Step 2: Answer Synthesis
# ---------------------------------------------------------------------------


async def _synthesize_answer(
    question: str,
    retrieved_chunks: list[str],
    chunk_scores: list[float],
    chunk_metadata: list[dict[str, Any]],
    retrieval_error: Optional[str] = None,
) -> str:
    """
    Step 2: Calls the LLM to synthesize a natural language answer from
    the retrieved document chunks.

    If retrieval produced an error or returned no chunks, the synthesis
    prompt is augmented with error context so the LLM can return a
    helpful response rather than a generic failure message.

    Returns:
        Synthesized natural language answer string.
    """
    logger.info(
        "Vector synthesis: synthesizing answer from %d chunk(s)%s",
        len(retrieved_chunks),
        f" (with retrieval error: {retrieval_error[:60]})" if retrieval_error else "",
    )
    start = time.perf_counter()

    user_message = build_vector_synthesis_user_message(
        question=question,
        retrieved_chunks=retrieved_chunks,
        chunk_scores=chunk_scores,
        chunk_metadata=chunk_metadata,
    )

    # Append explicit instructions when retrieval errored and returned nothing
    if retrieval_error and not retrieved_chunks:
        user_message += (
            f"\n\nIMPORTANT: The document retrieval system encountered an error: "
            f"{retrieval_error}\n"
            "Please inform the user that the document search is temporarily unavailable "
            "and suggest they try again. Be helpful and apologetic."
        )

    client = _get_openai_client()

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": build_vector_synthesis_system_prompt(),
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
        logger.error("Vector synthesis LLM call failed: %s", e, exc_info=True)
        return _FALLBACK_ANSWER

    elapsed_ms = (time.perf_counter() - start) * 1000
    answer = (response.choices[0].message.content or "").strip()

    if not answer:
        logger.warning("Vector synthesis returned empty answer (%.1fms)", elapsed_ms)
        return _FALLBACK_ANSWER

    logger.info(
        "Vector synthesis complete (%.1fms): %r",
        elapsed_ms,
        answer[:150],
    )
    return answer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_vector_pipeline(question: str) -> dict[str, Any]:
    """
    Executes the full Vector Search pipeline for a natural language question.

    Two-step process:
      1. Query ChromaDB for top-k semantically similar document chunks
      2. LLM synthesizes a natural language answer from the retrieved chunks

    Args:
        question: The user's natural language question (pre-validated by Pydantic).

    Returns:
        A dict with keys: answer, retrieved_chunks, chunk_scores, chunk_metadata, error.
        Matches models.schemas.Artifacts fields exactly.

    Notes:
        - Never raises an exception — all errors are caught and returned in
          the "error" field with a helpful "answer" string.
        - Retrieval failure: returns None for list fields + error + fallback answer.
        - Empty collection: returns error explaining no documents are available.
        - Low-relevance results: returned as-is; synthesis prompt handles gracefully.
        - ChromaDB query runs in a thread pool executor (non-blocking).
    """
    pipeline_start = time.perf_counter()
    logger.info("Vector pipeline starting for question: %r", question[:100])

    # ── Step 1: Retrieve Chunks ──────────────────────────────────────────────
    retrieved_chunks, chunk_scores, chunk_metadata, retrieval_error = (
        await _retrieve_chunks(
            question=question,
            n_results=settings.vector_top_k,
        )
    )

    # ── Step 2: Synthesize Answer ────────────────────────────────────────────
    # Always attempt synthesis — even on retrieval error — so the LLM can
    # provide a helpful explanation to the user
    answer = await _synthesize_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        chunk_scores=chunk_scores,
        chunk_metadata=chunk_metadata,
        retrieval_error=retrieval_error,
    )

    elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info(
        "Vector pipeline complete in %.1fms | chunks=%d | error=%s",
        elapsed_ms,
        len(retrieved_chunks),
        retrieval_error[:60] if retrieval_error else None,
    )

    # Return None for list fields when retrieval failed entirely (no chunks returned)
    # vs. populated lists when retrieval succeeded (even if results are low-relevance)
    if retrieval_error and not retrieved_chunks:
        return {
            "answer": answer,
            "retrieved_chunks": None,
            "chunk_scores": None,
            "chunk_metadata": None,
            "error": retrieval_error,
        }

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "chunk_scores": chunk_scores,
        "chunk_metadata": chunk_metadata,
        "error": retrieval_error,  # None on full success
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
        "What is Project Phoenix and what are its expected outcomes?",
        "What is the strategic rationale for the company restructuring?",
        "What are the EBITDA margin targets set by the board?",
        "What risks did the company identify in the restructuring?",
        "What is the new go-to-market strategy for Enterprise Sales?",
        "What is the company's dividend policy?",
    ]

    async def run_tests(questions: list[str]) -> None:
        from db.vector_init import init_vector_store
        init_vector_store()

        print("\n" + "=" * 70)
        print("VECTOR PIPELINE TEST — calling OpenAI API + ChromaDB")
        print("=" * 70)

        for i, question in enumerate(questions, 1):
            print(f"\n{'─'*70}")
            print(f"[{i}/{len(questions)}] Question: {question}")
            print("─" * 70)

            try:
                result = await execute_vector_pipeline(question)

                chunks = result["retrieved_chunks"] or []
                print(f"Chunks Retrieved: {len(chunks)}")
                if chunks:
                    for j, (chunk, score, meta) in enumerate(
                        zip(chunks, result["chunk_scores"], result["chunk_metadata"]), 1
                    ):
                        section = meta.get("section", "Unknown")
                        print(
                            f"  Chunk {j}: section='{section}' | "
                            f"distance={score:.4f} | "
                            f"preview='{chunk[:80]}...'"
                        )
                if result["error"]:
                    print(f"Error: {result['error']}")
                print(f"\nAnswer:\n{result['answer']}")

            except Exception as e:
                print(f"💥 UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*70}")
        print("Vector Pipeline tests complete.")
        print("=" * 70)

    if len(sys.argv) > 1:
        single_question = " ".join(sys.argv[1:])

        async def run_single() -> None:
            from db.vector_init import init_vector_store
            init_vector_store()

            print(f"\nQuestion: {single_question}\n")
            result = await execute_vector_pipeline(single_question)

            chunks = result["retrieved_chunks"] or []
            print(f"Chunks Retrieved: {len(chunks)}")
            if chunks:
                for j, (chunk, score, meta) in enumerate(
                    zip(chunks, result["chunk_scores"], result["chunk_metadata"]), 1
                ):
                    section = meta.get("section", "Unknown")
                    print(
                        f"  Chunk {j}: section='{section}' | "
                        f"distance={score:.4f} | "
                        f"preview='{chunk[:100]}...'"
                    )
            if result["error"]:
                print(f"Error: {result['error']}")
            print(f"\nAnswer:\n{result['answer']}")

        asyncio.run(run_single())
    else:
        asyncio.run(run_tests(TEST_QUESTIONS))
