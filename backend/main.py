"""
FastAPI Application — Enterprise Hybrid Data Agent

Entry point for the backend API. Wires together:
  - FastAPI app with CORS middleware
  - Lifespan handler: initializes SQLite and ChromaDB on startup
  - POST /api/query  — routes question to SQL/VECTOR/HYBRID pipeline
  - GET  /api/health — per-component health check

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import logging.config
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.hybrid_pipeline import execute_hybrid_pipeline
from agent.router import route_query
from agent.sql_pipeline import execute_sql_pipeline
from agent.vector_pipeline import execute_vector_pipeline, invalidate_collection_cache
from config import settings
from db.sqlite_init import get_connection, init_sqlite
from db.vector_init import init_vector_store
from models.schemas import Artifacts, HealthResponse, PipelineType, QueryRequest, QueryResponse

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """
    Configure structured logging for the application.
    Sets the root logger to INFO and quiets noisy third-party libraries.
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"],
            },
            "loggers": {
                # Quiet noisy libraries
                "httpx": {"level": "WARNING"},
                "httpcore": {"level": "WARNING"},
                "openai": {"level": "WARNING"},
                "chromadb": {"level": "WARNING"},
                "chromadb.telemetry": {"level": "ERROR"},
                "urllib3": {"level": "WARNING"},
                "uvicorn.access": {"level": "INFO"},
            },
        }
    )


_configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan handler
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup:
      1. Initialize SQLite database (create table + seed data if needed)
      2. Initialize ChromaDB vector store (embed + store memo chunks if needed)
      3. Store initialization status in app.state for health checks

    Shutdown:
      - Log shutdown message (ChromaDB PersistentClient flushes automatically)
    """
    logger.info("=" * 60)
    logger.info("Enterprise Hybrid Agent — starting up")
    logger.info("=" * 60)

    # Track per-component initialization status for health checks
    app.state.sqlite_ok = False
    app.state.chromadb_ok = False
    app.state.startup_errors: dict[str, str] = {}

    # ── SQLite initialization ────────────────────────────────────────────────
    try:
        logger.info("Initializing SQLite database at: %s", settings.database_path)
        init_sqlite()
        app.state.sqlite_ok = True
        logger.info("✅ SQLite initialized successfully")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        app.state.startup_errors["sqlite"] = error_msg
        logger.error("❌ SQLite initialization failed: %s", error_msg, exc_info=True)
        # Don't re-raise — allow the app to start in degraded mode so the
        # health endpoint can report the failure

    # ── ChromaDB initialization ──────────────────────────────────────────────
    try:
        logger.info(
            "Initializing ChromaDB at: %s (collection: %s)",
            settings.chroma_persist_dir,
            settings.chroma_collection_name,
        )
        init_vector_store()
        # Invalidate the vector pipeline's collection cache so it re-fetches
        # the freshly initialized collection on first use
        invalidate_collection_cache()
        app.state.chromadb_ok = True
        logger.info("✅ ChromaDB initialized successfully")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        app.state.startup_errors["chromadb"] = error_msg
        logger.error("❌ ChromaDB initialization failed: %s", error_msg, exc_info=True)

    # ── Startup summary ──────────────────────────────────────────────────────
    if app.state.sqlite_ok and app.state.chromadb_ok:
        logger.info("=" * 60)
        logger.info("🚀 All components initialized — agent is ready")
        logger.info("   API: http://%s:%d", settings.api_host, settings.api_port)
        logger.info("   Docs: http://%s:%d/docs", settings.api_host, settings.api_port)
        logger.info("=" * 60)
    else:
        logger.warning(
            "⚠️  Agent started in DEGRADED mode. Failed components: %s",
            list(app.state.startup_errors.keys()),
        )

    # ── Hand control to the application ─────────────────────────────────────
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Enterprise Hybrid Agent — shutting down")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise Hybrid Data Agent",
    description=(
        "An intelligent agent that answers natural language questions about "
        "enterprise financial data using SQL, vector search, or a hybrid of both. "
        "Powered by GPT-4o-mini, SQLite, and ChromaDB."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# ---------------------------------------------------------------------------
# Pipeline dispatcher
# ---------------------------------------------------------------------------

# Maps route strings to their pipeline executor functions.
# Using a dict avoids a long if/elif chain and makes adding new pipelines trivial.
_PIPELINE_EXECUTORS = {
    "SQL": execute_sql_pipeline,
    "VECTOR": execute_vector_pipeline,
    "HYBRID": execute_hybrid_pipeline,
}

# Maps route strings to PipelineType enum values for the response model.
_ROUTE_TO_PIPELINE_TYPE = {
    "SQL": PipelineType.SQL,
    "VECTOR": PipelineType.VECTOR,
    "HYBRID": PipelineType.HYBRID,
}


def _build_artifacts(pipeline_result: dict[str, Any]) -> Artifacts:
    """
    Assembles an Artifacts Pydantic model from a pipeline result dict.

    All pipeline functions return dicts with a superset of the Artifacts fields.
    This function extracts only the Artifacts fields, letting Pydantic handle
    validation and None-defaulting for fields not present in the result.

    Args:
        pipeline_result: Dict returned by any of the three pipeline executors.

    Returns:
        Validated Artifacts model instance.
    """
    return Artifacts(
        sql_query=pipeline_result.get("sql_query"),
        sql_results=pipeline_result.get("sql_results"),
        sql_row_count=pipeline_result.get("sql_row_count"),
        retrieved_chunks=pipeline_result.get("retrieved_chunks"),
        chunk_scores=pipeline_result.get("chunk_scores"),
        chunk_metadata=pipeline_result.get("chunk_metadata"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Submit a natural language question to the agent",
    description=(
        "Accepts a natural language question, routes it to the appropriate "
        "pipeline (SQL, VECTOR, or HYBRID), and returns a synthesized answer "
        "along with intermediate artifacts (generated SQL, retrieved document "
        "chunks, etc.) for transparency."
    ),
    responses={
        200: {"description": "Successful response with answer and artifacts"},
        422: {"description": "Validation error — question is blank or too long"},
        500: {"description": "Internal server error — returned as structured QueryResponse with error field"},
    },
)
async def query_endpoint(request_body: QueryRequest, request: Request) -> QueryResponse:
    """
    Main query endpoint — the heart of the agent.

    Flow:
      1. Validate request (handled by Pydantic QueryRequest)
      2. Call route_query() to classify the question (LLM call #1)
      3. Dispatch to the appropriate pipeline executor
      4. Assemble QueryResponse with answer, pipeline type, artifacts, and timing
      5. Return response

    Error handling:
      - Router failure: defaults to HYBRID pipeline (safest fallback)
      - Pipeline failure: each pipeline handles its own errors internally;
        the "error" field in the response carries any error message
      - Unexpected exception: caught here, returned as a structured error response
        (never a raw 500 with an HTML body)
    """
    question = request_body.question
    endpoint_start = time.perf_counter()

    logger.info(
        "POST /api/query | question=%r | client=%s",
        question[:100],
        request.client.host if request.client else "unknown",
    )

    # ── Step 1: Route the question ───────────────────────────────────────────
    route = "HYBRID"  # safe default if routing fails
    try:
        route = await route_query(question)
        logger.info("Router decision: %r → pipeline=%s", question[:60], route)
    except Exception as e:
        logger.error(
            "Router failed for question %r: %s — defaulting to HYBRID",
            question[:60],
            e,
            exc_info=True,
        )
        # Don't re-raise — fall through to HYBRID pipeline with the default route

    # ── Step 2: Dispatch to pipeline ─────────────────────────────────────────
    pipeline_type = _ROUTE_TO_PIPELINE_TYPE.get(route, PipelineType.HYBRID)
    executor = _PIPELINE_EXECUTORS.get(route, execute_hybrid_pipeline)

    logger.info("Dispatching to %s pipeline", route)

    try:
        pipeline_result = await executor(question)
    except Exception as e:
        # This should never happen — all pipelines catch their own errors.
        # But if it does, return a structured error response rather than a raw 500.
        elapsed_ms = (time.perf_counter() - endpoint_start) * 1000
        error_msg = f"Pipeline execution failed unexpectedly: {type(e).__name__}: {e}"
        logger.error(
            "Unexpected pipeline exception for question %r: %s",
            question[:60],
            error_msg,
            exc_info=True,
        )
        return QueryResponse(
            answer=(
                "I encountered an unexpected error while processing your question. "
                "Please try again. If the issue persists, try rephrasing your question."
            ),
            pipeline=pipeline_type,
            artifacts=Artifacts(),
            processing_time_ms=round(elapsed_ms, 1),
            error=error_msg,
        )

    # ── Step 3: Assemble response ────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - endpoint_start) * 1000

    artifacts = _build_artifacts(pipeline_result)
    answer = pipeline_result.get("answer", "")
    pipeline_error = pipeline_result.get("error")

    # Ensure we always return a non-empty answer string
    if not answer or not answer.strip():
        answer = (
            "I was unable to generate an answer for your question. "
            "Please try rephrasing it."
        )

    response = QueryResponse(
        answer=answer,
        pipeline=pipeline_type,
        artifacts=artifacts,
        processing_time_ms=round(elapsed_ms, 1),
        error=pipeline_error,
    )

    logger.info(
        "POST /api/query complete | pipeline=%s | time=%.1fms | error=%s",
        route,
        elapsed_ms,
        pipeline_error[:60] if pipeline_error else None,
    )

    return response


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns the overall service status and per-component readiness. "
        "Probes SQLite with a lightweight SELECT and ChromaDB with a count query. "
        "Returns 'ok' when all components are healthy, 'degraded' otherwise."
    ),
)
async def health_endpoint(request: Request) -> HealthResponse:
    """
    Health check endpoint.

    Probes each component with a lightweight operation:
      - SQLite: executes SELECT 1 against the database
      - ChromaDB: calls collection.count() to verify the collection is accessible
      - OpenAI: checks that the API key is configured (does not make an API call)

    Returns:
      - status: "ok" if all components are healthy, "degraded" if any are not
      - components: per-component status dict
      - version: API version string
    """
    components: dict[str, str] = {}

    # ── SQLite health check ──────────────────────────────────────────────────
    sqlite_startup_ok = getattr(request.app.state, "sqlite_ok", False)
    if not sqlite_startup_ok:
        startup_error = getattr(request.app.state, "startup_errors", {}).get("sqlite", "initialization failed")
        components["sqlite"] = f"unavailable: {startup_error[:80]}"
    else:
        try:
            conn = get_connection(read_only=True)
            conn.execute("SELECT 1").fetchone()
            # Also verify the main table exists and has data
            row_count = conn.execute(
                "SELECT COUNT(*) FROM erp_fin_qtr_export"
            ).fetchone()[0]
            conn.close()
            components["sqlite"] = f"ok ({row_count} rows in erp_fin_qtr_export)"
        except sqlite3.Error as e:
            components["sqlite"] = f"degraded: {type(e).__name__}: {e}"
            logger.warning("Health check: SQLite probe failed: %s", e)
        except Exception as e:
            components["sqlite"] = f"degraded: {type(e).__name__}: {e}"
            logger.warning("Health check: SQLite probe failed unexpectedly: %s", e)

    # ── ChromaDB health check ────────────────────────────────────────────────
    chromadb_startup_ok = getattr(request.app.state, "chromadb_ok", False)
    if not chromadb_startup_ok:
        startup_error = getattr(request.app.state, "startup_errors", {}).get("chromadb", "initialization failed")
        components["chromadb"] = f"unavailable: {startup_error[:80]}"
    else:
        try:
            from db.vector_init import get_vector_collection
            collection = get_vector_collection()
            doc_count = collection.count()
            components["chromadb"] = f"ok ({doc_count} documents in '{collection.name}')"
        except Exception as e:
            components["chromadb"] = f"degraded: {type(e).__name__}: {e}"
            logger.warning("Health check: ChromaDB probe failed: %s", e)

    # ── OpenAI health check (config only — no API call) ──────────────────────
    api_key = settings.openai_api_key
    if not api_key or api_key == "sk-your-key-here" or len(api_key) < 20:
        components["openai"] = "unavailable: API key not configured"
    else:
        # Mask the key for logging — show first 7 chars only
        masked_key = api_key[:7] + "..." + api_key[-4:]
        components["openai"] = f"ok (key configured: {masked_key}, model: {settings.llm_model})"

    # ── Determine overall status ─────────────────────────────────────────────
    all_ok = all(v.startswith("ok") for v in components.values())
    overall_status = "ok" if all_ok else "degraded"

    logger.debug(
        "GET /api/health | status=%s | components=%s",
        overall_status,
        {k: v[:40] for k, v in components.items()},
    )

    return HealthResponse(
        status=overall_status,
        components=components,
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Global exception handler — last resort for truly unexpected errors
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catches any unhandled exception that escapes the endpoint handlers.

    Returns a JSON response (not HTML) so the frontend always receives
    a parseable error body, even in catastrophic failure scenarios.
    """
    logger.error(
        "Unhandled exception on %s %s: %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred. Please try again.",
            "error_type": type(exc).__name__,
        },
    )


# ---------------------------------------------------------------------------
# Root redirect — convenience for browser navigation
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Redirects browser visitors to the API docs."""
    return JSONResponse(
        content={
            "message": "Enterprise Hybrid Data Agent API",
            "docs": "/docs",
            "health": "/api/health",
            "query": "POST /api/query",
        }
    )


# ---------------------------------------------------------------------------
# Development server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
