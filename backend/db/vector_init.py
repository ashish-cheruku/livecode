"""
ChromaDB initialisation — embeds and stores the Q3 2025 Strategic
Restructuring Memo in three semantically coherent chunks.

Chunk strategy:
  Chunk 1 — Executive Summary & Strategic Rationale
  Chunk 2 — Project Phoenix: Scope, Timeline & Operational Changes
  Chunk 3 — Financial Targets, Risk Factors & Expected Outcomes

Each chunk is stored with rich metadata so the retrieval pipeline can
surface source information alongside the answer.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memo content — Q3 2025 Strategic Restructuring Memo
# ---------------------------------------------------------------------------
# Written to be semantically rich enough to answer qualitative questions
# about strategy, Project Phoenix, expected outcomes, and financial targets.
# The numbers in the memo deliberately align with the SQL seed data so that
# Hybrid queries produce coherent, cross-referenced answers.
# ---------------------------------------------------------------------------

MEMO_CHUNKS: list[dict[str, Any]] = [
    {
        "id": "memo_chunk_001",
        "text": """
MEMORANDUM

TO:      All Department Heads, Finance Leadership, Board of Directors
FROM:    Office of the Chief Executive Officer
DATE:    September 12, 2025
RE:      Q3 2025 Strategic Restructuring Initiative — Executive Summary & Rationale

EXECUTIVE SUMMARY

This memorandum formally communicates the Board-approved Strategic Restructuring
Initiative for fiscal year 2025, effective immediately. The initiative, internally
designated as "Project Phoenix," represents the most significant organisational
transformation undertaken by the company in the past decade.

STRATEGIC RATIONALE

Following an extensive 14-week diagnostic conducted by our internal Strategy Office
in partnership with external advisors, the Board concluded that the company's current
operating model carries structural inefficiencies that are suppressing EBITDA margins
below peer-group benchmarks. Specifically:

1. Headcount density in technology and infrastructure functions (notably the Cloud
   Infrastructure division, department D_402) has grown 34% over 24 months without
   a commensurate increase in revenue productivity per FTE. The division's contractor
   workforce, which peaked at 51 contractors in Q4 2024, represents a disproportionate
   cost relative to the value delivered.

2. The Enterprise Sales organisation (department D_517) has historically pursued a
   broad, volume-based go-to-market strategy that has resulted in high customer
   acquisition costs and low average contract values. The strategic pivot to a focused
   enterprise-account model is expected to improve bookings quality and reduce the
   cost of sale.

3. Capital expenditure across both divisions has lacked rigorous prioritisation
   frameworks, resulting in capex commitments that do not align with the company's
   three-year technology roadmap.

The Board is confident that Project Phoenix will restore the company's competitive
positioning and deliver sustainable margin expansion. All department heads are
expected to cascade this communication to their direct reports within 48 hours.
        """.strip(),
        "metadata": {
            "source": "Q3_2025_Strategic_Restructuring_Memo",
            "chunk_id": "001",
            "section": "Executive Summary and Strategic Rationale",
            "date": "2025-09-12",
            "author": "Office of the CEO",
            "departments_referenced": "D_402, D_517",
        },
    },
    {
        "id": "memo_chunk_002",
        "text": """
PROJECT PHOENIX — SCOPE, TIMELINE & OPERATIONAL CHANGES

PROJECT OVERVIEW

Project Phoenix is a structured, phased programme designed to reduce the company's
cost base, sharpen strategic focus, and position the organisation for accelerated
growth in fiscal year 2026 and beyond. The project is sponsored by the CEO and
overseen by a dedicated Programme Management Office (PMO) reporting directly to
the CFO.

SCOPE OF CHANGES

Phase 1 — Workforce Optimisation (Q3 2025, effective October 1, 2025):
  • Cloud Infrastructure (D_402): Reduction of 29 full-time positions and 23
    contractor roles. Affected roles are primarily in legacy infrastructure
    maintenance and manual operations functions that will be replaced by
    automated tooling. Total one-time restructuring charge for D_402: $4.5M
    (severance, outplacement, and asset write-downs).
  • Enterprise Sales (D_517): Reduction of 17 full-time positions. The
    reorganisation consolidates regional sales pods into four global enterprise
    account teams, each aligned to a vertical market (Financial Services,
    Healthcare, Manufacturing, Public Sector). Total one-time restructuring
    charge for D_517: $2.8M.

Phase 2 — Capex Rationalisation (Q4 2025):
  • All capital expenditure requests above $500K require CFO sign-off.
  • D_402 capex budget reduced from $3.4M (Q4 2024 run-rate) to a target of
    $2.2M per quarter, achieved through deferral of non-critical infrastructure
    refresh cycles and renegotiation of three vendor contracts.

Phase 3 — Go-to-Market Transformation (Q4 2025 – Q2 2026):
  • D_517 will transition entirely to an enterprise-account model by end of Q2 2026.
  • Minimum deal size threshold raised to $500K ACV (Annual Contract Value).
  • Sales compensation plans restructured to reward multi-year enterprise contracts
    over transactional volume.
  • Target: pipeline value to exceed $200M by Q4 2025, up from $142.6M in Q3 2024.

TIMELINE SUMMARY
  Q3 2025: Restructuring charges recognised; workforce changes executed
  Q4 2025: Capex targets achieved; enterprise sales model operational
  Q1 2026: First full quarter of post-restructuring run-rate; margin targets active
  Q2 2026: Go-to-market transformation complete; pipeline quality review
        """.strip(),
        "metadata": {
            "source": "Q3_2025_Strategic_Restructuring_Memo",
            "chunk_id": "002",
            "section": "Project Phoenix Scope, Timeline and Operational Changes",
            "date": "2025-09-12",
            "author": "Office of the CEO",
            "departments_referenced": "D_402, D_517",
            "project_name": "Project Phoenix",
        },
    },
    {
        "id": "memo_chunk_003",
        "text": """
FINANCIAL TARGETS, RISK FACTORS & EXPECTED OUTCOMES

FINANCIAL TARGETS

The Board has approved the following financial targets as success criteria for
Project Phoenix. These targets are binding for departmental performance reviews
and executive compensation purposes.

EBITDA Margin Targets (post-restructuring):
  • Cloud Infrastructure (D_402): EBITDA margin to reach 28%+ by Q4 2025
    (up from 17.3% in Q3 2025 during the restructuring transition period,
    and from a pre-restructuring baseline of ~23% in FY2024).
  • Enterprise Sales (D_517): EBITDA margin to reach 25%+ by Q4 2025
    (up from 15.8% in Q3 2025).
  • Blended company EBITDA margin target for FY2026: 27%.

Revenue & Pipeline Targets:
  • D_402 quarterly revenue to exceed $50M by Q4 2025.
  • D_517 new bookings to exceed $45M per quarter by Q4 2025.
  • Total sales pipeline across both divisions to exceed $300M by Q2 2026.

Headcount Targets:
  • D_402: Stabilise at approximately 280 FTEs post-restructuring.
  • D_517: Stabilise at approximately 185 FTEs post-restructuring.
  • Combined contractor workforce not to exceed 45 across both divisions.

RISK FACTORS

1. Talent Retention Risk: The workforce reduction may create uncertainty among
   high-performing employees not directly affected by the restructuring. HR is
   implementing a retention bonus programme for identified critical talent.

2. Revenue Disruption Risk: The D_517 go-to-market pivot may cause a temporary
   slowdown in new bookings during Q3-Q4 2025 as the sales team transitions.
   A revenue bridge plan has been approved to manage this risk.

3. Execution Risk: The PMO has identified 12 interdependencies between Phase 1
   and Phase 2 workstreams. Weekly steering committee reviews are mandated.

EXPECTED OUTCOMES

Upon successful completion of Project Phoenix, the company expects to:
  • Achieve annualised cost savings of $18M–$22M by end of FY2026.
  • Improve blended EBITDA margins from ~22% (FY2024 baseline) to 27%+ (FY2026 target).
  • Increase average enterprise deal size by 40% through the D_517 go-to-market pivot.
  • Reduce combined contractor dependency by 60% versus Q4 2024 peak levels.
  • Position the company for a potential strategic transaction or IPO readiness
    review in H2 2026, subject to market conditions.

This memorandum is classified CONFIDENTIAL — BOARD AND EXECUTIVE DISTRIBUTION ONLY.
Questions should be directed to the CFO's office or the Project Phoenix PMO.
        """.strip(),
        "metadata": {
            "source": "Q3_2025_Strategic_Restructuring_Memo",
            "chunk_id": "003",
            "section": "Financial Targets, Risk Factors and Expected Outcomes",
            "date": "2025-09-12",
            "author": "Office of the CEO",
            "departments_referenced": "D_402, D_517",
            "project_name": "Project Phoenix",
        },
    },
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_vector_store() -> chromadb.Collection:
    """
    Idempotent initialisation of ChromaDB.

    - Creates (or loads) a persistent ChromaDB client at CHROMA_PERSIST_DIR.
    - Creates (or loads) the collection named CHROMA_COLLECTION_NAME.
    - Uses OpenAI's text-embedding-3-small for embeddings.
    - Inserts memo chunks only if the collection is empty (avoids duplicates
      on repeated startups without needing to delete the collection).

    Returns the ChromaDB Collection object for use by the vector pipeline.
    """
    logger.info(
        "Initialising ChromaDB at %s (collection: %s)",
        settings.chroma_persist_dir,
        settings.chroma_collection_name,
    )

    # Persistent client — data survives restarts
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))

    # OpenAI embedding function — ChromaDB calls this automatically on add/query
    openai_ef = OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )

    existing_count = collection.count()
    logger.info(
        "Collection '%s' currently has %d documents",
        settings.chroma_collection_name,
        existing_count,
    )

    if existing_count == 0:
        logger.info("Inserting %d memo chunks into ChromaDB...", len(MEMO_CHUNKS))
        collection.add(
            ids=[chunk["id"] for chunk in MEMO_CHUNKS],
            documents=[chunk["text"] for chunk in MEMO_CHUNKS],
            metadatas=[chunk["metadata"] for chunk in MEMO_CHUNKS],
        )
        logger.info(
            "Successfully inserted %d chunks. Collection now has %d documents.",
            len(MEMO_CHUNKS),
            collection.count(),
        )
    else:
        logger.info(
            "Collection already populated (%d docs) — skipping insertion.",
            existing_count,
        )

    return collection


def get_vector_collection() -> chromadb.Collection:
    """
    Returns the ChromaDB collection without re-inserting data.
    Use this in the query pipeline after init_vector_store() has been called.
    Raises RuntimeError if the collection doesn't exist yet.
    """
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))

    openai_ef = OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )

    # get_collection raises InvalidCollectionException if not found
    collection = client.get_collection(
        name=settings.chroma_collection_name,
        embedding_function=openai_ef,
    )
    return collection


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    col = init_vector_store()
    print(f"\nCollection '{col.name}' ready with {col.count()} documents.")

    # Quick sanity-check query
    results = col.query(
        query_texts=["What is Project Phoenix?"],
        n_results=2,
    )
    print("\nSanity-check query: 'What is Project Phoenix?'")
    for i, doc in enumerate(results["documents"][0]):
        score = results["distances"][0][i]
        print(f"\n  Chunk {i+1} (distance={score:.4f}):")
        print(f"  {doc[:200]}...")
