
"""
SQLite initialisation — creates the erp_fin_qtr_export table and seeds it
with 8 rows (2 departments × 4 quarters) that support realistic SQL queries
and trend analysis across fiscal years 2024-2025.

Column naming convention mirrors real enterprise ERP exports:
  cryptic name        → human-readable meaning
  ─────────────────────────────────────────────────────────────────────
  dept_id_ref         → Department ID (foreign key reference)
  dept_nm_short       → Department short name / label
  fy_yr               → Fiscal Year (integer, e.g. 2024)
  fq_lbl              → Fiscal Quarter label (e.g. "Q3", "Q4")
  rev_raw_m           → Revenue (raw, in USD millions)
  cogs_raw_m          → Cost of Goods Sold (in USD millions)
  gp_calc_m           → Gross Profit calculated (in USD millions)
  opex_tot_m          → Total Operating Expenses (in USD millions)
  ebitda_adj_m        → Adjusted EBITDA (in USD millions)
  ebitda_adj_pct      → Adjusted EBITDA margin (as decimal, e.g. 0.23)
  hc_fte_end          → Headcount — Full-Time Equivalents at period end
  hc_contract_end     → Headcount — Contractors at period end
  capex_m             → Capital Expenditure (in USD millions)
  restructure_chg_m   → One-time restructuring charges (in USD millions)
  pipeline_val_m      → Sales pipeline value (in USD millions)
  bookings_new_m      → New bookings in period (in USD millions)
  region_cd           → Region code (e.g. "AMER", "EMEA")
  cost_centre_cd      → Cost centre code
  last_updated_ts     → Row last-updated timestamp (ISO-8601)
"""

import sqlite3
import logging
from pathlib import Path
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS erp_fin_qtr_export (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    dept_id_ref         TEXT    NOT NULL,
    dept_nm_short       TEXT    NOT NULL,
    fy_yr               INTEGER NOT NULL,
    fq_lbl              TEXT    NOT NULL,
    rev_raw_m           REAL    NOT NULL,
    cogs_raw_m          REAL    NOT NULL,
    gp_calc_m           REAL    NOT NULL,
    opex_tot_m          REAL    NOT NULL,
    ebitda_adj_m        REAL    NOT NULL,
    ebitda_adj_pct      REAL    NOT NULL,
    hc_fte_end          INTEGER NOT NULL,
    hc_contract_end     INTEGER NOT NULL,
    capex_m             REAL    NOT NULL,
    restructure_chg_m   REAL    NOT NULL,
    pipeline_val_m      REAL    NOT NULL,
    bookings_new_m      REAL    NOT NULL,
    region_cd           TEXT    NOT NULL,
    cost_centre_cd      TEXT    NOT NULL,
    last_updated_ts     TEXT    NOT NULL,
    UNIQUE(dept_id_ref, fy_yr, fq_lbl)
);
"""

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------
# 8 rows: departments D_402 (Cloud Infrastructure) and D_517 (Enterprise Sales)
# across Q3 2024, Q4 2024, Q3 2025, Q4 2025.
#
# Narrative baked into the numbers:
#   • D_402 shows strong EBITDA improvement from Q3→Q4 2025 driven by
#     headcount reduction (Project Phoenix restructuring) and capex discipline.
#   • D_517 shows revenue growth and pipeline expansion in 2025 reflecting
#     the strategic pivot to enterprise accounts described in the memo.
#   • Restructuring charges appear in Q3 2025 for both departments (one-time).
# ---------------------------------------------------------------------------

SEED_ROWS = [
    # ── D_402 Cloud Infrastructure ──────────────────────────────────────────
    # Q3 2024 — baseline year, healthy but flat
    (
        "D_402", "Cloud Infra",
        2024, "Q3",
        42.80,   # rev_raw_m
        18.20,   # cogs_raw_m
        24.60,   # gp_calc_m
        12.40,   # opex_tot_m
        9.80,    # ebitda_adj_m
        0.229,   # ebitda_adj_pct  (22.9%)
        312,     # hc_fte_end
        47,      # hc_contract_end
        3.10,    # capex_m
        0.00,    # restructure_chg_m
        88.50,   # pipeline_val_m
        19.20,   # bookings_new_m
        "AMER",
        "CC-4020",
        "2024-10-15T08:00:00Z",
    ),
    # Q4 2024 — slight seasonal uplift
    (
        "D_402", "Cloud Infra",
        2024, "Q4",
        45.60,
        19.10,
        26.50,
        13.20,
        10.50,
        0.230,   # 23.0%
        318,
        51,
        3.40,
        0.00,
        92.10,
        21.80,
        "AMER",
        "CC-4020",
        "2025-01-15T08:00:00Z",
    ),
    # Q3 2025 — restructuring charges hit; headcount reduced; EBITDA temporarily compressed
    (
        "D_402", "Cloud Infra",
        2025, "Q3",
        47.30,
        18.80,
        28.50,
        14.10,
        8.20,    # ebitda_adj_m — compressed by restructuring opex
        0.173,   # 17.3% — margin dip during restructuring
        289,     # hc_fte_end — headcount reduced (Project Phoenix)
        28,      # contractors cut significantly
        2.60,    # capex reduced
        4.50,    # restructure_chg_m — one-time charge (Project Phoenix)
        96.40,
        23.10,
        "AMER",
        "CC-4020",
        "2025-10-15T08:00:00Z",
    ),
    # Q4 2025 — post-restructuring: leaner cost base, EBITDA margin expands sharply
    (
        "D_402", "Cloud Infra",
        2025, "Q4",
        51.20,
        17.90,   # cogs down — automation savings
        33.30,
        11.80,   # opex down — fewer FTEs
        14.60,   # ebitda_adj_m — strong recovery
        0.285,   # 28.5% — best margin in dataset
        281,     # headcount stable post-restructure
        22,
        2.20,    # capex discipline maintained
        0.00,    # no further restructuring charges
        104.80,
        27.40,
        "AMER",
        "CC-4020",
        "2026-01-15T08:00:00Z",
    ),

    # ── D_517 Enterprise Sales ───────────────────────────────────────────────
    # Q3 2024 — baseline
    (
        "D_517", "Ent Sales",
        2024, "Q3",
        38.10,
        8.40,
        29.70,
        22.30,
        6.20,
        0.163,   # 16.3%
        198,
        34,
        0.80,
        0.00,
        142.60,
        31.50,
        "EMEA",
        "CC-5170",
        "2024-10-15T08:00:00Z",
    ),
    # Q4 2024
    (
        "D_517", "Ent Sales",
        2024, "Q4",
        41.50,
        9.10,
        32.40,
        23.10,
        7.40,
        0.178,   # 17.8%
        204,
        38,
        0.90,
        0.00,
        158.20,
        36.80,
        "EMEA",
        "CC-5170",
        "2025-01-15T08:00:00Z",
    ),
    # Q3 2025 — restructuring charge; team reorganised around enterprise accounts
    (
        "D_517", "Ent Sales",
        2025, "Q3",
        44.90,
        9.60,
        35.30,
        24.80,
        7.10,
        0.158,   # 15.8% — margin dip during transition
        187,     # headcount reduced
        21,
        0.70,
        2.80,    # restructure_chg_m
        178.50,  # pipeline growing — strategic pivot working
        39.20,
        "EMEA",
        "CC-5170",
        "2025-10-15T08:00:00Z",
    ),
    # Q4 2025 — enterprise pivot paying off: revenue up, pipeline surges
    (
        "D_517", "Ent Sales",
        2025, "Q4",
        52.30,
        10.20,
        42.10,
        22.60,   # opex down post-restructure
        13.80,
        0.264,   # 26.4% — significant margin expansion
        183,
        18,
        0.75,
        0.00,
        221.40,  # pipeline nearly 50% higher than Q3 2024
        48.60,
        "EMEA",
        "CC-5170",
        "2026-01-15T08:00:00Z",
    ),
]

INSERT_SQL = """
INSERT OR IGNORE INTO erp_fin_qtr_export (
    dept_id_ref, dept_nm_short,
    fy_yr, fq_lbl,
    rev_raw_m, cogs_raw_m, gp_calc_m, opex_tot_m,
    ebitda_adj_m, ebitda_adj_pct,
    hc_fte_end, hc_contract_end,
    capex_m, restructure_chg_m,
    pipeline_val_m, bookings_new_m,
    region_cd, cost_centre_cd, last_updated_ts
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_sqlite() -> None:
    """
    Idempotent initialisation: creates the table if it doesn't exist and
    inserts seed rows (INSERT OR IGNORE — safe to call on every startup).
    """
    db_path = Path(settings.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising SQLite database at %s", db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")   # better concurrent read perf
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(CREATE_TABLE_SQL)
        conn.executemany(INSERT_SQL, SEED_ROWS)
        conn.commit()

        # Verify
        cursor = conn.execute("SELECT COUNT(*) FROM erp_fin_qtr_export")
        row_count = cursor.fetchone()[0]
        logger.info("erp_fin_qtr_export contains %d rows", row_count)
    finally:
        conn.close()


def get_connection(read_only: bool = True) -> sqlite3.Connection:
    """
    Return a SQLite connection.  When read_only=True (the default) the
    connection is opened via the immutable URI mode — any attempt to write
    will raise an OperationalError, providing a safety guardrail for the
    agent's SQL execution path.
    """
    db_path = Path(settings.database_path).resolve()

    if read_only:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(str(db_path))

    conn.row_factory = sqlite3.Row   # rows accessible as dicts
    return conn


# ---------------------------------------------------------------------------
# Column metadata — used by the SQL pipeline to give the LLM human-readable
# descriptions of every column so it can generate accurate SQL.
# ---------------------------------------------------------------------------

COLUMN_METADATA: dict[str, str] = {
    "id":                 "Auto-increment primary key (ignore in queries)",
    "dept_id_ref":        "Department ID — unique identifier for a department (e.g. 'D_402', 'D_517')",
    "dept_nm_short":      "Department short name / label (e.g. 'Cloud Infra', 'Ent Sales')",
    "fy_yr":              "Fiscal Year as integer (e.g. 2024, 2025)",
    "fq_lbl":             "Fiscal Quarter label (e.g. 'Q3', 'Q4')",
    "rev_raw_m":          "Revenue in USD millions (raw, before adjustments)",
    "cogs_raw_m":         "Cost of Goods Sold in USD millions",
    "gp_calc_m":          "Gross Profit in USD millions (calculated: rev_raw_m - cogs_raw_m)",
    "opex_tot_m":         "Total Operating Expenses in USD millions (excludes COGS)",
    "ebitda_adj_m":       "Adjusted EBITDA in USD millions (earnings before interest, tax, depreciation & amortisation, adjusted for one-time items)",
    "ebitda_adj_pct":     "Adjusted EBITDA margin as a decimal (e.g. 0.285 = 28.5%); multiply by 100 for percentage",
    "hc_fte_end":         "Headcount — number of Full-Time Employees at end of period",
    "hc_contract_end":    "Headcount — number of Contractors at end of period",
    "capex_m":            "Capital Expenditure in USD millions",
    "restructure_chg_m":  "One-time restructuring charges in USD millions (e.g. severance, asset write-downs)",
    "pipeline_val_m":     "Total sales pipeline value in USD millions",
    "bookings_new_m":     "New bookings / signed contracts in USD millions for the period",
    "region_cd":          "Geographic region code (e.g. 'AMER' = Americas, 'EMEA' = Europe/Middle East/Africa)",
    "cost_centre_cd":     "Internal cost centre code for accounting allocation",
    "last_updated_ts":    "Timestamp when this row was last updated (ISO-8601 format)",
}


def get_schema_description() -> str:
    """
    Returns a formatted string describing the table schema with human-readable
    column descriptions.  Injected into the Text-to-SQL prompt.
    """
    lines = [
        "Table: erp_fin_qtr_export",
        "Description: Quarterly financial and operational metrics exported from the ERP system.",
        "",
        "Columns:",
    ]
    for col, desc in COLUMN_METADATA.items():
        lines.append(f"  {col:<22} — {desc}")

    lines += [
        "",
        "Sample department IDs: D_402 (Cloud Infrastructure), D_517 (Enterprise Sales)",
        "Available fiscal years: 2024, 2025",
        "Available fiscal quarters: Q3, Q4",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_sqlite()
    print(get_schema_description())
