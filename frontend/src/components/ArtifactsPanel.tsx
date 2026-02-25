/**
 * ArtifactsPanel — collapsible panel showing intermediate pipeline artifacts.
 *
 * Renders different sections based on which pipeline was used:
 *   SQL    → Generated SQL query (code block) + results table + row count
 *   VECTOR → Retrieved document chunks with relevance scores + metadata
 *   HYBRID → Both SQL section AND Vector section
 *
 * All sections handle null/empty data gracefully.
 * Returns null if there are no artifacts worth displaying.
 */

import React, { useState, useCallback } from "react";
import type { Artifacts, PipelineType } from "../types";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ArtifactsPanelProps {
  artifacts: Artifacts;
  pipeline: PipelineType;
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/**
 * Maps a cosine distance score to a human-readable relevance label and color.
 * ChromaDB cosine distances: 0 = identical, 1 = orthogonal, 2 = opposite.
 * In practice for well-formed embeddings: < 0.3 = high, 0.3–0.5 = medium, > 0.5 = low.
 */
function getRelevanceInfo(score: number): {
  label: string;
  dotClass: string;
  textClass: string;
} {
  if (score < 0.3) {
    return {
      label: "High",
      dotClass: "bg-green-500",
      textClass: "text-green-600",
    };
  }
  if (score < 0.5) {
    return {
      label: "Medium",
      dotClass: "bg-yellow-500",
      textClass: "text-yellow-600",
    };
  }
  return {
    label: "Low",
    dotClass: "bg-red-400",
    textClass: "text-red-500",
  };
}

/**
 * Formats a cell value from a SQL result row for display.
 * Handles null, numbers, booleans, and strings.
 */
function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    // Format floats to 4 decimal places max, integers as-is
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

// ── Section header ──────────────────────────────────────────────────────────

const SectionHeader: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => (
  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
    {children}
  </p>
);

// ── SQL Query code block ─────────────────────────────────────────────────────

interface SqlQueryBlockProps {
  sql: string;
  rowCount: number | null;
}

const SqlQueryBlock: React.FC<SqlQueryBlockProps> = ({ sql, rowCount }) => (
  <div>
    <SectionHeader>Generated SQL</SectionHeader>
    <div className="relative group">
      <pre
        className="
          text-xs text-gray-800 bg-gray-950 text-green-400
          rounded-lg p-3 overflow-x-auto whitespace-pre-wrap break-words
          font-mono leading-relaxed border border-gray-800
          max-h-48 overflow-y-auto
        "
        aria-label="Generated SQL query"
      >
        {sql}
      </pre>
    </div>
    {rowCount !== null && (
      <p className="text-xs text-gray-400 mt-1.5 flex items-center gap-1">
        <svg
          className="w-3 h-3"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
          />
        </svg>
        {rowCount} {rowCount === 1 ? "row" : "rows"} returned
      </p>
    )}
  </div>
);

// ── SQL Results table ────────────────────────────────────────────────────────

interface SqlResultsTableProps {
  results: Record<string, unknown>[];
}

const SqlResultsTable: React.FC<SqlResultsTableProps> = ({ results }) => {
  if (results.length === 0) {
    return (
      <p className="text-xs text-gray-400 italic py-1">
        Query returned no rows.
      </p>
    );
  }

  // Derive column headers from the first row's keys
  const columns = Object.keys(results[0]);

  return (
    <div>
      <SectionHeader>
        Query Results ({results.length} {results.length === 1 ? "row" : "rows"})
      </SectionHeader>
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-xs border-collapse min-w-max">
          <thead>
            <tr className="bg-gray-100 border-b border-gray-200">
              {columns.map((col) => (
                <th
                  key={col}
                  className="px-3 py-2 text-left font-semibold text-gray-600 whitespace-nowrap"
                  title={col}
                >
                  {/* Show a shortened version of cryptic column names */}
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.map((row, rowIdx) => (
              <tr
                key={rowIdx}
                className={`
                  border-b border-gray-100 last:border-0
                  ${rowIdx % 2 === 0 ? "bg-white" : "bg-gray-50"}
                  hover:bg-blue-50 transition-colors duration-75
                `}
              >
                {columns.map((col) => (
                  <td
                    key={col}
                    className="px-3 py-2 text-gray-700 whitespace-nowrap font-mono"
                  >
                    {formatCellValue(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ── Individual chunk card ────────────────────────────────────────────────────

interface ChunkCardProps {
  chunk: string;
  score: number | undefined;
  metadata: Record<string, string> | undefined;
  index: number;
}

const CHUNK_PREVIEW_LENGTH = 300;

const ChunkCard: React.FC<ChunkCardProps> = ({
  chunk,
  score,
  metadata,
  index,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const isLong = chunk.length > CHUNK_PREVIEW_LENGTH;
  const displayText =
    isLong && !isExpanded ? chunk.slice(0, CHUNK_PREVIEW_LENGTH) + "…" : chunk;

  const relevance =
    score !== undefined ? getRelevanceInfo(score) : null;

  const section = metadata?.section ?? `Chunk ${index + 1}`;
  const source = metadata?.source ?? "";

  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
      {/* Chunk header */}
      <div className="flex items-start justify-between gap-2 px-3 py-2 bg-gray-50 border-b border-gray-100">
        <div className="flex-1 min-w-0">
          <p
            className="text-xs font-semibold text-gray-700 truncate"
            title={section}
          >
            {section}
          </p>
          {source && (
            <p className="text-xs text-gray-400 truncate mt-0.5" title={source}>
              {source.replace(/_/g, " ")}
            </p>
          )}
        </div>

        {/* Relevance indicator */}
        {relevance && (
          <div
            className="flex items-center gap-1 shrink-0"
            title={`Cosine distance: ${score?.toFixed(4)} — ${relevance.label} relevance`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${relevance.dotClass}`}
              aria-hidden="true"
            />
            <span className={`text-xs font-medium ${relevance.textClass}`}>
              {relevance.label}
            </span>
            <span className="text-xs text-gray-400 tabular-nums">
              ({score?.toFixed(3)})
            </span>
          </div>
        )}
      </div>

      {/* Chunk text */}
      <div className="px-3 py-2">
        <p className="text-xs text-gray-600 leading-relaxed whitespace-pre-wrap break-words">
          {displayText}
        </p>

        {isLong && (
          <button
            type="button"
            onClick={() => setIsExpanded((prev) => !prev)}
            className="mt-1.5 text-xs text-blue-600 hover:text-blue-800 font-medium focus:outline-none focus:underline transition-colors"
            aria-expanded={isExpanded}
          >
            {isExpanded ? "Show less ↑" : "Show more ↓"}
          </button>
        )}
      </div>
    </div>
  );
};

// ── SQL artifacts section ────────────────────────────────────────────────────

interface SqlSectionProps {
  artifacts: Artifacts;
}

const SqlSection: React.FC<SqlSectionProps> = ({ artifacts }) => {
  const hasSql = Boolean(artifacts.sql_query);
  const hasResults =
    artifacts.sql_results !== null && artifacts.sql_results !== undefined;

  if (!hasSql && !hasResults) return null;

  return (
    <div className="space-y-4">
      {hasSql && (
        <SqlQueryBlock
          sql={artifacts.sql_query!}
          rowCount={artifacts.sql_row_count ?? null}
        />
      )}
      {hasResults && artifacts.sql_results!.length > 0 && (
        <SqlResultsTable results={artifacts.sql_results!} />
      )}
      {hasResults && artifacts.sql_results!.length === 0 && hasSql && (
        <p className="text-xs text-gray-400 italic">
          The query executed successfully but returned no rows.
        </p>
      )}
    </div>
  );
};

// ── Vector artifacts section ─────────────────────────────────────────────────

interface VectorSectionProps {
  artifacts: Artifacts;
}

const VectorSection: React.FC<VectorSectionProps> = ({ artifacts }) => {
  const chunks = artifacts.retrieved_chunks;
  if (!chunks || chunks.length === 0) return null;

  return (
    <div>
      <SectionHeader>
        Retrieved Document Chunks ({chunks.length})
      </SectionHeader>
      <div className="space-y-2">
        {chunks.map((chunk, i) => (
          <ChunkCard
            key={i}
            chunk={chunk}
            score={artifacts.chunk_scores?.[i]}
            metadata={artifacts.chunk_metadata?.[i]}
            index={i}
          />
        ))}
      </div>
    </div>
  );
};

// ── Divider ──────────────────────────────────────────────────────────────────

const Divider: React.FC = () => (
  <div className="border-t border-gray-100" aria-hidden="true" />
);

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const ArtifactsPanel: React.FC<ArtifactsPanelProps> = ({
  artifacts,
  pipeline,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleOpen = useCallback(() => setIsOpen((prev) => !prev), []);

  // Determine what's available to show
  const hasSqlArtifacts = Boolean(artifacts.sql_query);
  const hasVectorArtifacts = Boolean(
    artifacts.retrieved_chunks && artifacts.retrieved_chunks.length > 0
  );

  // Nothing to show — render nothing
  if (!hasSqlArtifacts && !hasVectorArtifacts) {
    return null;
  }

  // Build a summary label for the toggle button
  const summaryParts: string[] = [];
  if (hasSqlArtifacts) {
    const rowCount = artifacts.sql_row_count;
    summaryParts.push(
      rowCount !== null ? `${rowCount} SQL row${rowCount !== 1 ? "s" : ""}` : "SQL query"
    );
  }
  if (hasVectorArtifacts) {
    const chunkCount = artifacts.retrieved_chunks!.length;
    summaryParts.push(`${chunkCount} chunk${chunkCount !== 1 ? "s" : ""}`);
  }
  const summaryLabel = summaryParts.join(" · ");

  return (
    <div
      className="rounded-xl border border-gray-200 overflow-hidden bg-white"
      aria-label="Pipeline artifacts"
    >
      {/* ── Toggle button ──────────────────────────────────────────────── */}
      <button
        type="button"
        onClick={toggleOpen}
        className="
          w-full flex items-center justify-between
          px-3 py-2.5
          bg-gray-50 hover:bg-gray-100
          transition-colors duration-100
          text-gray-600 text-xs font-medium
          focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500
        "
        aria-expanded={isOpen}
        aria-controls="artifacts-panel-content"
      >
        {/* Left side: icon + label */}
        <span className="flex items-center gap-2">
          {/* Info icon */}
          <svg
            className="w-3.5 h-3.5 text-gray-400 shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z"
            />
          </svg>
          <span>
            {isOpen ? "Hide" : "Show"} details
            {summaryLabel && (
              <span className="ml-1 text-gray-400 font-normal">
                · {summaryLabel}
              </span>
            )}
          </span>
        </span>

        {/* Right side: chevron */}
        <svg
          className={`
            w-3.5 h-3.5 text-gray-400 shrink-0
            transition-transform duration-200
            ${isOpen ? "rotate-180" : "rotate-0"}
          `}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2.5}
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19.5 8.25l-7.5 7.5-7.5-7.5"
          />
        </svg>
      </button>

      {/* ── Collapsible content ─────────────────────────────────────────── */}
      {isOpen && (
        <div
          id="artifacts-panel-content"
          className="divide-y divide-gray-100"
          role="region"
          aria-label="Artifact details"
        >
          {/* SQL section */}
          {hasSqlArtifacts && (
            <div className="px-3 py-3">
              <SqlSection artifacts={artifacts} />
            </div>
          )}

          {/* Divider between SQL and Vector sections in Hybrid mode */}
          {hasSqlArtifacts && hasVectorArtifacts && <Divider />}

          {/* Vector section */}
          {hasVectorArtifacts && (
            <div className="px-3 py-3">
              <VectorSection artifacts={artifacts} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ArtifactsPanel;
