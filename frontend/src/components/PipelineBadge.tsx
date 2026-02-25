/**
 * PipelineBadge — color-coded badge indicating which retrieval pipeline
 * was used to answer the query.
 *
 * SQL    → Blue  🟦  (database icon)
 * VECTOR → Green 🟩  (document/search icon)
 * HYBRID → Amber 🟨  (merge/arrows icon)
 *
 * Used inside MessageBubble for every agent response.
 */

import React from "react";
import type { PipelineType } from "../types";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface PipelineBadgeProps {
  /** Which pipeline produced this response. */
  pipeline: PipelineType;
  /** Optional extra CSS classes for layout adjustments. */
  className?: string;
}

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

interface PipelineConfig {
  label: string;
  description: string;
  /** Tailwind classes for the badge container */
  badgeClass: string;
  /** Tailwind classes for the icon */
  iconClass: string;
  icon: React.ReactNode;
}

const PIPELINE_CONFIG: Record<PipelineType, PipelineConfig> = {
  SQL: {
    label: "SQL",
    description: "Answered by querying the live financial database",
    badgeClass:
      "bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100",
    iconClass: "text-blue-500",
    icon: (
      <svg
        className="w-3 h-3"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
        aria-hidden="true"
      >
        {/* Database cylinder */}
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M3 5v14c0 1.657 4.03 3 9 3s9-1.343 9-3V5" />
        <path d="M3 12c0 1.657 4.03 3 9 3s9-1.343 9-3" />
      </svg>
    ),
  },
  VECTOR: {
    label: "Vector",
    description: "Answered by searching strategic documents",
    badgeClass:
      "bg-green-50 text-green-700 border border-green-200 hover:bg-green-100",
    iconClass: "text-green-500",
    icon: (
      <svg
        className="w-3 h-3"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
        aria-hidden="true"
      >
        {/* Document with search */}
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9zm3.75 11.625a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z"
        />
      </svg>
    ),
  },
  HYBRID: {
    label: "Hybrid",
    description: "Answered by combining database queries and document search",
    badgeClass:
      "bg-amber-50 text-amber-700 border border-amber-200 hover:bg-amber-100",
    iconClass: "text-amber-500",
    icon: (
      <svg
        className="w-3 h-3"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
        aria-hidden="true"
      >
        {/* Merge arrows */}
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5"
        />
      </svg>
    ),
  },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const PipelineBadge: React.FC<PipelineBadgeProps> = ({
  pipeline,
  className = "",
}) => {
  const config = PIPELINE_CONFIG[pipeline] ?? PIPELINE_CONFIG.HYBRID;

  return (
    <span
      className={`
        inline-flex items-center gap-1.5 px-2 py-0.5
        rounded-md text-xs font-semibold
        transition-colors duration-100 cursor-default select-none
        ${config.badgeClass}
        ${className}
      `}
      title={config.description}
      aria-label={`Pipeline: ${config.label} — ${config.description}`}
    >
      <span className={config.iconClass}>{config.icon}</span>
      {config.label}
    </span>
  );
};

export default PipelineBadge;
