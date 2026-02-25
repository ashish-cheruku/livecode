/**
 * MessageBubble — renders a single chat message (user or agent).
 *
 * User messages:
 *   - Right-aligned, blue gradient background
 *   - Simple text content
 *
 * Agent messages:
 *   - Left-aligned, white card with subtle shadow
 *   - Shows a typing indicator (3 animated dots) while isLoading=true
 *   - Shows the answer text once loaded
 *   - Shows a PipelineBadge indicating which pipeline was used
 *   - Shows an ArtifactsPanel (collapsible) with SQL/vector details
 *   - Shows an error banner if errorMessage is set
 */

import React, { memo } from "react";
import type { ChatMessage } from "../types";
import ArtifactsPanel from "./ArtifactsPanel";
import PipelineBadge from "./PipelineBadge";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface MessageBubbleProps {
  message: ChatMessage;
}

// ---------------------------------------------------------------------------
// Typing indicator (three animated dots)
// ---------------------------------------------------------------------------

const TypingIndicator: React.FC = () => (
  <div
    className="flex items-center gap-1 py-1"
    role="status"
    aria-label="Agent is thinking"
  >
    {[0, 1, 2].map((i) => (
      <span
        key={i}
        className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
        style={{ animationDelay: `${i * 150}ms`, animationDuration: "0.8s" }}
        aria-hidden="true"
      />
    ))}
    <span className="sr-only">Agent is thinking…</span>
  </div>
);

// ---------------------------------------------------------------------------
// Timestamp formatter
// ---------------------------------------------------------------------------

function formatTimestamp(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

// ---------------------------------------------------------------------------
// Error banner (shown when the API call failed entirely)
// ---------------------------------------------------------------------------

const ErrorBanner: React.FC<{ message: string }> = ({ message }) => (
  <div
    className="flex items-start gap-2 mt-2 p-3 rounded-lg bg-red-50 border border-red-200"
    role="alert"
  >
    <svg
      className="w-4 h-4 text-red-500 shrink-0 mt-0.5"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden="true"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
      />
    </svg>
    <p className="text-xs text-red-700 leading-relaxed">{message}</p>
  </div>
);

// ---------------------------------------------------------------------------
// Processing time badge
// ---------------------------------------------------------------------------

const ProcessingTimeBadge: React.FC<{ ms: number }> = ({ ms }) => {
  const seconds = (ms / 1000).toFixed(1);
  return (
    <span
      className="text-xs text-gray-400 tabular-nums"
      title={`Processed in ${ms.toFixed(0)}ms`}
    >
      {seconds}s
    </span>
  );
};

// ---------------------------------------------------------------------------
// User message bubble
// ---------------------------------------------------------------------------

const UserBubble: React.FC<{ message: ChatMessage }> = ({ message }) => (
  <div className="flex justify-end mb-4" role="listitem">
    <div className="flex flex-col items-end gap-1 max-w-[75%] min-w-0">
      {/* Bubble */}
      <div className="px-4 py-2.5 rounded-2xl rounded-tr-sm bg-gradient-to-br from-blue-600 to-blue-700 text-white shadow-sm">
        <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
          {message.content}
        </p>
      </div>

      {/* Timestamp */}
      <span className="text-xs text-gray-400 px-1">
        {formatTimestamp(message.timestamp)}
      </span>
    </div>
  </div>
);

// ---------------------------------------------------------------------------
// Agent avatar icon
// ---------------------------------------------------------------------------

const AgentAvatar: React.FC = () => (
  <div
    className="shrink-0 w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm mt-0.5"
    aria-hidden="true"
  >
    <svg
      className="w-4 h-4 text-white"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"
      />
    </svg>
  </div>
);

// ---------------------------------------------------------------------------
// Agent message bubble
// ---------------------------------------------------------------------------

const AgentBubble: React.FC<{ message: ChatMessage }> = ({ message }) => {
  const { isLoading, queryResponse, errorMessage, content, timestamp } = message;

  return (
    <div className="flex justify-start mb-4" role="listitem">
      <div className="flex gap-3 max-w-[85%] min-w-0 w-full">
        {/* Agent avatar */}
        <AgentAvatar />

        {/* Content area */}
        <div className="flex-1 min-w-0">
          {/* Answer card */}
          <div className="bg-white rounded-2xl rounded-tl-sm border border-gray-200 shadow-sm px-4 py-3">
            {/* Loading state: typing indicator */}
            {isLoading && <TypingIndicator />}

            {/* Answer text */}
            {!isLoading && content && (
              <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap break-words">
                {content}
              </p>
            )}

            {/* Full API failure error banner */}
            {!isLoading && errorMessage && (
              <ErrorBanner message={errorMessage} />
            )}
          </div>

          {/* Footer: pipeline badge + timing + partial error + artifacts */}
          {!isLoading && queryResponse && (
            <div className="mt-2 space-y-2">
              {/* Badge row */}
              <div className="flex items-center gap-2 flex-wrap px-1">
                <PipelineBadge pipeline={queryResponse.pipeline} />

                {queryResponse.processing_time_ms != null && (
                  <ProcessingTimeBadge ms={queryResponse.processing_time_ms} />
                )}

                {/* Partial pipeline error indicator */}
                {queryResponse.error && (
                  <span
                    className="inline-flex items-center gap-1 text-xs text-amber-600"
                    title={`Partial error: ${queryResponse.error}`}
                  >
                    <svg
                      className="w-3.5 h-3.5 shrink-0"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                      aria-hidden="true"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
                      />
                    </svg>
                    Partial result
                  </span>
                )}
              </div>

              {/* Artifacts panel */}
              <ArtifactsPanel
                artifacts={queryResponse.artifacts}
                pipeline={queryResponse.pipeline}
              />
            </div>
          )}

          {/* Timestamp */}
          {!isLoading && (
            <span className="text-xs text-gray-400 px-1 mt-1 block">
              {formatTimestamp(timestamp)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main component (memoized to prevent re-renders when only input changes)
// ---------------------------------------------------------------------------

const MessageBubble: React.FC<MessageBubbleProps> = memo(({ message }) => {
  if (message.role === "user") {
    return <UserBubble message={message} />;
  }
  return <AgentBubble message={message} />;
});

MessageBubble.displayName = "MessageBubble";

export default MessageBubble;
