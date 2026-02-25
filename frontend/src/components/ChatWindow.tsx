/**
 * ChatWindow — scrollable container that renders the full message list.
 *
 * Responsibilities:
 *   - Renders a list of ChatMessage objects via MessageBubble
 *   - Auto-scrolls to the bottom when new messages are added
 *   - Shows an empty state / welcome screen when no messages exist
 *   - Provides a stable scroll container for the chat UI
 */

import React, { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";
import MessageBubble from "./MessageBubble";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ChatWindowProps {
  /** Ordered list of messages to display (oldest first). */
  messages: ChatMessage[];
  /** Whether the agent is currently processing a query. */
  isLoading: boolean;
  /** Called when the user clicks an example prompt card. */
  onPromptClick?: (text: string) => void;
}

// ---------------------------------------------------------------------------
// Empty state component
// ---------------------------------------------------------------------------

interface EmptyStateProps {
  onPromptClick?: (text: string) => void;
}

const EmptyState: React.FC<EmptyStateProps> = ({ onPromptClick }) => (
  <div className="flex flex-col items-center justify-center h-full text-center px-6 py-12 select-none">
    {/* Icon */}
    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center mb-6 shadow-lg">
      <svg
        className="w-8 h-8 text-white"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.5}
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193-.34.027-.68.052-1.02.072v3.091l-3-3c-1.354 0-2.694-.055-4.02-.163a2.115 2.115 0 01-.825-.242m9.345-8.334a2.126 2.126 0 00-.476-.095 48.64 48.64 0 00-8.048 0c-1.131.094-1.976 1.057-1.976 2.192v4.286c0 .837.46 1.58 1.155 1.951m9.345-8.334V6.637c0-1.621-1.152-3.026-2.76-3.235A48.455 48.455 0 0011.25 3c-2.115 0-4.198.137-6.24.402-1.608.209-2.76 1.614-2.76 3.235v6.226c0 1.621 1.152 3.026 2.76 3.235.577.075 1.157.14 1.74.194V21l4.155-4.155"
        />
      </svg>
    </div>

    {/* Heading */}
    <h2 className="text-xl font-semibold text-gray-800 mb-2">
      Ask anything about your enterprise data
    </h2>

    {/* Subtitle */}
    <p className="text-sm text-gray-500 max-w-md mb-8 leading-relaxed">
      I can answer questions using live database queries, strategic documents,
      or a combination of both. Try one of the examples below.
    </p>

    {/* Example prompts */}
    <div className="grid gap-3 w-full max-w-lg">
      {EXAMPLE_PROMPTS.map((prompt) => (
        <ExamplePromptCard
          key={prompt.label}
          {...prompt}
          onClick={onPromptClick}
        />
      ))}
    </div>
  </div>
);

// ---------------------------------------------------------------------------
// Example prompt cards
// ---------------------------------------------------------------------------

interface ExamplePrompt {
  label: string;
  text: string;
  badge: string;
  badgeColor: string;
  icon: React.ReactNode;
  onClick?: (text: string) => void;
}

const EXAMPLE_PROMPTS: ExamplePrompt[] = [
  {
    label: "SQL",
    text: "What was the total revenue for department D_402 in fiscal year 2025?",
    badge: "SQL",
    badgeColor: "bg-blue-100 text-blue-700",
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582 4-8 4s8 1.79 8 4" />
      </svg>
    ),
  },
  {
    label: "Vector",
    text: "What is Project Phoenix and what are its expected outcomes?",
    badge: "VECTOR",
    badgeColor: "bg-green-100 text-green-700",
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25z" />
      </svg>
    ),
  },
  {
    label: "Hybrid",
    text: "Why did department D_402's EBITDA margin improve from Q3 to Q4 2025, and what strategic initiative drove it?",
    badge: "HYBRID",
    badgeColor: "bg-amber-100 text-amber-700",
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
      </svg>
    ),
  },
];

const ExamplePromptCard: React.FC<ExamplePrompt> = ({
  text,
  badge,
  badgeColor,
  icon,
  onClick,
}) => (
  <button
    type="button"
    onClick={() => onClick?.(text)}
    className="flex items-start gap-3 p-3 rounded-xl border border-gray-200 bg-white/60 backdrop-blur-sm text-left group hover:border-indigo-300 hover:bg-white/90 hover:shadow-sm active:scale-[0.99] transition-all duration-150 cursor-pointer w-full"
    aria-label={`Ask: ${text}`}
  >
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-semibold shrink-0 mt-0.5 ${badgeColor}`}
    >
      {icon}
      {badge}
    </span>
    <p className="text-sm text-gray-600 leading-relaxed">{text}</p>
  </button>
);

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const ChatWindow: React.FC<ChatWindowProps> = ({ messages, isLoading, onPromptClick }) => {
  // Ref to the invisible sentinel element at the bottom of the message list.
  // We scroll this into view whenever messages change.
  const bottomRef = useRef<HTMLDivElement>(null);

  // Ref to the scroll container itself — used to detect if the user has
  // manually scrolled up (in which case we don't force-scroll to bottom).
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change or loading state changes.
  // We use a small timeout to allow React to finish rendering the new message
  // before we scroll, ensuring the scroll target exists in the DOM.
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    // Check if the user is near the bottom (within 150px).
    // If they've scrolled up to read history, don't interrupt them.
    const isNearBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight < 150;

    // Always scroll on new user messages or when loading starts.
    // Only scroll on agent messages if already near bottom.
    const shouldScroll = isNearBottom || isLoading;

    if (shouldScroll) {
      // Use requestAnimationFrame to ensure the DOM has updated
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      });
    }
  }, [messages, isLoading]);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div
      ref={scrollContainerRef}
      className="flex-1 overflow-y-auto overflow-x-hidden"
      role="log"
      aria-label="Chat conversation"
      aria-live="polite"
      aria-relevant="additions"
    >
      {messages.length === 0 ? (
        <EmptyState onPromptClick={onPromptClick} />
      ) : (
        <div className="flex flex-col gap-1 px-4 py-6 max-w-4xl mx-auto w-full">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}

          {/* Invisible sentinel — scrolled into view to reach the bottom */}
          <div ref={bottomRef} className="h-1 shrink-0" aria-hidden="true" />
        </div>
      )}
    </div>
  );
};

export default ChatWindow;
