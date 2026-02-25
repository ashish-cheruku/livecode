/**
 * App — root component for CompanyInsights.AI
 *
 * Owns all application state:
 *   - messages: ChatMessage[]  — the full conversation history
 *   - isLoading: boolean       — true while an API request is in-flight
 *
 * Query submission flow:
 *   1. User submits a question via ChatInput
 *   2. Append user message to messages (immediate, optimistic)
 *   3. Append a loading agent message (shows typing indicator)
 *   4. Cancel any in-flight request (prevents race conditions)
 *   5. Call sendQuery() from the API client
 *   6a. On success: update the loading agent message with the response
 *   6b. On error: update the loading agent message with the error
 *   7. Set isLoading = false
 *
 * Layout:
 *   ┌─────────────────────────────────────┐
 *   │  Header (dark, fixed height)        │
 *   ├─────────────────────────────────────┤
 *   │  ChatWindow (flex-1, scrollable)    │
 *   ├─────────────────────────────────────┤
 *   │  ChatInput (fixed bottom)           │
 *   └─────────────────────────────────────┘
 */

import React, { useCallback, useRef, useState } from "react";
import ChatInput from "./components/ChatInput";
import ChatWindow from "./components/ChatWindow";
import { ApiError, createCancellableRequest, sendQuery } from "./api/client";
import type { ChatMessage } from "./types";

// ---------------------------------------------------------------------------
// ID generator — uses crypto.randomUUID() with a Date.now() fallback
// ---------------------------------------------------------------------------

function generateId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback for environments without crypto.randomUUID
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

// ---------------------------------------------------------------------------
// Header component
// ---------------------------------------------------------------------------

const Header: React.FC<{ isLoading: boolean }> = ({ isLoading }) => (
  <header className="shrink-0 bg-gradient-to-r from-indigo-950 via-indigo-900 to-purple-900 border-b border-indigo-800/50 shadow-lg">
    <div className="max-w-4xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
      {/* Brand */}
      <div className="flex items-center gap-2.5 min-w-0">
        {/* Logo mark */}
        <div className="shrink-0 w-7 h-7 rounded-lg bg-white/10 border border-white/20 flex items-center justify-center">
          <svg
            className="w-4 h-4 text-white"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"
            />
          </svg>
        </div>

        {/* Title */}
        <div className="min-w-0">
          <h1 className="text-sm font-semibold text-white tracking-tight truncate">
            CompanyInsights
            <span className="text-indigo-300">.AI</span>
          </h1>
          <p className="text-xs text-indigo-400 leading-none hidden sm:block">
            Enterprise Hybrid Data Agent
          </p>
        </div>
      </div>

      {/* Right side: status indicator */}
      <div className="shrink-0 flex items-center gap-2">
        {isLoading ? (
          <span className="flex items-center gap-1.5 text-xs text-indigo-300">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-300" />
            </span>
            <span className="hidden sm:inline">Processing…</span>
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-xs text-indigo-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400" aria-hidden="true" />
            <span className="hidden sm:inline">Ready</span>
          </span>
        )}

        {/* Pipeline legend */}
        <div className="hidden md:flex items-center gap-2 ml-2 pl-2 border-l border-indigo-800/60">
          <span className="flex items-center gap-1 text-xs text-indigo-400">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" aria-hidden="true" />
            SQL
          </span>
          <span className="flex items-center gap-1 text-xs text-indigo-400">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400" aria-hidden="true" />
            Vector
          </span>
          <span className="flex items-center gap-1 text-xs text-indigo-400">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400" aria-hidden="true" />
            Hybrid
          </span>
        </div>
      </div>
    </div>
  </header>
);

// ---------------------------------------------------------------------------
// Main App component
// ---------------------------------------------------------------------------

const App: React.FC = () => {
  // ── State ──────────────────────────────────────────────────────────────────
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Ref to the cancel function for the current in-flight request.
  // Using a ref (not state) because changing it should not trigger a re-render.
  const cancelCurrentRequest = useRef<(() => void) | null>(null);

  // ── Query submission handler ───────────────────────────────────────────────

  const handleSubmit = useCallback(async (question: string): Promise<void> => {
    // ── 1. Cancel any in-flight request ─────────────────────────────────────
    if (cancelCurrentRequest.current) {
      cancelCurrentRequest.current();
      cancelCurrentRequest.current = null;
    }

    // ── 2. Create message IDs ────────────────────────────────────────────────
    const userMessageId = generateId();
    const agentMessageId = generateId();
    const now = new Date().toISOString();

    // ── 3. Append user message (immediate) ──────────────────────────────────
    const userMessage: ChatMessage = {
      id: userMessageId,
      role: "user",
      content: question,
      timestamp: now,
    };

    // ── 4. Append loading agent message ─────────────────────────────────────
    const loadingAgentMessage: ChatMessage = {
      id: agentMessageId,
      role: "agent",
      content: "",
      timestamp: new Date().toISOString(),
      isLoading: true,
      queryResponse: null,
      errorMessage: null,
    };

    setMessages((prev) => [...prev, userMessage, loadingAgentMessage]);
    setIsLoading(true);

    // ── 5. Set up cancellable request ────────────────────────────────────────
    const { signal, cancel } = createCancellableRequest();
    cancelCurrentRequest.current = cancel;

    // ── 6. Call the API ──────────────────────────────────────────────────────
    let wasCancelled = false;
    try {
      const response = await sendQuery(question, { signal });

      // ── 6a. Success: update the loading agent message with the response ────
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === agentMessageId
            ? {
                ...msg,
                isLoading: false,
                content: response.answer,
                queryResponse: response,
                errorMessage: null,
                timestamp: new Date().toISOString(),
              }
            : msg
        )
      );
    } catch (err) {
      // ── 6b. Error: update the loading agent message with the error ─────────

      // Don't show an error if the request was intentionally cancelled
      // (user submitted a new question before this one completed)
      if (err instanceof ApiError && err.isNetworkError && signal.aborted) {
        wasCancelled = true;
        // Request was cancelled — remove the loading agent message silently
        setMessages((prev) =>
          prev.filter((msg) => msg.id !== agentMessageId)
        );
        // Don't set isLoading = false here — the new request will manage it
        return;
      }

      // Determine the user-facing error message
      let errorMessage: string;
      if (err instanceof ApiError) {
        errorMessage = err.userMessage;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      } else {
        errorMessage =
          "An unexpected error occurred. Please try again.";
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === agentMessageId
            ? {
                ...msg,
                isLoading: false,
                content: "",
                queryResponse: null,
                errorMessage,
                timestamp: new Date().toISOString(),
              }
            : msg
        )
      );
    } finally {
      // Clear the cancel ref if this request is still the current one
      if (cancelCurrentRequest.current === cancel) {
        cancelCurrentRequest.current = null;
      }
      // Don't clear the loading state if this request was cancelled — the new
      // request that triggered the cancellation is managing isLoading itself.
      if (!wasCancelled) {
        setIsLoading(false);
      }
    }
  }, []);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full bg-gray-50 overflow-hidden">
      {/* Dark header with branding */}
      <Header isLoading={isLoading} />

      {/* Scrollable chat area */}
      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        onPromptClick={handleSubmit}
      />

      {/* Fixed input at the bottom */}
      <ChatInput
        onSubmit={handleSubmit}
        isLoading={isLoading}
        placeholder="Ask about revenue, EBITDA, Project Phoenix, headcount…"
      />
    </div>
  );
};

export default App;
