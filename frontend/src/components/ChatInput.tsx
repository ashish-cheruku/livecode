/**
 * ChatInput — text input area with send button for submitting questions.
 *
 * Responsibilities:
 *   - Controlled textarea that auto-resizes as the user types
 *   - Send button that submits the question
 *   - Enter key submits (Shift+Enter inserts a newline)
 *   - Disabled state while the agent is processing (isLoading=true)
 *   - Character count indicator when approaching the 2000-char limit
 *   - Clears and resets height after submission
 */

import React, {
    useCallback,
    useEffect,
    useRef,
    useState,
  } from "react";
  
  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------
  
  const MAX_CHARS = 2000;
  const MIN_CHARS = 3;
  /** Show character counter when this many characters remain */
  const CHAR_COUNTER_THRESHOLD = 200;
  /** Maximum textarea height in pixels before it stops growing */
  const MAX_TEXTAREA_HEIGHT = 200;
  
  // ---------------------------------------------------------------------------
  // Props
  // ---------------------------------------------------------------------------
  
  interface ChatInputProps {
    /**
     * Called when the user submits a question.
     * The question string is already trimmed.
     */
    onSubmit: (question: string) => void;
    /** Whether the agent is currently processing — disables the input. */
    isLoading: boolean;
    /** Optional placeholder text for the textarea. */
    placeholder?: string;
  }
  
  // ---------------------------------------------------------------------------
  // Send icon
  // ---------------------------------------------------------------------------
  
  const SendIcon: React.FC<{ className?: string }> = ({ className }) => (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
    </svg>
  );
  
  // ---------------------------------------------------------------------------
  // Loading spinner (shown inside send button while processing)
  // ---------------------------------------------------------------------------
  
  const LoadingSpinner: React.FC<{ className?: string }> = ({ className }) => (
    <svg
      className={`animate-spin ${className ?? ""}`}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
  
  // ---------------------------------------------------------------------------
  // Main component
  // ---------------------------------------------------------------------------
  
  const ChatInput: React.FC<ChatInputProps> = ({
    onSubmit,
    isLoading,
    placeholder = "Ask a question about your enterprise data…",
  }) => {
    const [value, setValue] = useState("");
    const textareaRef = useRef<HTMLTextAreaElement>(null);
  
    // ---------------------------------------------------------------------------
    // Auto-resize textarea
    // ---------------------------------------------------------------------------
  
    const resizeTextarea = useCallback(() => {
      const el = textareaRef.current;
      if (!el) return;
      // Reset height to auto so shrinkage works correctly
      el.style.height = "auto";
      // Set to scrollHeight, capped at MAX_TEXTAREA_HEIGHT
      el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
    }, []);
  
    useEffect(() => {
      resizeTextarea();
    }, [value, resizeTextarea]);
  
    // Focus the textarea when loading completes (after agent responds)
    useEffect(() => {
      if (!isLoading) {
        // Small delay to let the UI settle before focusing
        const timer = setTimeout(() => {
          textareaRef.current?.focus();
        }, 100);
        return () => clearTimeout(timer);
      }
    }, [isLoading]);
  
    // ---------------------------------------------------------------------------
    // Submission logic
    // ---------------------------------------------------------------------------
  
    const handleSubmit = useCallback(() => {
      const trimmed = value.trim();
      if (trimmed.length < MIN_CHARS || trimmed.length > MAX_CHARS || isLoading) {
        return;
      }
      onSubmit(trimmed);
      setValue("");
      // Reset textarea height after clearing
      requestAnimationFrame(() => {
        if (textareaRef.current) {
          textareaRef.current.style.height = "auto";
        }
      });
    }, [value, isLoading, onSubmit]);
  
    const handleKeyDown = useCallback(
      (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault(); // Prevent newline on plain Enter
          handleSubmit();
        }
      },
      [handleSubmit]
    );
  
    const handleChange = useCallback(
      (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const newValue = e.target.value;
        // Enforce max length at the input level
        if (newValue.length <= MAX_CHARS) {
          setValue(newValue);
        }
      },
      []
    );
  
    // ---------------------------------------------------------------------------
    // Derived state
    // ---------------------------------------------------------------------------
  
    const trimmedLength = value.trim().length;
    const isSubmittable = trimmedLength >= MIN_CHARS && !isLoading;
    const charsRemaining = MAX_CHARS - value.length;
    const showCharCounter = charsRemaining <= CHAR_COUNTER_THRESHOLD;
    const isNearLimit = charsRemaining <= 50;
    const isAtLimit = charsRemaining <= 0;
  
    // ---------------------------------------------------------------------------
    // Render
    // ---------------------------------------------------------------------------
  
    return (
      <div className="border-t border-gray-200 bg-white px-4 py-3">
        <div className="max-w-4xl mx-auto">
          {/* Input container */}
          <div
            className={`
              flex items-end gap-2 rounded-2xl border bg-gray-50 px-4 py-3
              transition-all duration-150
              ${isLoading
                ? "border-gray-200 bg-gray-50 opacity-80"
                : "border-gray-300 hover:border-gray-400 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20 focus-within:bg-white"
              }
            `}
          >
            {/* Textarea */}
            <textarea
              ref={textareaRef}
              value={value}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              placeholder={isLoading ? "Thinking…" : placeholder}
              rows={1}
              aria-label="Question input"
              aria-describedby="chat-input-hint"
              className={`
                flex-1 resize-none bg-transparent text-sm text-gray-900
                placeholder:text-gray-400 outline-none leading-relaxed
                disabled:cursor-not-allowed disabled:text-gray-400
                min-h-[24px] py-0.5
              `}
              style={{ maxHeight: `${MAX_TEXTAREA_HEIGHT}px` }}
            />
  
            {/* Send button */}
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!isSubmittable}
              aria-label={isLoading ? "Processing…" : "Send question"}
              className={`
                shrink-0 w-8 h-8 rounded-xl flex items-center justify-center
                transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-offset-1
                ${isSubmittable
                  ? "bg-blue-600 text-white hover:bg-blue-700 active:scale-95 focus:ring-blue-500 shadow-sm"
                  : "bg-gray-200 text-gray-400 cursor-not-allowed"
                }
              `}
            >
              {isLoading ? (
                <LoadingSpinner className="w-4 h-4" />
              ) : (
                <SendIcon className="w-4 h-4" />
              )}
            </button>
          </div>
  
          {/* Footer row: hint text + character counter */}
          <div className="flex items-center justify-between mt-1.5 px-1">
            <p
              id="chat-input-hint"
              className="text-xs text-gray-400"
            >
              {isLoading
                ? "Processing your question…"
                : "Press Enter to send · Shift+Enter for new line"}
            </p>
  
            {showCharCounter && (
              <span
                className={`text-xs font-medium tabular-nums ${
                  isAtLimit
                    ? "text-red-500"
                    : isNearLimit
                    ? "text-amber-500"
                    : "text-gray-400"
                }`}
                aria-live="polite"
                aria-label={`${charsRemaining} characters remaining`}
              >
                {charsRemaining}
              </span>
            )}
          </div>
        </div>
      </div>
    );
  };
  
  export default ChatInput;
  