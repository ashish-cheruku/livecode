/**
 * ErrorBoundary — catches unhandled React render errors and displays a
 * graceful fallback UI instead of a blank white screen.
 *
 * Usage:
 *   <ErrorBoundary>
 *     <App />
 *   </ErrorBoundary>
 *
 * React class component is required here because error boundaries must use
 * componentDidCatch / getDerivedStateFromError lifecycle methods, which have
 * no functional-component equivalent.
 */

import React, { Component } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ErrorBoundaryProps {
  children: React.ReactNode;
  /** Optional custom fallback. Receives the error and a reset function. */
  fallback?: (error: Error, reset: () => void) => React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

// ---------------------------------------------------------------------------
// Default fallback UI
// ---------------------------------------------------------------------------

const DefaultFallback: React.FC<{ error: Error; onReset: () => void }> = ({
  error,
  onReset,
}) => (
  <div className="flex flex-col items-center justify-center h-full min-h-screen bg-gray-50 px-6 py-12 text-center">
    <div className="w-16 h-16 rounded-2xl bg-red-100 flex items-center justify-center mb-6">
      <svg
        className="w-8 h-8 text-red-500"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.5}
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
        />
      </svg>
    </div>

    <h2 className="text-xl font-semibold text-gray-800 mb-2">
      Something went wrong
    </h2>

    <p className="text-sm text-gray-500 max-w-md mb-2 leading-relaxed">
      An unexpected error occurred in the application. This is usually a
      temporary issue.
    </p>

    {error.message && (
      <p className="text-xs text-red-500 font-mono bg-red-50 border border-red-100 rounded-lg px-3 py-2 max-w-md mb-6 wrap-break-word">
        {error.message}
      </p>
    )}

    <button
      type="button"
      onClick={onReset}
      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 active:bg-indigo-800 transition-colors duration-150"
    >
      <svg
        className="w-4 h-4"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"
        />
      </svg>
      Reload application
    </button>
  </div>
);

// ---------------------------------------------------------------------------
// ErrorBoundary class
// ---------------------------------------------------------------------------

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
    this.handleReset = this.handleReset.bind(this);
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error("[ErrorBoundary] Caught render error:", error, info);
  }

  handleReset(): void {
    this.setState({ hasError: false, error: null });
  }

  render(): React.ReactNode {
    if (this.state.hasError && this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.handleReset);
      }
      return (
        <DefaultFallback error={this.state.error} onReset={this.handleReset} />
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
