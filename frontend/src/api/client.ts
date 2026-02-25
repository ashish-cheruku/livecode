/**
 * API Client for the Enterprise Hybrid Data Agent backend.
 *
 * Provides typed fetch wrappers for all backend endpoints.
 * Handles HTTP errors, network errors, and JSON parse errors
 * with a custom ApiError class that carries structured error details.
 *
 * Usage:
 *   import { sendQuery, checkHealth } from './api/client';
 *
 *   const response = await sendQuery("What was D_402 revenue in 2025?");
 *   console.log(response.answer, response.pipeline);
 */

import type {
  ApiErrorDetails,
  FetchOptions,
  HealthResponse,
  QueryResponse,
} from "../types";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * Base URL for the FastAPI backend.
 *
 * In development: reads from VITE_API_BASE_URL env var (set in .env.local),
 * falling back to http://localhost:8000 (the uvicorn default).
 *
 * In production: set VITE_API_BASE_URL to your deployed backend URL.
 *
 * Note: Vite exposes env vars prefixed with VITE_ via import.meta.env.
 * The variable must be set at build time (not runtime).
 */
const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000";

/**
 * Default request timeout in milliseconds.
 * LLM calls can take 10-30 seconds; 60s gives comfortable headroom.
 */
const DEFAULT_TIMEOUT_MS = 60_000;

// ---------------------------------------------------------------------------
// Custom error class
// ---------------------------------------------------------------------------

/**
 * Structured error thrown by all API client functions.
 *
 * Distinguishes between:
 *   - Network errors (fetch failed, CORS blocked): status = 0
 *   - HTTP errors (non-2xx response): status = HTTP status code
 *   - Timeout errors: status = 0, message contains "timeout"
 *
 * The `body` field contains the parsed JSON error response from the backend
 * (e.g., FastAPI's `{"detail": "..."}` validation error format), or the
 * raw text if JSON parsing failed.
 */
export class ApiError extends Error {
  public readonly status: number;
  public readonly body: unknown;

  constructor(details: ApiErrorDetails) {
    super(details.message);
    this.name = "ApiError";
    this.status = details.status;
    this.body = details.body;

    // Maintains proper prototype chain in transpiled environments
    Object.setPrototypeOf(this, ApiError.prototype);
  }

  /**
   * Returns true if this is a client-side error (4xx status code).
   * Useful for distinguishing validation errors from server errors.
   */
  get isClientError(): boolean {
    return this.status >= 400 && this.status < 500;
  }

  /**
   * Returns true if this is a server-side error (5xx status code).
   */
  get isServerError(): boolean {
    return this.status >= 500;
  }

  /**
   * Returns true if this is a network/timeout error (no HTTP response received).
   */
  get isNetworkError(): boolean {
    return this.status === 0;
  }

  /**
   * Returns a user-friendly error message suitable for display in the UI.
   * Extracts FastAPI's `detail` field from validation errors when available.
   */
  get userMessage(): string {
    // FastAPI validation errors have shape: { detail: [...] } or { detail: "..." }
    if (this.body && typeof this.body === "object" && "detail" in this.body) {
      const detail = (this.body as { detail: unknown }).detail;
      if (typeof detail === "string") {
        return detail;
      }
      if (Array.isArray(detail) && detail.length > 0) {
        // Pydantic v2 validation error format: array of { loc, msg, type }
        const firstError = detail[0] as { msg?: string; loc?: string[] };
        if (firstError.msg) {
          const location = firstError.loc?.slice(1).join(".") ?? "field";
          return `Validation error on ${location}: ${firstError.msg}`;
        }
      }
    }
    return this.message;
  }
}

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

/**
 * Generic typed fetch wrapper with timeout, error handling, and JSON parsing.
 *
 * @param endpoint - API path relative to API_BASE_URL (e.g., "/api/query")
 * @param init     - Standard RequestInit options (method, body, headers, etc.)
 * @param options  - Additional client options (signal, extra headers)
 * @returns        - Parsed JSON response body typed as T
 * @throws         - ApiError for all error conditions
 */
async function apiFetch<T>(
  endpoint: string,
  init: RequestInit = {},
  options: FetchOptions = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  // Set up timeout via AbortController
  const timeoutController = new AbortController();
  const timeoutId = setTimeout(
    () => timeoutController.abort(),
    DEFAULT_TIMEOUT_MS
  );

  // Combine caller's signal with our timeout signal.
  // If either aborts, the fetch is cancelled.
  let combinedSignal: AbortSignal;
  if (options.signal) {
    if (typeof AbortSignal.any === "function") {
      combinedSignal = AbortSignal.any([options.signal, timeoutController.signal]);
    } else {
      // Fallback: forward caller's abort to the timeout controller
      options.signal.addEventListener("abort", () => timeoutController.abort());
      combinedSignal = timeoutController.signal;
    }
  } else {
    combinedSignal = timeoutController.signal;
  }

  const defaultHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...options.headers,
  };

  let response: Response;

  try {
    response = await fetch(url, {
      ...init,
      headers: {
        ...defaultHeaders,
        ...(init.headers as Record<string, string> | undefined),
      },
      signal: combinedSignal,
    });
  } catch (err) {
    clearTimeout(timeoutId);

    if (err instanceof DOMException && err.name === "AbortError") {
      if (timeoutController.signal.aborted && !options.signal?.aborted) {
        throw new ApiError({
          status: 0,
          message: `Request timed out after ${DEFAULT_TIMEOUT_MS / 1000}s. The server may be processing a complex query — please try again.`,
        });
      }
      throw new ApiError({
        status: 0,
        message: "Request was cancelled.",
      });
    }

    const networkMessage =
      err instanceof Error ? err.message : "Unknown network error";
    throw new ApiError({
      status: 0,
      message: `Network error: ${networkMessage}. Please check that the backend server is running at ${API_BASE_URL}.`,
    });
  } finally {
    clearTimeout(timeoutId);
  }

  // Parse response body (always attempt JSON, fall back to text)
  let responseBody: unknown;
  const contentType = response.headers.get("content-type") ?? "";

  try {
    if (contentType.includes("application/json")) {
      responseBody = await response.json() as unknown;
    } else {
      responseBody = await response.text();
    }
  } catch {
    responseBody = null;
  }

  // Handle non-2xx HTTP responses
  if (!response.ok) {
    const statusMessages: Record<number, string> = {
      400: "Bad request — the server rejected the request format.",
      401: "Unauthorized — authentication required.",
      403: "Forbidden — you don't have permission to access this resource.",
      404: "Not found — the requested endpoint does not exist.",
      422: "Validation error — the request data is invalid.",
      429: "Too many requests — please wait a moment before trying again.",
      500: "Internal server error — the backend encountered an unexpected error.",
      502: "Bad gateway — the backend server is unreachable.",
      503: "Service unavailable — the backend is temporarily down.",
      504: "Gateway timeout — the backend took too long to respond.",
    };

    const defaultMessage =
      statusMessages[response.status] ??
      `HTTP ${response.status}: ${response.statusText}`;

    throw new ApiError({
      status: response.status,
      message: defaultMessage,
      body: responseBody,
    });
  }

  return responseBody as T;
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/**
 * Sends a natural language question to the agent and returns the full response.
 *
 * The backend routes the question to the appropriate pipeline (SQL, VECTOR,
 * or HYBRID), executes it, and returns a synthesized answer along with
 * intermediate artifacts.
 *
 * @param question - The user's natural language question (non-blank, 3–2000 chars)
 * @param options  - Optional fetch options (AbortSignal for cancellation)
 * @returns        - Full QueryResponse with answer, pipeline type, and artifacts
 * @throws         - ApiError if the request fails (network, HTTP error, timeout)
 *
 * @example
 * ```typescript
 * try {
 *   const response = await sendQuery("What was D_402 revenue in 2025?");
 *   console.log(response.answer);             // "Department D_402's total revenue..."
 *   console.log(response.pipeline);           // "SQL"
 *   console.log(response.artifacts.sql_query); // "SELECT SUM(rev_raw_m)..."
 * } catch (err) {
 *   if (err instanceof ApiError) {
 *     console.error(err.userMessage);
 *   }
 * }
 * ```
 */
export async function sendQuery(
  question: string,
  options: FetchOptions = {}
): Promise<QueryResponse> {
  if (!question || !question.trim()) {
    throw new ApiError({
      status: 0,
      message: "Question must not be blank.",
    });
  }

  const trimmedQuestion = question.trim();

  if (trimmedQuestion.length < 3) {
    throw new ApiError({
      status: 0,
      message: "Question must be at least 3 characters long.",
    });
  }

  if (trimmedQuestion.length > 2000) {
    throw new ApiError({
      status: 0,
      message: "Question must not exceed 2000 characters.",
    });
  }

  return apiFetch<QueryResponse>(
    "/api/query",
    {
      method: "POST",
      body: JSON.stringify({ question: trimmedQuestion }),
    },
    options
  );
}

/**
 * Checks the health of the backend API and its components.
 *
 * Probes SQLite, ChromaDB, and OpenAI API key configuration.
 * Does not make an actual OpenAI API call — only checks configuration.
 *
 * @param options - Optional fetch options (AbortSignal for cancellation)
 * @returns       - HealthResponse with overall status and per-component details
 * @throws        - ApiError if the health endpoint is unreachable
 *
 * @example
 * ```typescript
 * const health = await checkHealth();
 * if (health.status !== "ok") {
 *   console.warn("Backend degraded:", health.components);
 * }
 * ```
 */
export async function checkHealth(
  options: FetchOptions = {}
): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/health", { method: "GET" }, options);
}

// ---------------------------------------------------------------------------
// Utility exports
// ---------------------------------------------------------------------------

/**
 * The resolved API base URL being used by this client.
 * Useful for displaying in debug panels or error messages.
 */
export { API_BASE_URL };

/**
 * Creates a new AbortController and returns both the signal and a
 * convenience `cancel()` function. Use this to implement request cancellation
 * when the user navigates away or submits a new question.
 *
 * @example
 * ```typescript
 * const { signal, cancel } = createCancellableRequest();
 *
 * sendQuery(question, { signal }).catch((err) => {
 *   if (err instanceof ApiError && err.isNetworkError) {
 *     // Cancelled — ignore
 *   }
 * });
 *
 * // Cancel on component unmount or new question submission
 * cancel();
 * ```
 */
export function createCancellableRequest(): {
  signal: AbortSignal;
  cancel: () => void;
} {
  const controller = new AbortController();
  return {
    signal: controller.signal,
    cancel: () => controller.abort(),
  };
}
