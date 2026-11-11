// When a failed request is worth repeating, and how long to wait first.
// Pure: no timers, no fetch. Used two ways: as the QueryClient's default
// `retry` / `retryDelay` (app/lib/queryClient.ts), and as the "is the server
// waking up?" test inside retryOnWakeup (app/lib/retryOnWakeup.ts), so both
// agree on what "transient" means.
//
// Errors are recognised structurally (`name`, `status`, `code`), like
// src/domain/errors.ts, so this module imports nothing from app/.

import { isCancellation } from "./errors";

// How long a single attempt gets before it's treated as hung. Render's
// free-tier wake can leave a request with no response at all until its edge
// gives up minutes later -- see app/lib/retryOnWakeup.ts.
export const RETRY_ATTEMPT_TIMEOUT_MS = 15_000;
export const RETRY_INITIAL_DELAY_MS = 2_000;
export const RETRY_MAX_DELAY_MS = 10_000;
export const RETRY_BACKOFF_FACTOR = 2;
// A Render free-tier cold start (~30-60s) plus margin. Bounds how long the UI
// keeps saying "waking up" before it shows the real error.
export const RETRY_MAX_TOTAL_MS = 90_000;

/** `code` on an attempt abandoned after RETRY_ATTEMPT_TIMEOUT_MS. */
export const HUNG_ATTEMPT_CODE = "attempt_timeout";

export type FailureKind =
  | "transient" // network error, hung attempt, 502/503: the server is (re)starting
  | "cancelled" // the user or a newer run stopped it; never an error to show
  | "contract" // the response didn't match src/domain/contracts; retrying won't fix it
  | "busy" // 429 (HEAVY_WORK_LOCK held, or boot warm-up): has its own UI state
  | "rejected" // any other 4xx: the request itself is wrong
  | "permanent"; // 500 and anything unrecognised

type ErrorFields = {
  name?: unknown;
  status?: unknown;
  code?: unknown;
  retryAfterSeconds?: unknown;
};

function fieldsOf(error: unknown): ErrorFields {
  return typeof error === "object" && error !== null ? error : {};
}

export function classifyFailure(error: unknown): FailureKind {
  if (isCancellation(error)) {
    return "cancelled";
  }
  const { name, status, code } = fieldsOf(error);
  if (name === "ContractError" || code === "contract_error") {
    return "contract";
  }
  if (code === "network_error" || code === HUNG_ATTEMPT_CODE || status === 502 || status === 503) {
    return "transient";
  }
  if (status === 429) {
    return "busy";
  }
  if (typeof status === "number" && status >= 400 && status < 500) {
    return "rejected";
  }
  return "permanent";
}

/** The one definition of "the server is waking up or restarting". */
export function isTransientFailure(error: unknown): boolean {
  return classifyFailure(error) === "transient";
}

/** The wait before retry number `retryIndex` (0-based), ignoring retry-after. */
export function backoffDelayMs(retryIndex: number): number {
  return Math.min(RETRY_INITIAL_DELAY_MS * RETRY_BACKOFF_FACTOR ** retryIndex, RETRY_MAX_DELAY_MS);
}

/**
 * TanStack Query's `retryDelay`: `failureCount` is how many attempts failed
 * before this one. A server-sent retry-after wins over the backoff.
 */
export function retryDelayMs(failureCount: number, error: unknown): number {
  const { retryAfterSeconds } = fieldsOf(error);
  if (typeof retryAfterSeconds === "number" && Number.isFinite(retryAfterSeconds)) {
    return Math.min(Math.max(0, retryAfterSeconds * 1000), RETRY_MAX_TOTAL_MS);
  }
  return backoffDelayMs(failureCount);
}

/**
 * TanStack Query's `retry`: only transient failures, and only while the
 * backoff waited so far plus the next one stays inside RETRY_MAX_TOTAL_MS --
 * the same budget retryOnWakeup enforces with a clock.
 */
export function shouldRetry(failureCount: number, error: unknown): boolean {
  if (!isTransientFailure(error)) {
    return false;
  }
  let waited = 0;
  for (let index = 0; index <= failureCount; index += 1) {
    waited += backoffDelayMs(index);
  }
  return waited < RETRY_MAX_TOTAL_MS;
}

/**
 * True while a query is retrying after a transient failure: the moment to
 * say "waking the server" instead of flashing an error.
 */
export function isRetryingTransient(query: {
  fetchStatus: "fetching" | "paused" | "idle";
  failureCount: number;
  failureReason: unknown;
}): boolean {
  return (
    query.fetchStatus === "fetching" &&
    query.failureCount > 0 &&
    isTransientFailure(query.failureReason)
  );
}
