import {
  backoffDelayMs,
  isTransientFailure,
  RETRY_ATTEMPT_TIMEOUT_MS,
  RETRY_MAX_TOTAL_MS,
} from "../../src/domain/retryPolicy";

// The timing constants live in src/domain/retryPolicy.ts, shared with the
// QueryClient's retry policy. RETRY_ATTEMPT_TIMEOUT_MS: how long a single
// attempt gets before we treat it as hung and move on. Measured against the
// deployed Render free-tier backend: a healthy response (warm or a normal
// cold start) lands in low single-digit seconds -- see CLAUDE.md's
// "poisoned floor" section. But Render's own hibernate-wake path can also
// fail silently: a request can sit with no response at all (confirmed live
// -- see docs/PROGRESS.md) until Render's edge itself gives up and returns a
// bare 503 with no CORS headers, which the browser then hides behind an
// opaque network error. Without a per-attempt timeout, retryOnWakeup would
// never even get a chance to retry: the very first attempt could hang for
// that entire multi-minute window before `fn` ever rejects.
// RETRY_MAX_TOTAL_MS: target budget for a Render free-tier cold start
// (~30-60s) plus margin. A hibernate-wake failure can take much longer than
// this in the worst case; this bound exists so the UI doesn't hang
// indefinitely, not to guarantee recovery from every possible outage.
export {
  RETRY_ATTEMPT_TIMEOUT_MS,
  RETRY_INITIAL_DELAY_MS,
  RETRY_MAX_DELAY_MS,
  RETRY_MAX_TOTAL_MS,
} from "../../src/domain/retryPolicy";

export class RetryCancelledError extends Error {
  constructor() {
    super("Cancelled");
    this.name = "RetryCancelledError";
  }
}

// A network/CORS-opaque error, a hung attempt, or a bare 502/503. One
// definition, shared with the QueryClient's retry policy.
export const isWakeupSignature = isTransientFailure;

// Abortable so a cancellation mid-backoff rejects immediately instead of
// waiting out the rest of the delay.
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new RetryCancelledError());
      return;
    }
    const timeoutId = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timeoutId);
        reject(new RetryCancelledError());
      },
      { once: true },
    );
  });
}

// Retries `fn` with bounded exponential backoff while it keeps failing with
// the "server is waking up" signature (a network/CORS-opaque error, a bare
// 502/503, or an attempt that never settles at all) -- covers a Render
// free-tier cold start without surfacing the scary red error for what's
// actually a normal (if slow) startup. Any other error is rethrown
// immediately, and so is the last error once the retry budget runs out.
//
// `fn` receives a per-attempt AbortSignal (aborted on that attempt's own
// timeout, or immediately if the caller's `signal` aborts) so callers can
// thread it into their own fetch call -- every attempt is cancellable, not
// just the wait between them.
//
// The defaults suit a small request. Never wrap a large upload body in this:
// the per-attempt timeout covers the whole request, so a slow upload gets
// aborted and re-sent from zero -- use uploadWhenAwake instead.
export async function retryOnWakeup<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  options: {
    signal?: AbortSignal;
    onRetrying?: (elapsedMs: number) => void;
    attemptTimeoutMs?: number;
    maxTotalMs?: number;
  } = {},
): Promise<T> {
  const {
    signal,
    attemptTimeoutMs = RETRY_ATTEMPT_TIMEOUT_MS,
    maxTotalMs = RETRY_MAX_TOTAL_MS,
  } = options;
  if (signal?.aborted) {
    throw new RetryCancelledError();
  }

  const startedAt = Date.now();
  const deadline = startedAt + maxTotalMs;
  let retryIndex = 0;

  while (true) {
    const attemptController = new AbortController();
    const abortAttempt = () => attemptController.abort();
    signal?.addEventListener("abort", abortAttempt);
    const timeoutId = setTimeout(abortAttempt, attemptTimeoutMs);

    try {
      return await fn(attemptController.signal);
    } catch (error) {
      if (signal?.aborted) {
        throw new RetryCancelledError();
      }
      const hung = attemptController.signal.aborted;
      if (!hung && !isWakeupSignature(error)) {
        throw error;
      }
      const delay = backoffDelayMs(retryIndex);
      if (Date.now() + delay >= deadline) {
        throw error;
      }
      options.onRetrying?.(Date.now() - startedAt);
      await sleep(delay, signal);
      retryIndex += 1;
    } finally {
      clearTimeout(timeoutId);
      signal?.removeEventListener("abort", abortAttempt);
    }
  }
}
