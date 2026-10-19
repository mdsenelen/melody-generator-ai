import { getPublicBackendBaseUrl } from "./backendUrl";
import { createNetworkError, type RequestError } from "./request";
import { retryOnWakeup } from "./retryOnWakeup";

// A GET /health is tiny, so a ping that hasn't answered in this long is hung
// -- move on to the next one rather than waiting out Render's edge timeout.
export const WAIT_ATTEMPT_TIMEOUT_MS = 12_000;
// Render's free-tier hibernate-wake can hang for several minutes before its
// edge gives up with a bare, CORS-less 503 (confirmed live -- see commit
// 6ba17de / docs/PROGRESS.md). Cover that observed window; beyond it, the
// server is more likely down than slow.
export const WAIT_MAX_TOTAL_MS = 240_000;

async function pingHealth(signal: AbortSignal): Promise<void> {
  let response: Response;
  try {
    response = await fetch(getPublicBackendBaseUrl("/health"), { signal, cache: "no-store" });
  } catch {
    // A TypeError (unreachable / CORS-opaque) or this ping's own timeout.
    // Both mean "not awake yet"; report them uniformly so the last error
    // surfaced on budget exhaustion is the actionable network message, not
    // a raw AbortError. A caller abort is handled by retryOnWakeup itself.
    throw createNetworkError();
  }
  // Any other answer -- even a 500 -- means the process is up and serving.
  if (response.status === 502 || response.status === 503) {
    const error = new Error(`Request failed (${response.status})`) as RequestError;
    error.status = response.status;
    throw error;
  }
}

// Resolves once the backend answers /health, polling with retryOnWakeup's
// bounded backoff. Lets a caller wake the server with a cheap request before
// committing a large upload body to it. `onWaiting` fires each time a ping
// fails, i.e. only when the server really is asleep -- a warm server never
// triggers it.
export function waitForBackend(
  options: { signal?: AbortSignal; onWaiting?: (elapsedMs: number) => void } = {},
): Promise<void> {
  return retryOnWakeup(pingHealth, {
    signal: options.signal,
    onRetrying: options.onWaiting,
    attemptTimeoutMs: WAIT_ATTEMPT_TIMEOUT_MS,
    maxTotalMs: WAIT_MAX_TOTAL_MS,
  });
}
