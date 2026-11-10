import { RETRY_ATTEMPT_TIMEOUT_MS } from "../../src/domain/retryPolicy";
import { createHungAttemptError } from "./request";

/**
 * Runs one attempt of a small request with its own deadline. `fn` gets a
 * signal that aborts when the caller's `signal` does, or when the attempt
 * runs past `timeoutMs`; the latter rejects with a hung-attempt error, which
 * the retry policy treats as transient (Render's wake can leave a request
 * unanswered for minutes). A caller abort rejects with whatever fetch threw.
 */
export async function withAttemptTimeout<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  { signal, timeoutMs = RETRY_ATTEMPT_TIMEOUT_MS }: { signal?: AbortSignal; timeoutMs?: number },
): Promise<T> {
  const controller = new AbortController();
  const abortFromCaller = () => controller.abort(signal?.reason);
  if (signal?.aborted) {
    abortFromCaller();
  } else {
    signal?.addEventListener("abort", abortFromCaller, { once: true });
  }
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    return await fn(controller.signal);
  } catch (error) {
    if (timedOut && !signal?.aborted) {
      throw createHungAttemptError();
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
    signal?.removeEventListener("abort", abortFromCaller);
  }
}
