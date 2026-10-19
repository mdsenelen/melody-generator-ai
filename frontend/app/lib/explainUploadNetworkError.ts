import { getPublicBackendBaseUrl } from "./backendUrl";
import { createNetworkError, type RequestError } from "./request";
import { RetryCancelledError } from "./retryOnWakeup";

// A probe is a tiny GET; past this, call the server unreachable rather than
// make the user wait out another hung Render wake.
export const REACHABILITY_PROBE_TIMEOUT_MS = 10_000;

export const SERVER_REJECTED_MESSAGE =
  "The server rejected the upload — please try again; if it keeps happening, report it.";
export const SERVER_UNREACHABLE_MESSAGE =
  "The server is restarting or unreachable — try again in a minute.";

// Resolves if anything answers at the backend's origin. A no-cors fetch gets
// an opaque response whether or not CORS headers are present, so it only
// rejects when no HTTP response arrives at all.
async function isServerReachable(signal?: AbortSignal): Promise<boolean> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort, { once: true });
  const timeoutId = setTimeout(abort, REACHABILITY_PROBE_TIMEOUT_MS);
  try {
    await fetch(getPublicBackendBaseUrl("/health"), {
      mode: "no-cors",
      cache: "no-store",
      signal: controller.signal,
    });
    return true;
  } catch {
    if (signal?.aborted) {
      throw new RetryCancelledError();
    }
    return false;
  } finally {
    clearTimeout(timeoutId);
    signal?.removeEventListener("abort", abort);
  }
}

// A network_error only says the browser got no readable response. Probe
// once to split that into "the server is down/asleep" and "the server is up
// but this request failed" (a response without CORS headers, or the process
// dying mid-upload) -- they call for different advice. Keeps
// code: "network_error" so isWakeupSignature and existing callers still
// treat it the same; anything else is returned untouched.
export async function explainUploadNetworkError(
  error: unknown,
  signal?: AbortSignal,
): Promise<unknown> {
  if ((error as RequestError | null)?.code !== "network_error") {
    return error;
  }
  return createNetworkError(
    (await isServerReachable(signal)) ? SERVER_REJECTED_MESSAGE : SERVER_UNREACHABLE_MESSAGE,
  );
}
