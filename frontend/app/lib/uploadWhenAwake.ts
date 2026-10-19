import { explainUploadNetworkError } from "./explainUploadNetworkError";
import { isWakeupSignature, RetryCancelledError } from "./retryOnWakeup";
import { uploadFile, type UploadResponse } from "./upload";
import { waitForBackend } from "./waitForBackend";

export type UploadPhase = "waking" | "uploading";

export const WAKING_STATUS_MESSAGE =
  "Waking the server — this can take a minute or two on the free tier…";

// A second wake-then-upload cycle covers a server that answered /health and
// then restarted (or OOM'd) mid-upload. More than that re-sends a large body
// over and over for what is probably a real outage.
const MAX_CYCLES = 2;

// Wakes the backend with a cheap /health poll, then sends the upload once,
// with no per-attempt timeout -- a 50MB body on an ordinary connection can
// take well over a minute to send, and aborting it only restarts it from
// zero. Still cancellable through the caller's `signal`.
export async function uploadWhenAwake(
  file: File,
  options: { signal?: AbortSignal; onPhase?: (phase: UploadPhase) => void } = {},
): Promise<UploadResponse> {
  const { signal, onPhase } = options;

  for (let cycle = 1; ; cycle++) {
    try {
      await waitForBackend({ signal, onWaiting: () => onPhase?.("waking") });
    } catch (error) {
      if (error instanceof RetryCancelledError) {
        throw error;
      }
      throw await explainUploadNetworkError(error, signal);
    }
    onPhase?.("uploading");
    try {
      return await uploadFile(file, { signal });
    } catch (error) {
      if (signal?.aborted) {
        throw new RetryCancelledError();
      }
      if (cycle >= MAX_CYCLES || !isWakeupSignature(error)) {
        throw await explainUploadNetworkError(error, signal);
      }
    }
  }
}
