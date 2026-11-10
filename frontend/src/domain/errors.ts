// Every failure the audio → melody pipeline can show, in the user's terms.
// Framework-free: no React, no fetch. Errors from app/lib are recognised
// structurally (by `name`, `status`, `code`) so the domain doesn't import them.

export type DomainErrorCode =
  | "invalid_input" // the user's input can't be used as-is; retrying won't help
  | "busy" // HEAVY_WORK_LOCK held by another heavy audio task (429)
  | "warming_up" // boot warm-up grace window, also a 429 (x-error-code tells it apart)
  | "restarting" // 502/503/network failure: the free-tier instance restarting
  | "job_lost" // a transcription job lost to a restart
  | "failed";

export type DomainError = {
  code: DomainErrorCode;
  message: string;
  retryable: boolean;
  retryAfterSeconds?: number;
};

export const RESTARTED_MESSAGE =
  "The server restarted. Your clip wasn't lost — try again in a moment.";

type RequestErrorFields = { status?: number; code?: string; retryAfterSeconds?: number };

function messageOf(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

export function invalidInput(message: string): DomainError {
  return { code: "invalid_input", message, retryable: false };
}

export function failed(message: string): DomainError {
  return { code: "failed", message, retryable: true };
}

/** Classifies a failed generate-variants request (see app/lib/request.ts). */
export function toGenerationError(
  error: unknown,
  fallback = "Variant generation failed",
): DomainError {
  if (!(error instanceof Error)) {
    return failed(fallback);
  }
  const { status, code, retryAfterSeconds } = error as Error & RequestErrorFields;
  if (code === "warming_up") {
    return { code: "warming_up", message: error.message, retryable: true, retryAfterSeconds };
  }
  if (status === 429) {
    return { code: "busy", message: error.message, retryable: true, retryAfterSeconds };
  }
  if (code === "network_error" || status === 502 || status === 503) {
    return { code: "restarting", message: RESTARTED_MESSAGE, retryable: true };
  }
  return failed(error.message);
}

/**
 * Classifies a failed transcription. Unlike generation, the raw message is
 * kept for every failure: /analyse has always shown it verbatim.
 */
export function toTranscriptionError(
  error: unknown,
  fallback = "Transcription failed",
): DomainError {
  if (error instanceof Error && error.name === "TranscribeJobLostError") {
    return { code: "job_lost", message: error.message, retryable: true };
  }
  return failed(messageOf(error, fallback));
}

const CANCELLATION_NAMES = new Set(["RetryCancelledError", "TranscribeJobSupersededError"]);

/**
 * True when a run stopped because it was cancelled (new clip, page left),
 * not because something failed. Cancelling the browser path terminates the
 * worker and rejects with a plain AbortError.
 */
export function isCancellation(error: unknown): boolean {
  if (typeof DOMException !== "undefined" && error instanceof DOMException) {
    return error.name === "AbortError";
  }
  return error instanceof Error && CANCELLATION_NAMES.has(error.name);
}
