import {
  createTranscribeJobResponseSchema,
  jobResultStatusSchema,
  transcribeJobStatusSchema,
  type CreateTranscribeJobResponse,
  type JobResultStatusResponse,
  type RegisterTranscriptionRequest,
  type TranscribeJobStatusResponse,
  type TranscriptionResult,
} from "../../src/domain/contracts/job";
import { classifyFailure } from "../../src/domain/retryPolicy";
import { withAttemptTimeout } from "./attemptTimeout";
import { getPublicBackendApiUrl } from "./backendUrl";
import { requestJson, type RequestError } from "./request";

// Response shapes live in src/domain/contracts/job.ts and are parsed there.

// Bypasses the Next.js /api proxy (same reasoning as uploadFile in
// lib/upload.ts): the file-upload fallback path below carries the full
// audio file, which can exceed Vercel's ~4.5MB serverless function body
// limit.
//
// `uploaded` lets the caller reference a file already stored via
// POST /api/upload instead of sending the same bytes again -- by the time
// page.tsx calls this, the file has always already been uploaded, so this
// is the path actually taken in practice. `file` is only sent directly
// when no prior upload reference exists.
export async function createTranscribeJob(
  file: File,
  uploaded?: { id: string; filename: string } | null,
  options: { signal?: AbortSignal } = {},
): Promise<CreateTranscribeJobResponse> {
  const formData = new FormData();
  if (uploaded) {
    formData.append("upload_id", uploaded.id);
    formData.append("filename", uploaded.filename);
  } else {
    formData.append("file", file);
  }

  return requestJson(getPublicBackendApiUrl("/transcribe"), createTranscribeJobResponseSchema, {
    method: "POST",
    body: formData,
    expectedContentType: "application/json",
    signal: options.signal,
  });
}

// Registers note events the browser already transcribed (Basic Pitch via
// TF.js) as a completed job. Same response shape as a finished
// GET /transcribe/{id}, so the caller skips polling entirely.
export async function registerBrowserTranscription(
  payload: RegisterTranscriptionRequest,
  options: { signal?: AbortSignal } = {},
): Promise<TranscribeJobStatusResponse> {
  return requestJson(getPublicBackendApiUrl("/transcriptions"), transcribeJobStatusSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    expectedContentType: "application/json",
    signal: options.signal,
  });
}

function jobStatusUrl(jobId: string) {
  return getPublicBackendApiUrl(`/transcribe/${encodeURIComponent(jobId)}`);
}

/** GET /transcribe/{id} for a transcription job. */
export async function getTranscribeJob(
  jobId: string,
  options: { signal: AbortSignal },
): Promise<TranscribeJobStatusResponse> {
  return requestJson(jobStatusUrl(jobId), transcribeJobStatusSchema, {
    method: "GET",
    expectedContentType: "application/json",
    signal: options.signal,
  });
}

/**
 * GET /transcribe/{id} for any job kind -- generate-variants and
 * generate-progression results are stored as jobs too (GP3). The result
 * comes back tagged with its `kind`.
 */
export async function getJob(
  jobId: string,
  options: { signal: AbortSignal },
): Promise<JobResultStatusResponse> {
  return requestJson(jobStatusUrl(jobId), jobResultStatusSchema, {
    method: "GET",
    expectedContentType: "application/json",
    signal: options.signal,
  });
}

const POLL_INITIAL_DELAY_MS = 1000;
const POLL_MAX_DELAY_MS = 5000;
const POLL_BACKOFF_FACTOR = 1.5;
// How long the browser keeps polling before giving up
// (TranscribeJobTimeoutError). Must cover the worst realistic recovery
// chain end to end, not just one normal transcription -- see CLAUDE.md's
// "poisoned floor" section for the full chain this budget has to survive:
//   - up to ~125s to detect a dead worker and have its job reclaimed:
//     the backend's claim lease (jobs/worker.py's DEFAULT_LEASE_SECONDS,
//     120s) plus one worker idle-tick interval (~5s) before a live
//     worker notices the stale lease
//   - up to WARMUP_MAX_SECONDS (240s) if the reclaimed job lands on a
//     container that's still warming up after a restart
//   - up to MAX_UPLOAD_DURATION_SEC (600s) for the reclaimed job's own
//     transcription, run at roughly real-time speed
// 125 + 240 + 600 = 965s minimum. 20 minutes (1200s) keeps ~235s (~24%)
// of margin on top for network latency, backoff overhead, and Render's
// own container-restart time (not bounded by anything in this codebase).
// Raised from 15 minutes (900s), which didn't even cover the *old*
// worst-case lease (900s) on its own, let alone warm-up or transcription
// stacked on top of it.
export const POLL_MAX_TOTAL_MS = 20 * 60 * 1000;
const POLL_MAX_CONSECUTIVE_FAILURES = 3;

export class TranscribeJobSupersededError extends Error {
  constructor() {
    super("Superseded by a newer request");
    this.name = "TranscribeJobSupersededError";
  }
}

export class TranscribeJobTimeoutError extends Error {
  constructor() {
    super("This is taking longer than expected. Please try again in a moment.");
    this.name = "TranscribeJobTimeoutError";
  }
}

// Thrown when the job can no longer be found (404) -- on this project's
// free-tier instance, a restart can happen mid-job (see CLAUDE.md's
// "poisoned floor"). Production job state lives in Postgres, so a restart
// alone shouldn't 404 an existing job id; this is for the genuine edge case
// (e.g. a truly expired/evicted row) where it does. Deliberately a distinct
// error from a generic failure so the caller can show a calm "upload again"
// message instead of a red error toast.
export class TranscribeJobLostError extends Error {
  constructor() {
    super("The server restarted and lost your clip. Please upload it again.");
    this.name = "TranscribeJobLostError";
  }
}

// Abortable, so cancelling a poll rejects at once instead of waiting out
// the backoff.
function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new TranscribeJobSupersededError());
      return;
    }
    const onAbort = () => {
      clearTimeout(timeoutId);
      reject(new TranscribeJobSupersededError());
    };
    const timeoutId = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

// Polls GET /transcribe/{id} with bounded exponential backoff until the
// job completes, fails, is cancelled through `signal` (a newer analysis
// started, or the component unmounted), or exceeds POLL_MAX_TOTAL_MS.
// Cancelling aborts the in-flight request and the wait between polls, and
// rejects with TranscribeJobSupersededError (see isCancellation). A handful
// of consecutive unexpected failures are tolerated rather than failing the
// whole flow on one dropped response.
export async function pollTranscribeJob(
  jobId: string,
  options: {
    signal: AbortSignal;
    onStatusChange?: (status: TranscribeJobStatusResponse, elapsedMs: number) => void;
    // Called instead of onStatusChange when a poll hits a 502/503, a
    // network error, or a hung request -- the signature of a free-tier
    // restart (raw platform 502 while nothing's listening yet, our own 503
    // while warming back up, or a dropped connection mid-restart). Doesn't
    // count toward the consecutive-failure cutoff below: the job survives
    // in Postgres across a restart, so the right move is to keep polling
    // within the normal POLL_MAX_TOTAL_MS budget, not give up after a
    // handful of attempts.
    onRestarting?: (elapsedMs: number) => void;
  },
): Promise<TranscriptionResult> {
  const { signal } = options;
  const startedAt = Date.now();
  const deadline = startedAt + POLL_MAX_TOTAL_MS;
  let delay = POLL_INITIAL_DELAY_MS;
  let consecutiveFailures = 0;

  while (true) {
    if (signal.aborted) {
      throw new TranscribeJobSupersededError();
    }
    if (Date.now() > deadline) {
      throw new TranscribeJobTimeoutError();
    }

    let status: TranscribeJobStatusResponse;
    try {
      status = await withAttemptTimeout(
        (attemptSignal) => getTranscribeJob(jobId, { signal: attemptSignal }),
        { signal },
      );
      consecutiveFailures = 0;
    } catch (pollError) {
      if (signal.aborted) {
        throw new TranscribeJobSupersededError();
      }
      if ((pollError as RequestError).status === 404) {
        // The job row itself is gone -- not worth retrying indefinitely,
        // unlike a transient restart below.
        throw new TranscribeJobLostError();
      }
      const kind = classifyFailure(pollError);
      if (kind === "contract") {
        // The same body would fail the same way on every poll.
        throw pollError;
      }
      if (kind === "transient") {
        options.onRestarting?.(Date.now() - startedAt);
      } else {
        consecutiveFailures += 1;
        if (consecutiveFailures >= POLL_MAX_CONSECUTIVE_FAILURES) {
          throw pollError;
        }
      }
      await sleep(delay, signal);
      delay = Math.min(delay * POLL_BACKOFF_FACTOR, POLL_MAX_DELAY_MS);
      continue;
    }

    if (signal.aborted) {
      throw new TranscribeJobSupersededError();
    }
    options.onStatusChange?.(status, Date.now() - startedAt);

    if (status.status === "completed") {
      if (!status.result) {
        throw new Error("Job completed with no result");
      }
      return status.result;
    }
    if (status.status === "failed") {
      throw new Error(status.error || "Transcription failed");
    }

    await sleep(delay, signal);
    delay = Math.min(delay * POLL_BACKOFF_FACTOR, POLL_MAX_DELAY_MS);
  }
}
