import type { RequestError } from "../request";
import { retryOnWakeup } from "../retryOnWakeup";
import {
  registerBrowserTranscription,
  type TranscribeJobStatusResponse,
  type TranscriptionResult,
} from "../transcribeJob";
import { checkBrowserEligibility, type IneligibleReason } from "./eligibility";
import { shouldFallBackToServer, transcribeInBrowser } from "./transcribeInBrowser";

// Registration runs on the backend's 0.1 vCPU -- on a cold container that's
// shared with the ~145 s Basic Pitch boot warm-up -- so one attempt gets
// longer than retryOnWakeup's 15 s default before it's treated as hung.
const REGISTER_ATTEMPT_TIMEOUT_MS = 45_000;

export type StoredUpload = { id: string; filename: string };

export type FallbackReason = IneligibleReason | "browser_failed" | "register_rejected";

export type BrowserOutcome =
  | {
      kind: "completed";
      stored: StoredUpload;
      job: TranscribeJobStatusResponse;
      result: TranscriptionResult;
    }
  // `stored` is the upload when it already happened, so the server path
  // reuses it instead of sending the file again.
  | { kind: "fallback"; reason: FallbackReason; stored: StoredUpload | null };

type Deps = {
  checkEligibility: typeof checkBrowserEligibility;
  transcribe: typeof transcribeInBrowser;
  register: typeof registerBrowserTranscription;
};

const DEFAULT_DEPS: Deps = {
  checkEligibility: checkBrowserEligibility,
  transcribe: transcribeInBrowser,
  register: registerBrowserTranscription,
};

// Rejected registrations go to the server path rather than the error toast:
// 422 means the server wouldn't accept these notes (e.g. over the note-count
// cap), 429 means the registration bucket is spent -- POST /api/transcribe
// has its own bucket, so the fallback isn't charged twice.
function isRegisterRejection(error: unknown): boolean {
  const status = (error as RequestError | undefined)?.status;
  return status === 422 || status === 429;
}

// Transcribes in the browser (Basic Pitch on TF.js in a Worker) while the
// upload runs in parallel, then registers the notes as a completed job. Any
// outcome that should be retried on the server comes back as `fallback`;
// only cancellations, upload failures and unexpected server errors throw.
export async function runBrowserTranscription({
  file,
  signal,
  ensureUploaded,
  onProgress,
  onRegisterRetrying,
  deps = DEFAULT_DEPS,
}: {
  file: File;
  signal: AbortSignal;
  ensureUploaded: () => Promise<StoredUpload>;
  onProgress?: (fraction: number) => void;
  onRegisterRetrying?: () => void;
  deps?: Deps;
}): Promise<BrowserOutcome> {
  const eligibility = await deps.checkEligibility(file);
  if (!eligibility.eligible) {
    return { kind: "fallback", reason: eligibility.reason, stored: null };
  }

  const upload = ensureUploaded();
  // Observed below on every path; this just stops an upload failure that
  // lands mid-transcription from being reported as unhandled in between.
  upload.catch(() => {});

  let transcription;
  try {
    transcription = await deps.transcribe(file, { signal, onProgress });
  } catch (error) {
    if (!shouldFallBackToServer(error)) {
      throw error;
    }
    console.warn("[analysis] browser transcription failed; using the server", { error });
    return { kind: "fallback", reason: "browser_failed", stored: await upload };
  }

  const stored = await upload;
  let job: TranscribeJobStatusResponse;
  try {
    job = await retryOnWakeup(
      (attemptSignal) =>
        deps.register(
          {
            upload_id: stored.id,
            source_filename: file.name,
            source_duration_sec: transcription.durationSec,
            tfjs_backend: transcription.tfjsBackend,
            note_events: transcription.noteEvents,
          },
          { signal: attemptSignal },
        ),
      { signal, onRetrying: onRegisterRetrying, attemptTimeoutMs: REGISTER_ATTEMPT_TIMEOUT_MS },
    );
  } catch (error) {
    if (!isRegisterRejection(error)) {
      throw error;
    }
    console.warn("[analysis] browser transcription not accepted; using the server", { error });
    return { kind: "fallback", reason: "register_rejected", stored };
  }

  if (job.status !== "completed" || !job.result) {
    return { kind: "fallback", reason: "register_rejected", stored };
  }
  return { kind: "completed", stored, job, result: job.result };
}
