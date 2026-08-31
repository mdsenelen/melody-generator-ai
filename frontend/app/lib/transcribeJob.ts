import { getPublicBackendApiUrl } from "./backendUrl";
import { requestJson } from "./request";

export type TranscriptionResult = {
  n_notes: number;
  duration_sec: number;
  source_duration_sec: number;
  truncated: boolean;
  midi_b64: string;
  wav_b64: string | null;
  midi_filename: string;
  wav_filename: string;
  mood_label: "happy" | "sad" | "neutral";
  mood_idx: number;
  detected_chords: string[];
  key: string;
  pitch_histogram: number[];
  tempo_bpm: number;
  average_pitch: number;
};

export type TranscribeJobStatus = "queued" | "processing" | "completed" | "failed" | "expired";

// Generic over the result shape because GET /transcribe/{job_id} (see
// getTranscribeJob below) also serves generate-variants and
// generate-progression results now (see jobs/service.create_completed_job,
// GP3) -- the store doesn't discriminate by job kind, only by id.
export type JobStatusResponse<TResult> = {
  job_id: string;
  status: TranscribeJobStatus;
  progress: number;
  result: TResult | null;
  error: string | null;
};

export type TranscribeJobStatusResponse = JobStatusResponse<TranscriptionResult>;

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
): Promise<TranscribeJobStatusResponse> {
  const formData = new FormData();
  if (uploaded) {
    formData.append("upload_id", uploaded.id);
    formData.append("filename", uploaded.filename);
  } else {
    formData.append("file", file);
  }

  return requestJson<TranscribeJobStatusResponse>(getPublicBackendApiUrl("/transcribe"), {
    method: "POST",
    body: formData,
    expectedContentType: "application/json",
  });
}

export async function getTranscribeJob<TResult = TranscriptionResult>(
  jobId: string,
): Promise<JobStatusResponse<TResult>> {
  return requestJson<JobStatusResponse<TResult>>(
    getPublicBackendApiUrl(`/transcribe/${encodeURIComponent(jobId)}`),
    { method: "GET", expectedContentType: "application/json" },
  );
}

const POLL_INITIAL_DELAY_MS = 1000;
const POLL_MAX_DELAY_MS = 5000;
const POLL_BACKOFF_FACTOR = 1.5;
// Matches the backend worker's lease margin (DEFAULT_LEASE_SECONDS, see
// jobs/worker.py) above the longest a job can legitimately take
// (MAX_ANALYSIS_DURATION_SEC's 240s, plus overhead) -- without this, a
// long-but-healthy transcription could still get abandoned client-side
// even though the server would have finished it.
const POLL_MAX_TOTAL_MS = 15 * 60 * 1000;
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

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

// Polls GET /transcribe/{id} with bounded exponential backoff until the
// job completes, fails, is superseded (a newer analysis started, or the
// component unmounted), or exceeds POLL_MAX_TOTAL_MS. A handful of
// consecutive transient network failures are tolerated rather than failing
// the whole flow on one dropped response.
export async function pollTranscribeJob(
  jobId: string,
  options: {
    isSuperseded: () => boolean;
    onStatusChange?: (status: TranscribeJobStatusResponse, elapsedMs: number) => void;
  },
): Promise<TranscriptionResult> {
  const startedAt = Date.now();
  const deadline = startedAt + POLL_MAX_TOTAL_MS;
  let delay = POLL_INITIAL_DELAY_MS;
  let consecutiveFailures = 0;

  while (true) {
    if (options.isSuperseded()) {
      throw new TranscribeJobSupersededError();
    }
    if (Date.now() > deadline) {
      throw new TranscribeJobTimeoutError();
    }

    let status: TranscribeJobStatusResponse;
    try {
      status = await getTranscribeJob(jobId);
      consecutiveFailures = 0;
    } catch (pollError) {
      consecutiveFailures += 1;
      if (consecutiveFailures >= POLL_MAX_CONSECUTIVE_FAILURES) {
        throw pollError;
      }
      await sleep(delay);
      delay = Math.min(delay * POLL_BACKOFF_FACTOR, POLL_MAX_DELAY_MS);
      continue;
    }

    if (options.isSuperseded()) {
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

    await sleep(delay);
    delay = Math.min(delay * POLL_BACKOFF_FACTOR, POLL_MAX_DELAY_MS);
  }
}
