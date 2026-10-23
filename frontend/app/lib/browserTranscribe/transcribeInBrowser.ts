import { RetryCancelledError } from "../retryOnWakeup";
import { TranscribeJobSupersededError, type NoteEvent, type TfjsBackend } from "../transcribeJob";
import { createTranscribeWorker } from "./createWorker";
import { decodeToMono22050, type DecodedAudio } from "./decode";
import { toNoteEvents } from "./noteEvents";
import type { WorkerFailureStage, WorkerRequest, WorkerResponse } from "./protocol";

export { BrowserDecodeError } from "./decode";

// Served from public/ by scripts/copy-ml-assets.mjs, so the model and the
// wasm binaries always match the installed package versions.
const MODEL_URL = "/ml/basic-pitch/model.json";
const WASM_PATH = "/ml/tfjs-wasm/";

// How long the worker may go without a progress/done message before it's
// presumed hung. Basic Pitch reports progress once per 2 s audio window,
// and one window takes well under a few seconds even on the cpu backend, so
// the longest legitimate silence is before the first report: TF.js backend
// init (WebGL shader compile or wasm fetch), the ~2 MB model + wasm
// download, and framing the whole clip. 30 s covers that on a slow
// connection while still handing a hung worker to the server long before
// the user gives up.
export const WORKER_WATCHDOG_MS = 30_000;

type BrowserWorkerFailureStage = WorkerFailureStage | "crash" | "timeout";

export class BrowserWorkerError extends Error {
  readonly stage: BrowserWorkerFailureStage;
  constructor(stage: BrowserWorkerFailureStage, message: string) {
    super(message);
    this.name = "BrowserWorkerError";
    this.stage = stage;
  }
}

function isCancellation(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === "AbortError") ||
    error instanceof TranscribeJobSupersededError ||
    error instanceof RetryCancelledError
  );
}

// Whether a failed browser attempt should be retried on the server rather
// than shown to the user. Only a cancellation stops quietly; everything
// else falls back, because the server path is the one that has always
// worked:
//   - BrowserDecodeError: WebAudio can't decode some containers the
//     server's ffmpeg can (and a genuinely corrupt file then fails there
//     with the server's own, already user-facing, message).
//   - BrowserWorkerError (backend init, model load, inference), RangeError
//     (allocation failure on a low-memory device), and anything unforeseen.
export function shouldFallBackToServer(error: unknown): boolean {
  return !isCancellation(error);
}

export type WorkerLike = {
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage(message: WorkerRequest, transfer: Transferable[]): void;
  terminate(): void;
};

export type BrowserTranscription = {
  noteEvents: NoteEvent[];
  durationSec: number;
  tfjsBackend: TfjsBackend;
};

type Deps = {
  decode: (file: File) => Promise<DecodedAudio>;
  createWorker: () => WorkerLike;
};

const abortError = () => new DOMException("Browser transcription cancelled", "AbortError");

export async function transcribeInBrowser(
  file: File,
  {
    signal,
    onProgress,
    deps = { decode: decodeToMono22050, createWorker: createTranscribeWorker },
  }: {
    signal?: AbortSignal;
    onProgress?: (fraction: number) => void;
    deps?: Deps;
  } = {},
): Promise<BrowserTranscription> {
  if (signal?.aborted) {
    throw abortError();
  }
  const { audio, durationSec } = await deps.decode(file);
  if (signal?.aborted) {
    throw abortError();
  }

  const worker = deps.createWorker();
  return new Promise<BrowserTranscription>((resolve, reject) => {
    let watchdog: ReturnType<typeof setTimeout> | undefined;
    const settle = (finish: () => void) => {
      clearTimeout(watchdog);
      // Terminating frees the worker's TF.js tensors / WebGL context right
      // away instead of waiting for GC.
      worker.terminate();
      signal?.removeEventListener("abort", onAbort);
      finish();
    };
    const fail = (stage: BrowserWorkerFailureStage, message: string) =>
      settle(() => reject(new BrowserWorkerError(stage, message)));
    const armWatchdog = () => {
      clearTimeout(watchdog);
      watchdog = setTimeout(
        () => fail("timeout", `Worker silent for ${WORKER_WATCHDOG_MS / 1000}s`),
        WORKER_WATCHDOG_MS,
      );
    };
    const onAbort = () => settle(() => reject(abortError()));
    signal?.addEventListener("abort", onAbort, { once: true });

    worker.onmessage = ({ data }) => {
      if (data.type === "progress") {
        armWatchdog();
        onProgress?.(data.fraction);
      } else if (data.type === "done") {
        settle(() =>
          resolve({ noteEvents: toNoteEvents(data.notes), durationSec, tfjsBackend: data.backend }),
        );
      } else {
        fail(data.stage, data.message);
      }
    };
    worker.onerror = (event) => fail("crash", event.message || "Worker crashed");
    worker.onmessageerror = () => fail("crash", "Worker message could not be deserialized");

    // Transfer, don't copy: the main thread has no further use for the PCM.
    worker.postMessage({ type: "transcribe", audio, modelUrl: MODEL_URL, wasmPath: WASM_PATH }, [
      audio.buffer,
    ]);
    armWatchdog();
  });
}
