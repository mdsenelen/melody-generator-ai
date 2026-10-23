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

export class BrowserWorkerError extends Error {
  readonly stage: WorkerFailureStage | "crash";
  constructor(stage: WorkerFailureStage | "crash", message: string) {
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
    const settle = (finish: () => void) => {
      // Terminating frees the worker's TF.js tensors / WebGL context right
      // away instead of waiting for GC.
      worker.terminate();
      signal?.removeEventListener("abort", onAbort);
      finish();
    };
    const onAbort = () => settle(() => reject(abortError()));
    signal?.addEventListener("abort", onAbort, { once: true });

    worker.onmessage = ({ data }) => {
      if (data.type === "progress") {
        onProgress?.(data.fraction);
      } else if (data.type === "done") {
        settle(() =>
          resolve({ noteEvents: toNoteEvents(data.notes), durationSec, tfjsBackend: data.backend }),
        );
      } else {
        settle(() => reject(new BrowserWorkerError(data.stage, data.message)));
      }
    };
    worker.onerror = (event) =>
      settle(() => reject(new BrowserWorkerError("crash", event.message || "Worker crashed")));

    // Transfer, don't copy: the main thread has no further use for the PCM.
    worker.postMessage({ type: "transcribe", audio, modelUrl: MODEL_URL, wasmPath: WASM_PATH }, [
      audio.buffer,
    ]);
  });
}
