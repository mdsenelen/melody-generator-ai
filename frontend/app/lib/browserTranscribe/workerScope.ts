import type { WorkerFailureStage, WorkerRequest, WorkerResponse } from "./protocol";

// Framework-free pieces of transcribe.worker.ts, split out so jest can test
// them without loading TF.js.

// @tensorflow/tfjs-core@3.21.0 dist/platforms/platform_browser.js:53 --
// setTimeoutCustom does `if (!window || ...)`, which throws "window is not
// defined" in a Worker (it's reached from the WebGL backend's async reads
// during evaluateModel). Defining `window` as an undefined property makes
// that check true, so tfjs falls through to plain setTimeout. The value
// stays undefined -- `typeof window` is unchanged, so nothing that sniffs
// for a DOM starts believing it has one. Can't upgrade past tfjs 3.x:
// @spotify/basic-pitch@1.0.1 requires ^3.2.0.
export function installWindowShim(scope: object) {
  if (!("window" in scope)) {
    Object.defineProperty(scope, "window", { value: undefined, configurable: true });
  }
}

export class StageError extends Error {
  constructor(
    readonly stage: WorkerFailureStage,
    cause: unknown,
  ) {
    super(cause instanceof Error ? cause.message : String(cause));
  }
}

export type TranscribeScope = Pick<EventTarget, "addEventListener"> & {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: WorkerResponse): void;
};

function toErrorResponse(error: unknown): WorkerResponse {
  return {
    type: "error",
    stage: error instanceof StageError ? error.stage : "inference",
    message: error instanceof Error ? error.message : String(error),
  };
}

// Every failure becomes an {type:"error"} message, so the main thread falls
// back to the server instead of waiting on a worker that will never answer.
export function attachTranscribeHandlers(
  scope: TranscribeScope,
  transcribe: (request: WorkerRequest) => Promise<void>,
) {
  scope.onmessage = (event) => {
    // Promise.resolve().then() also routes a synchronous throw to .catch.
    Promise.resolve()
      .then(() => transcribe(event.data))
      .catch((error: unknown) => scope.postMessage(toErrorResponse(error)));
  };
  // A rejection inside TF.js that no caller awaits never reaches the catch
  // above -- the request's own promise just never settles.
  scope.addEventListener("unhandledrejection", (event) => {
    event.preventDefault();
    const reason: unknown = (event as Partial<PromiseRejectionEvent>).reason;
    scope.postMessage(toErrorResponse(reason));
  });
}
