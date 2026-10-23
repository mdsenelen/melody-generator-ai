import { runInNewContext } from "node:vm";
import type { WorkerRequest, WorkerResponse } from "../../app/lib/browserTranscribe/protocol";
import {
  attachTranscribeHandlers,
  installWindowShim,
  StageError,
  type TranscribeScope,
} from "../../app/lib/browserTranscribe/workerScope";

describe("installWindowShim", () => {
  it("makes a bare `!window` evaluate true instead of throwing ReferenceError", () => {
    // A fresh vm context has no `window`, like a Worker's global scope.
    const context: Record<string, unknown> = {};
    expect(() => runInNewContext("!window", context)).toThrow(/window is not defined/);

    installWindowShim(context);

    expect(runInNewContext("!window", context)).toBe(true);
    expect(runInNewContext("typeof window", context)).toBe("undefined");
  });

  it("leaves a real window alone", () => {
    const realWindow = {};
    const scope = { window: realWindow };
    installWindowShim(scope);
    expect(scope.window).toBe(realWindow);
  });
});

class FakeScope extends EventTarget implements TranscribeScope {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null = null;
  posted: WorkerResponse[] = [];
  postMessage(message: WorkerResponse) {
    this.posted.push(message);
  }
  receive(data: WorkerRequest) {
    this.onmessage?.({ data } as MessageEvent<WorkerRequest>);
  }
}

const REQUEST: WorkerRequest = {
  type: "transcribe",
  audio: new Float32Array(1),
  modelUrl: "/m",
  wasmPath: "/w/",
};

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("attachTranscribeHandlers", () => {
  it("posts an error carrying the stage when transcribe rejects with a StageError", async () => {
    const scope = new FakeScope();
    attachTranscribeHandlers(scope, () =>
      Promise.reject(new StageError("model", new Error("404 model.json"))),
    );
    scope.receive(REQUEST);
    await flush();

    expect(scope.posted).toEqual([{ type: "error", stage: "model", message: "404 model.json" }]);
  });

  it("posts an inference error when transcribe rejects with anything else", async () => {
    const scope = new FakeScope();
    attachTranscribeHandlers(scope, () => Promise.reject("boom"));
    scope.receive(REQUEST);
    await flush();

    expect(scope.posted).toEqual([{ type: "error", stage: "inference", message: "boom" }]);
  });

  it("posts an inference error when transcribe throws synchronously", async () => {
    const scope = new FakeScope();
    attachTranscribeHandlers(scope, () => {
      throw new Error("sync");
    });
    scope.receive(REQUEST);
    await flush();

    expect(scope.posted).toEqual([{ type: "error", stage: "inference", message: "sync" }]);
  });

  it("posts an error for a rejection nothing awaited (the tfjs `window` crash)", () => {
    const scope = new FakeScope();
    attachTranscribeHandlers(scope, () => new Promise(() => {}));
    const event = Object.assign(new Event("unhandledrejection", { cancelable: true }), {
      reason: new ReferenceError("window is not defined"),
    });
    scope.dispatchEvent(event);

    expect(scope.posted).toEqual([
      { type: "error", stage: "inference", message: "window is not defined" },
    ]);
    expect(event.defaultPrevented).toBe(true);
  });
});
