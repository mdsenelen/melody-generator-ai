import {
  BROWSER_MAX_DURATION_SEC,
  browserPathEligibility,
  LOW_MEMORY_MAX_DURATION_SEC,
} from "../../app/lib/browserTranscribe/eligibility";
import { toNoteEvents } from "../../app/lib/browserTranscribe/noteEvents";
import type { WorkerResponse } from "../../app/lib/browserTranscribe/protocol";
import {
  BrowserDecodeError,
  BrowserWorkerError,
  shouldFallBackToServer,
  transcribeInBrowser,
  type WorkerLike,
} from "../../app/lib/browserTranscribe/transcribeInBrowser";
import { RetryCancelledError } from "../../app/lib/retryOnWakeup";
import { TranscribeJobSupersededError } from "../../app/lib/transcribeJob";

jest.mock("../../app/lib/browserTranscribe/createWorker", () => ({
  createTranscribeWorker: jest.fn(),
}));

describe("toNoteEvents", () => {
  it("maps Basic Pitch's timed notes onto the server's NoteEvent shape", () => {
    expect(
      toNoteEvents([
        { startTimeSeconds: 1.23456, durationSeconds: 0.5, pitchMidi: 64, amplitude: 0.5 },
      ]),
    ).toEqual([{ start: 1.235, end: 1.735, pitch: 64, velocity: 64 }]);
  });

  it("clamps velocity to 1..127 like the server's _coerce_note_events", () => {
    const [quiet, loud] = toNoteEvents([
      { startTimeSeconds: 0, durationSeconds: 0.1, pitchMidi: 60, amplitude: 0 },
      { startTimeSeconds: 0, durationSeconds: 0.1, pitchMidi: 60, amplitude: 1.4 },
    ]);
    expect(quiet.velocity).toBe(1);
    expect(loud.velocity).toBe(127);
  });

  it("never produces end < start and sorts by start time", () => {
    const notes = toNoteEvents([
      { startTimeSeconds: 2, durationSeconds: 0.2, pitchMidi: 62, amplitude: 0.5 },
      { startTimeSeconds: 1, durationSeconds: -0.1, pitchMidi: 60, amplitude: 0.5 },
    ]);
    expect(notes.map((n) => n.start)).toEqual([1, 2]);
    expect(notes[0].end).toBe(notes[0].start);
  });
});

describe("browserPathEligibility", () => {
  it.each([
    [BROWSER_MAX_DURATION_SEC, undefined, true],
    [BROWSER_MAX_DURATION_SEC + 1, undefined, false],
    [BROWSER_MAX_DURATION_SEC, 8, true],
    [LOW_MEMORY_MAX_DURATION_SEC, 2, true],
    [LOW_MEMORY_MAX_DURATION_SEC + 1, 2, false],
    [LOW_MEMORY_MAX_DURATION_SEC + 1, 4, true],
  ])("duration %ss with deviceMemory %s GB -> eligible %s", (durationSec, deviceMemoryGb, ok) => {
    expect(browserPathEligibility({ durationSec, deviceMemoryGb }).eligible).toBe(ok);
  });

  it("routes to the server when the duration couldn't be probed", () => {
    expect(browserPathEligibility({ durationSec: null })).toEqual({
      eligible: false,
      reason: "duration_unknown",
    });
  });

  it("names why a long file was routed to the server", () => {
    expect(browserPathEligibility({ durationSec: 600 })).toEqual({
      eligible: false,
      reason: "too_long",
    });
    expect(browserPathEligibility({ durationSec: 200, deviceMemoryGb: 2 })).toEqual({
      eligible: false,
      reason: "low_memory_too_long",
    });
  });
});

describe("shouldFallBackToServer", () => {
  const abort = new DOMException("aborted", "AbortError");
  it.each<[string, unknown, boolean]>([
    ["abort", abort, false],
    ["superseded", new TranscribeJobSupersededError(), false],
    ["retry cancelled", new RetryCancelledError(), false],
    ["undecodable in the browser (server has ffmpeg)", new BrowserDecodeError("x"), true],
    ["worker / model / backend failure", new BrowserWorkerError("model", "x"), true],
    ["out of memory", new RangeError("Array buffer allocation failed"), true],
    ["unknown", new Error("?"), true],
    ["non-Error throw", "boom", true],
  ])("%s", (_label, error, expected) => {
    expect(shouldFallBackToServer(error)).toBe(expected);
  });
});

class FakeWorker implements WorkerLike {
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  posted: unknown[] = [];
  terminate = jest.fn();
  postMessage(message: unknown) {
    this.posted.push(message);
  }
  emit(data: WorkerResponse) {
    this.onmessage?.({ data } as MessageEvent<WorkerResponse>);
  }
}

function setup() {
  const worker = new FakeWorker();
  const audio = new Float32Array([0.1, 0.2]);
  const deps = {
    decode: jest.fn().mockResolvedValue({ audio, durationSec: 3.5 }),
    createWorker: () => worker,
  };
  return { worker, audio, deps };
}

// Let the awaited decode settle so the worker has been created and messaged.
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("transcribeInBrowser", () => {
  it("decodes, hands the audio to the worker, reports progress, and returns notes + backend", async () => {
    const { worker, audio, deps } = setup();
    const onProgress = jest.fn();
    const pending = transcribeInBrowser(new File(["x"], "a.mp3"), { onProgress, deps });
    await flush();

    expect(worker.posted[0]).toMatchObject({ type: "transcribe", audio });
    worker.emit({ type: "progress", fraction: 0.5 });
    worker.emit({
      type: "done",
      backend: "webgl",
      notes: [{ startTimeSeconds: 0, durationSeconds: 0.5, pitchMidi: 60, amplitude: 0.5 }],
    });

    await expect(pending).resolves.toEqual({
      noteEvents: [{ start: 0, end: 0.5, pitch: 60, velocity: 64 }],
      durationSec: 3.5,
      tfjsBackend: "webgl",
    });
    expect(onProgress).toHaveBeenCalledWith(0.5);
    expect(worker.terminate).toHaveBeenCalled();
  });

  it("rejects with a BrowserWorkerError carrying the failing stage", async () => {
    const { worker, deps } = setup();
    const pending = transcribeInBrowser(new File(["x"], "a.mp3"), { deps });
    await flush();
    worker.emit({ type: "error", stage: "model", message: "404 model.json" });

    await expect(pending).rejects.toMatchObject({ name: "BrowserWorkerError", stage: "model" });
    expect(worker.terminate).toHaveBeenCalled();
  });

  it("rejects when the worker script itself crashes", async () => {
    const { worker, deps } = setup();
    const pending = transcribeInBrowser(new File(["x"], "a.mp3"), { deps });
    await flush();
    worker.onerror?.({ message: "chunk load failed" } as ErrorEvent);

    await expect(pending).rejects.toBeInstanceOf(BrowserWorkerError);
  });

  it("terminates the worker and rejects with AbortError when cancelled", async () => {
    const { worker, deps } = setup();
    const controller = new AbortController();
    const pending = transcribeInBrowser(new File(["x"], "a.mp3"), {
      signal: controller.signal,
      deps,
    });
    await flush();
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(worker.terminate).toHaveBeenCalled();
  });

  it("does not start a worker when already aborted", async () => {
    const { deps } = setup();
    const createWorker = jest.fn();
    const controller = new AbortController();
    controller.abort();

    await expect(
      transcribeInBrowser(new File(["x"], "a.mp3"), {
        signal: controller.signal,
        deps: { ...deps, createWorker },
      }),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(createWorker).not.toHaveBeenCalled();
  });

  it("surfaces a decode failure as BrowserDecodeError", async () => {
    const { deps } = setup();
    deps.decode.mockRejectedValueOnce(new BrowserDecodeError("EncodingError"));
    await expect(transcribeInBrowser(new File(["x"], "a.mp3"), { deps })).rejects.toBeInstanceOf(
      BrowserDecodeError,
    );
  });
});
