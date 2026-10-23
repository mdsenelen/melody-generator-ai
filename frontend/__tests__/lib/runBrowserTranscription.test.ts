import type { BrowserTranscription } from "../../app/lib/browserTranscribe/transcribeInBrowser";
import { runBrowserTranscription } from "../../app/lib/browserTranscribe/runBrowserTranscription";
import type { RequestError } from "../../app/lib/request";
import type { TranscribeJobStatusResponse } from "../../app/lib/transcribeJob";

jest.mock("../../app/lib/browserTranscribe/createWorker", () => ({
  createTranscribeWorker: jest.fn(),
}));

const FILE = new File(["x"], "vokal izole.mp3", { type: "audio/mpeg" });
const STORED = { id: "up-1", filename: "upload_up-1.mp3" };
const BROWSER: BrowserTranscription = {
  noteEvents: [{ start: 0, end: 0.5, pitch: 60, velocity: 90 }],
  durationSec: 185.74,
  tfjsBackend: "webgl",
};
const RESULT = {
  n_notes: 1,
  duration_sec: 185.74,
  source_duration_sec: 185.74,
  truncated: false,
  midi_b64: "AAA=",
  wav_b64: null,
  midi_filename: "t.mid",
  wav_filename: "",
  note_events: BROWSER.noteEvents,
  transcribed_by: "browser" as const,
  tfjs_backend: "webgl" as const,
};
const COMPLETED: TranscribeJobStatusResponse = {
  job_id: "job-b",
  status: "completed",
  progress: 100,
  result: RESULT,
  error: null,
};

function httpError(status: number): RequestError {
  const error = new Error(`HTTP ${status}`) as RequestError;
  error.status = status;
  return error;
}

function setup() {
  const deps = {
    checkEligibility: jest.fn().mockResolvedValue({ eligible: true, durationSec: 185.74 }),
    transcribe: jest.fn().mockResolvedValue(BROWSER),
    register: jest.fn().mockResolvedValue(COMPLETED),
  };
  const ensureUploaded = jest.fn().mockResolvedValue(STORED);
  const signal = new AbortController().signal;
  return { deps, ensureUploaded, signal };
}

beforeEach(() => {
  jest.spyOn(console, "warn").mockImplementation(() => {});
});
afterEach(() => {
  jest.restoreAllMocks();
});

describe("runBrowserTranscription", () => {
  it("uploads while transcribing in the browser, then registers the notes as a completed job", async () => {
    const { deps, ensureUploaded, signal } = setup();

    const outcome = await runBrowserTranscription({ file: FILE, signal, ensureUploaded, deps });

    expect(outcome).toEqual({ kind: "completed", stored: STORED, job: COMPLETED, result: RESULT });
    // Both started before either finished -- the upload hides behind inference.
    expect(ensureUploaded).toHaveBeenCalledTimes(1);
    expect(deps.transcribe).toHaveBeenCalledWith(FILE, expect.objectContaining({ signal }));
    expect(deps.register).toHaveBeenCalledWith(
      {
        upload_id: "up-1",
        source_filename: "vokal izole.mp3",
        source_duration_sec: 185.74,
        tfjs_backend: "webgl",
        note_events: BROWSER.noteEvents,
      },
      expect.anything(),
    );
  });

  it("goes straight to the server for an ineligible file, without uploading or transcribing", async () => {
    const { deps, ensureUploaded, signal } = setup();
    deps.checkEligibility.mockResolvedValueOnce({ eligible: false, reason: "too_long" });

    const outcome = await runBrowserTranscription({ file: FILE, signal, ensureUploaded, deps });

    expect(outcome).toEqual({ kind: "fallback", reason: "too_long", stored: null });
    expect(deps.transcribe).not.toHaveBeenCalled();
    expect(ensureUploaded).not.toHaveBeenCalled();
  });

  it("falls back with the finished upload when the browser attempt fails", async () => {
    const { deps, ensureUploaded, signal } = setup();
    deps.transcribe.mockRejectedValueOnce(new Error("WebGL context lost"));

    const outcome = await runBrowserTranscription({ file: FILE, signal, ensureUploaded, deps });

    expect(outcome).toEqual({ kind: "fallback", reason: "browser_failed", stored: STORED });
    expect(deps.register).not.toHaveBeenCalled();
  });

  it.each([422, 429])("falls back (not an error) when registration returns %s", async (status) => {
    const { deps, ensureUploaded, signal } = setup();
    deps.register.mockRejectedValueOnce(httpError(status));

    const outcome = await runBrowserTranscription({ file: FILE, signal, ensureUploaded, deps });

    expect(outcome).toEqual({ kind: "fallback", reason: "register_rejected", stored: STORED });
  });

  it("propagates other registration errors", async () => {
    const { deps, ensureUploaded, signal } = setup();
    deps.register.mockRejectedValueOnce(httpError(500));

    await expect(
      runBrowserTranscription({ file: FILE, signal, ensureUploaded, deps }),
    ).rejects.toMatchObject({ status: 500 });
  });

  it("propagates a cancellation instead of falling back", async () => {
    const { deps, ensureUploaded, signal } = setup();
    deps.transcribe.mockRejectedValueOnce(new DOMException("cancelled", "AbortError"));

    await expect(
      runBrowserTranscription({ file: FILE, signal, ensureUploaded, deps }),
    ).rejects.toMatchObject({ name: "AbortError" });
  });

  it("propagates an upload failure (the server path would hit it too)", async () => {
    const { deps, ensureUploaded, signal } = setup();
    ensureUploaded.mockRejectedValueOnce(httpError(413));

    await expect(
      runBrowserTranscription({ file: FILE, signal, ensureUploaded, deps }),
    ).rejects.toMatchObject({ status: 413 });
  });
});
