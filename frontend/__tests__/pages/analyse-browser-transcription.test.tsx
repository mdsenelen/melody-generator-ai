import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import AnalysePage from "../../app/analyse/page";
import { analyzeClip } from "../../app/lib/analyzeClip";
import { createTranscribeWorker } from "../../app/lib/browserTranscribe/createWorker";
import { decodeToMono22050 } from "../../app/lib/browserTranscribe/decode";
import { checkBrowserEligibility } from "../../app/lib/browserTranscribe/eligibility";
import type { WorkerRequest, WorkerResponse } from "../../app/lib/browserTranscribe/protocol";
import {
  WORKER_WATCHDOG_MS,
  type WorkerLike,
} from "../../app/lib/browserTranscribe/transcribeInBrowser";
import {
  createTranscribeJob,
  pollTranscribeJob,
  registerBrowserTranscription,
} from "../../app/lib/transcribeJob";
import { uploadFile } from "../../app/lib/upload";
import { waitForBackend } from "../../app/lib/waitForBackend";

// The real orchestration (runBrowserTranscription) and worker runner
// (transcribeInBrowser) run here; only the three things jsdom can't do are
// stubbed: probing duration with an <audio> element, WebAudio decoding, and
// spawning a Worker.
jest.mock("../../app/lib/browserTranscribe/eligibility", () => ({
  checkBrowserEligibility: jest.fn(),
}));
jest.mock("../../app/lib/browserTranscribe/decode", () => {
  const actual = jest.requireActual("../../app/lib/browserTranscribe/decode");
  return { ...actual, decodeToMono22050: jest.fn() };
});
jest.mock("../../app/lib/browserTranscribe/createWorker", () => ({
  createTranscribeWorker: jest.fn(),
}));
jest.mock("../../app/lib/upload", () => ({ uploadFile: jest.fn() }));
jest.mock("../../app/lib/waitForBackend", () => ({ waitForBackend: jest.fn() }));
jest.mock("../../app/lib/analyzeClip", () => ({ analyzeClip: jest.fn() }));
jest.mock("../../app/lib/transcribeJob", () => {
  const actual = jest.requireActual("../../app/lib/transcribeJob");
  return {
    ...actual,
    createTranscribeJob: jest.fn(),
    pollTranscribeJob: jest.fn(),
    registerBrowserTranscription: jest.fn(),
  };
});

const mockedEligibility = checkBrowserEligibility as jest.Mock;
const mockedDecode = decodeToMono22050 as jest.Mock;
const mockedCreateWorker = createTranscribeWorker as jest.Mock;
const mockedRegister = registerBrowserTranscription as jest.Mock;
const mockedCreateJob = createTranscribeJob as jest.Mock;
const mockedPollJob = pollTranscribeJob as jest.Mock;

class FakeWorker implements WorkerLike {
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  terminate = jest.fn();
  // Set per test: how the "worker" answers once it's handed the audio.
  reply: ((worker: FakeWorker) => void) | null = null;
  postMessage(_message: WorkerRequest) {
    queueMicrotask(() => this.reply?.(this));
  }
  emit(data: WorkerResponse) {
    this.onmessage?.({ data } as MessageEvent<WorkerResponse>);
  }
}

const BROWSER_NOTES = [
  { startTimeSeconds: 0, durationSeconds: 0.5, pitchMidi: 60, amplitude: 0.5 },
];

function result(overrides: Record<string, unknown> = {}) {
  return {
    n_notes: 1,
    duration_sec: 90,
    source_duration_sec: 90,
    truncated: false,
    midi_b64: "AAA=",
    wav_b64: null,
    midi_filename: "t.mid",
    wav_filename: "",
    note_events: [{ start: 0, end: 0.5, pitch: 60, velocity: 64 }],
    ...overrides,
  };
}

const ANALYSIS = {
  tempo_bpm: 120,
  average_pitch: 61,
  mood_idx: 0,
  mood_label: "happy" as const,
  key: "C major",
  pitch_histogram: new Array(12).fill(0.1),
  detected_chords: ["C"],
  n_notes: 1,
  clip_start_sec: 0,
  clip_end_sec: 60,
};

let worker: FakeWorker;

async function uploadClip(user = userEvent.setup()) {
  await user.upload(
    screen.getByLabelText(/upload audio/i) as HTMLInputElement,
    new File(["audio-bytes"], "vokal izole.mp3", { type: "audio/mpeg" }),
  );
}

function expectServerPathWithoutErrorToast() {
  return waitFor(() => {
    expect(mockedCreateJob).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("link", { name: /view & download result/i })).toHaveAttribute(
      "href",
      "/result/job-server",
    );
    expect(screen.queryByText(/couldn't transcribe/i)).not.toBeInTheDocument();
  });
}

describe("Analyse page: transcription in the browser", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(console, "warn").mockImplementation(() => {});
    global.URL.createObjectURL = jest.fn().mockReturnValue("blob:fake-url");
    global.URL.revokeObjectURL = jest.fn();

    worker = new FakeWorker();
    worker.reply = (w) => w.emit({ type: "done", notes: BROWSER_NOTES, backend: "webgl" });
    mockedCreateWorker.mockImplementation(() => worker);
    mockedEligibility.mockResolvedValue({ eligible: true, durationSec: 90 });
    mockedDecode.mockResolvedValue({ audio: new Float32Array(4), durationSec: 90 });
    mockedRegister.mockResolvedValue({
      job_id: "job-browser",
      status: "completed",
      progress: 100,
      result: result({ transcribed_by: "browser", tfjs_backend: "webgl" }),
      error: null,
    });

    (waitForBackend as jest.Mock).mockResolvedValue(undefined);
    (uploadFile as jest.Mock).mockResolvedValue({ id: "up-1", filename: "upload_up-1.mp3" });
    mockedCreateJob.mockResolvedValue({
      job_id: "job-server",
      status: "queued",
      progress: 0,
      result: null,
      error: null,
    });
    mockedPollJob.mockResolvedValue(result({ transcribed_by: "server" }));
    (analyzeClip as jest.Mock).mockResolvedValue(ANALYSIS);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("registers the browser's notes and never creates a server transcription job", async () => {
    render(<AnalysePage />);
    await uploadClip();

    await waitFor(() =>
      expect(screen.getByRole("link", { name: /view & download result/i })).toHaveAttribute(
        "href",
        "/result/job-browser",
      ),
    );
    expect(mockedRegister).toHaveBeenCalledWith(
      expect.objectContaining({
        upload_id: "up-1",
        source_filename: "vokal izole.mp3",
        tfjs_backend: "webgl",
        note_events: [{ start: 0, end: 0.5, pitch: 60, velocity: 64 }],
      }),
      expect.anything(),
    );
    expect(mockedCreateJob).not.toHaveBeenCalled();
    expect(mockedPollJob).not.toHaveBeenCalled();
    expect(worker.terminate).toHaveBeenCalled();
    await waitFor(() =>
      expect(analyzeClip).toHaveBeenCalledWith(
        expect.objectContaining({ job_id: "job-browser" }),
        expect.anything(),
      ),
    );
  });

  it("logs path, backend, note count and pick-to-ready time for a browser transcription", async () => {
    const info = jest.spyOn(console, "info").mockImplementation(() => {});
    render(<AnalysePage />);
    await uploadClip();

    await waitFor(() =>
      expect(info).toHaveBeenCalledWith(
        expect.stringMatching(
          /^\[transcribe\] path=browser backend=webgl notes=1 elapsed=\d+\.\ds$/,
        ),
      ),
    );
  });

  it("logs path=server backend=none when the clip was transcribed on the server", async () => {
    const info = jest.spyOn(console, "info").mockImplementation(() => {});
    mockedEligibility.mockResolvedValueOnce({ eligible: false, reason: "too_long" });
    render(<AnalysePage />);
    await uploadClip();

    await waitFor(() =>
      expect(info).toHaveBeenCalledWith(
        expect.stringMatching(/^\[transcribe\] path=server backend=none notes=1 elapsed=\d+\.\ds$/),
      ),
    );
  });

  it("shows browser progress in the status", async () => {
    worker.reply = (w) => w.emit({ type: "progress", fraction: 0.42 });
    render(<AnalysePage />);
    await uploadClip();

    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent(
        /transcribing vokal izole\.mp3 in your browser.*40%/i,
      ),
    );
  });

  it("falls back to the server, with no error toast, when the browser attempt fails", async () => {
    worker.reply = (w) => w.emit({ type: "error", stage: "model", message: "404 model.json" });
    render(<AnalysePage />);
    await uploadClip();

    await expectServerPathWithoutErrorToast();
    expect(mockedRegister).not.toHaveBeenCalled();
    expect(mockedCreateJob).toHaveBeenCalledWith(
      expect.any(File),
      { id: "up-1", filename: "upload_up-1.mp3" },
      expect.anything(),
    );
  });

  it("falls back to the server, with no error toast, when registration returns 422", async () => {
    mockedRegister.mockRejectedValueOnce(
      Object.assign(new Error("note_events extend past source_duration_sec"), { status: 422 }),
    );
    render(<AnalysePage />);
    await uploadClip();

    await expectServerPathWithoutErrorToast();
  });

  it.each([404, 405])(
    "falls back to the server, with no error toast, when the backend predates POST /api/transcriptions (%s)",
    async (status) => {
      mockedRegister.mockRejectedValueOnce(
        Object.assign(new Error(`Request failed (${status})`), { status }),
      );
      render(<AnalysePage />);
      await uploadClip();

      await expectServerPathWithoutErrorToast();
    },
  );

  it("sends an ineligible file straight to the server without starting a worker", async () => {
    mockedEligibility.mockResolvedValueOnce({ eligible: false, reason: "too_long" });
    render(<AnalysePage />);
    await uploadClip();

    await expectServerPathWithoutErrorToast();
    expect(mockedCreateWorker).not.toHaveBeenCalled();
    expect(mockedDecode).not.toHaveBeenCalled();
  });

  it("a silent worker is terminated by the watchdog and the clip finishes on the server", async () => {
    // The real hang: tfjs threw inside the worker where nothing caught it,
    // so no message ever came back and the UI sat at 0% forever.
    jest.useFakeTimers();
    try {
      worker.reply = (w) => w.emit({ type: "progress", fraction: 0 }); // then silence
      render(<AnalysePage />);
      await uploadClip(userEvent.setup({ advanceTimers: jest.advanceTimersByTime }));
      await waitFor(() => expect(mockedCreateWorker).toHaveBeenCalled());
      expect(mockedCreateJob).not.toHaveBeenCalled();

      await jest.advanceTimersByTimeAsync(WORKER_WATCHDOG_MS);

      expect(worker.terminate).toHaveBeenCalled();
      await expectServerPathWithoutErrorToast();
      expect(mockedRegister).not.toHaveBeenCalled();
    } finally {
      jest.useRealTimers();
    }
  });

  it("terminates the worker on cancel (unmount) and makes no further state updates", async () => {
    const consoleError = jest.spyOn(console, "error").mockImplementation(() => {});
    worker.reply = null; // the worker never answers
    const { unmount } = render(<AnalysePage />);
    await uploadClip();
    await waitFor(() => expect(mockedCreateWorker).toHaveBeenCalled());

    unmount();

    expect(worker.terminate).toHaveBeenCalled();
    // Let the rejected promise chain settle; nothing may log or fall back.
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(mockedRegister).not.toHaveBeenCalled();
    expect(mockedCreateJob).not.toHaveBeenCalled();
    expect(consoleError).not.toHaveBeenCalled();
  });
});
