import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import AnalysePage from "../../app/analyse/page";
import { analyzeClip } from "../../app/lib/analyzeClip";
import { createTranscribeJob, pollTranscribeJob } from "../../app/lib/transcribeJob";
import { uploadFile } from "../../app/lib/upload";

jest.mock("../../app/lib/upload", () => ({ uploadFile: jest.fn() }));
jest.mock("../../app/lib/analyzeClip", () => ({ analyzeClip: jest.fn() }));
jest.mock("../../app/lib/transcribeJob", () => {
  const actual = jest.requireActual("../../app/lib/transcribeJob");
  return { ...actual, createTranscribeJob: jest.fn(), pollTranscribeJob: jest.fn() };
});

const mockedUploadFile = uploadFile as jest.Mock;
const mockedCreateJob = createTranscribeJob as jest.Mock;
const mockedPollJob = pollTranscribeJob as jest.Mock;
const mockedAnalyzeClip = analyzeClip as jest.Mock;

const BASE_RESULT = {
  n_notes: 5,
  duration_sec: 90.0,
  source_duration_sec: 90.0,
  truncated: false,
  midi_b64: "AAA=",
  wav_b64: null,
  midi_filename: "t.mid",
  wav_filename: "",
  mood_label: "happy" as const,
  mood_idx: 0,
  detected_chords: ["C", "G"],
  key: "C major",
  pitch_histogram: new Array(12).fill(0.1),
  tempo_bpm: 120,
  average_pitch: 61,
  note_events: [],
};

const BASE_ANALYSIS = {
  tempo_bpm: 120,
  average_pitch: 61,
  mood_idx: 0,
  mood_label: "happy" as const,
  key: "C major",
  pitch_histogram: new Array(12).fill(0.1),
  detected_chords: ["C", "G"],
  n_notes: 5,
  clip_start_sec: 0,
  clip_end_sec: 60,
};

function makeFile(name = "clip.wav") {
  return new File(["audio-bytes"], name, { type: "audio/wav" });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function upload(user: ReturnType<typeof userEvent.setup>, file: File) {
  const input = screen.getByLabelText(/upload audio/i) as HTMLInputElement;
  await user.upload(input, file);
}

describe("Analyse page: transcription + clip analysis", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.URL.createObjectURL = jest.fn().mockReturnValue("blob:fake-url");
    global.URL.revokeObjectURL = jest.fn();
    mockedUploadFile.mockResolvedValue({ id: "up-1", filename: "upload_up-1.wav" });
    mockedCreateJob.mockResolvedValue({
      job_id: "job-1",
      status: "queued",
      progress: 0,
      result: null,
      error: null,
    });
    mockedPollJob.mockResolvedValue(BASE_RESULT);
    mockedAnalyzeClip.mockResolvedValue(BASE_ANALYSIS);
  });

  it("transcribes, then auto-runs a default-window analysis", async () => {
    const user = userEvent.setup();
    render(<AnalysePage />);

    await upload(user, makeFile("my-riff.wav"));

    // transcription card
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "my-riff.wav" })).toBeInTheDocument(),
    );
    expect(screen.getByText(/5 notes/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /view & download result/i })).toHaveAttribute(
      "href",
      "/result/job-1",
    );
    expect(screen.queryByRole("button", { name: /download midi/i })).not.toBeInTheDocument();

    // analysis card, from /api/analyze -- default window [0, 60]
    await waitFor(() => expect(screen.getByText(/C major/)).toBeInTheDocument());
    expect(mockedAnalyzeClip).toHaveBeenCalledWith(
      { job_id: "job-1", clip_start_sec: 0, clip_end_sec: 60 },
      expect.objectContaining({ signal: expect.any(Object) }),
    );
    expect(screen.getByText(/Mood: happy/)).toBeInTheDocument();
  });

  it("re-analyses a different clip window on commit", async () => {
    mockedAnalyzeClip.mockResolvedValueOnce(BASE_ANALYSIS).mockResolvedValueOnce({
      ...BASE_ANALYSIS,
      key: "A minor",
      clip_start_sec: 10,
      clip_end_sec: 20,
    });

    const user = userEvent.setup();
    render(<AnalysePage />);
    await upload(user, makeFile("riff.wav"));
    await waitFor(() => expect(screen.getByText(/C major/)).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText(/analysis window start/i), { target: { value: "10" } });
    fireEvent.change(screen.getByLabelText(/analysis window end/i), { target: { value: "20" } });

    await user.click(screen.getByRole("button", { name: /analyse this section/i }));

    await waitFor(() => expect(screen.getByText(/A minor/)).toBeInTheDocument());
    expect(mockedAnalyzeClip).toHaveBeenLastCalledWith(
      { job_id: "job-1", clip_start_sec: 10, clip_end_sec: 20 },
      expect.anything(),
    );
  });

  it("shows the transcription error when the job fails, and no analysis card", async () => {
    mockedPollJob.mockRejectedValue(new Error("Could not decode audio"));

    const user = userEvent.setup();
    render(<AnalysePage />);
    await upload(user, makeFile("bad.wav"));

    await waitFor(() => expect(screen.getByText("Could not decode audio")).toBeInTheDocument());
    expect(screen.queryByText(/clip analysis/i)).not.toBeInTheDocument();
    expect(mockedAnalyzeClip).not.toHaveBeenCalled();
  });

  it("surfaces an analysis failure without losing the transcription", async () => {
    mockedAnalyzeClip.mockRejectedValue(new Error("analyze boom"));

    const user = userEvent.setup();
    render(<AnalysePage />);
    await upload(user, makeFile("riff.wav"));

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "riff.wav" })).toBeInTheDocument(),
    );
    await waitFor(() => expect(screen.getByText("analyze boom")).toBeInTheDocument());
    // transcription + downloads still there
    expect(screen.getByRole("link", { name: /view & download result/i })).toBeInTheDocument();
  });

  it("only renders the most recent transcription (stale request protection)", async () => {
    mockedUploadFile
      .mockResolvedValueOnce({ id: "up-1", filename: "upload_up-1.wav" })
      .mockResolvedValueOnce({ id: "up-2", filename: "upload_up-2.wav" });
    mockedCreateJob
      .mockResolvedValueOnce({
        job_id: "job-1",
        status: "queued",
        progress: 0,
        result: null,
        error: null,
      })
      .mockResolvedValueOnce({
        job_id: "job-2",
        status: "queued",
        progress: 0,
        result: null,
        error: null,
      });

    const firstPoll = deferred<typeof BASE_RESULT>();
    mockedPollJob.mockImplementation((jobId: string) =>
      jobId === "job-1" ? firstPoll.promise : Promise.resolve(BASE_RESULT),
    );

    const user = userEvent.setup();
    render(<AnalysePage />);

    await upload(user, makeFile("first.wav"));
    await waitFor(() => expect(mockedPollJob).toHaveBeenCalledWith("job-1", expect.anything()));

    await upload(user, makeFile("second.wav"));
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "second.wav" })).toBeInTheDocument(),
    );

    firstPoll.resolve(BASE_RESULT);
    await Promise.resolve();
    await Promise.resolve();

    expect(screen.getByRole("heading", { name: "second.wav" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "first.wav" })).not.toBeInTheDocument();
  });

  it("does not update state after unmounting mid-poll", async () => {
    const poll = deferred<typeof BASE_RESULT>();
    mockedPollJob.mockReturnValue(poll.promise);

    const consoleError = jest.spyOn(console, "error").mockImplementation(() => {});
    const user = userEvent.setup();
    const { unmount } = render(<AnalysePage />);

    await upload(user, makeFile("clip.wav"));
    await waitFor(() => expect(mockedPollJob).toHaveBeenCalled());

    unmount();
    poll.resolve(BASE_RESULT);
    await Promise.resolve();
    await Promise.resolve();

    const stateUpdateWarnings = consoleError.mock.calls.filter((call) =>
      String(call[0]).includes("state update"),
    );
    expect(stateUpdateWarnings).toHaveLength(0);
    consoleError.mockRestore();
  });
});
