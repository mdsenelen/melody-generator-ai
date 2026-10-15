import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import AnalysePage from "../../app/analyse/page";
import { analyzeClip } from "../../app/lib/analyzeClip";
import {
  RETRY_ATTEMPT_TIMEOUT_MS,
  RETRY_INITIAL_DELAY_MS,
  RETRY_MAX_TOTAL_MS,
} from "../../app/lib/retryOnWakeup";
import { useSessionStore } from "../../app/lib/session-store";
import { createTranscribeJob, pollTranscribeJob } from "../../app/lib/transcribeJob";
import { uploadFile } from "../../app/lib/upload";

function networkError() {
  return Object.assign(new Error("Couldn't reach the server."), { code: "network_error" });
}

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
  n_chunks: 3,
  note_events: [{ start: 0, end: 0.5, pitch: 60, velocity: 90 }],
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

  it("stores the completed job's id in the session, for generate-variants to reuse", async () => {
    const user = userEvent.setup();
    render(<AnalysePage />);
    await upload(user, makeFile("my-riff.wav"));

    await waitFor(() => expect(useSessionStore.getState().lastUpload?.jobId).toBe("job-1"));
  });

  it("shows the top detected notes, derived from the pitch histogram", async () => {
    mockedAnalyzeClip.mockResolvedValue({
      ...BASE_ANALYSIS,
      pitch_histogram: [0.7, 0, 0, 0, 0, 0.6, 0, 0, 0, 0.9, 0, 0],
    });

    const user = userEvent.setup();
    render(<AnalysePage />);
    await upload(user, makeFile("riff.wav"));

    await waitFor(() => expect(screen.getByText("Detected notes")).toBeInTheDocument());
    // highest-weighted pitch classes first: A (0.9), C (0.7), F (0.6)
    const notesCard = screen.getByText("Detected notes").parentElement!;
    expect(within(notesCard).getByText("A")).toBeInTheDocument();
    expect(within(notesCard).getByText("C")).toBeInTheDocument();
    expect(within(notesCard).getByText("F")).toBeInTheDocument();
  });

  it("shows the chord progression in playback order plus a most-used summary", async () => {
    mockedAnalyzeClip.mockResolvedValue({
      ...BASE_ANALYSIS,
      // playback order with an immediate repeat
      detected_chords: ["C", "C", "Am", "F", "G", "C"],
    });

    const user = userEvent.setup();
    render(<AnalysePage />);
    await upload(user, makeFile("riff.wav"));

    await waitFor(() => expect(screen.getByText("Chord progression")).toBeInTheDocument());
    // consecutive "C C" collapses; order preserved -> C, Am, F, G, C (5 chips)
    const chordButtons = screen.getAllByRole("button", {
      name: /show its fingering diagram/i,
    });
    expect(chordButtons).toHaveLength(5);
    // most-used: C appears 3x
    expect(screen.getByText(/Most used:/)).toHaveTextContent("C ×3");
  });

  it("keeps the transcription area hidden until audio has been transcribed", async () => {
    const pendingPoll = deferred<typeof BASE_RESULT>();
    mockedPollJob.mockReturnValue(pendingPoll.promise);

    const user = userEvent.setup();
    render(<AnalysePage />);

    expect(
      screen.queryByText(/upload or record audio to see the transcription and downloads/i),
    ).not.toBeInTheDocument();

    await upload(user, makeFile("pending.wav"));
    await waitFor(() => expect(screen.getByText(/queued for transcription/i)).toBeInTheDocument());
    expect(
      screen.queryByText(/upload or record audio to see the transcription and downloads/i),
    ).not.toBeInTheDocument();
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

  it("shows a calm startup message, not an error, while the job waits out the warm-up window", async () => {
    const pending = deferred<typeof BASE_RESULT>();
    mockedPollJob.mockImplementation((jobId: string, options: { onStatusChange?: Function }) => {
      options.onStatusChange?.(
        {
          job_id: jobId,
          status: "queued",
          progress: 0,
          result: null,
          error: null,
          waiting_reason: "warming_up",
        },
        1000,
      );
      return pending.promise;
    });

    const user = userEvent.setup();
    render(<AnalysePage />);
    await upload(user, makeFile("my-riff.wav"));

    await waitFor(() => expect(screen.getByText(/starting up/i)).toBeInTheDocument());
    expect(screen.getByText(/2 to 4 minutes/i)).toBeInTheDocument();
    // Not an error -- the calm text only, no red ErrorToast (which would
    // show this exact sanitized message for a real terminal failure).
    expect(screen.queryByText(/transcription failed/i)).not.toBeInTheDocument();
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

  describe("server waking up during job creation", () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it("shows a calm waking-up status instead of the raw error, then completes once the retry lands", async () => {
      mockedCreateJob.mockRejectedValueOnce(networkError()).mockResolvedValueOnce({
        job_id: "job-1",
        status: "queued",
        progress: 0,
        result: null,
        error: null,
      });

      render(<AnalysePage />);
      fireEvent.change(screen.getByLabelText(/upload audio/i), {
        target: { files: [makeFile("riff.wav")] },
      });

      await waitFor(() => expect(screen.getByText(/still waking up/i)).toBeInTheDocument());
      expect(screen.queryByText(/couldn't transcribe/i)).not.toBeInTheDocument();

      await act(() => jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS));

      await waitFor(() =>
        expect(screen.getByRole("heading", { name: "riff.wav" })).toBeInTheDocument(),
      );
    });

    it("shows the real transcription error only once the retry budget is exhausted", async () => {
      mockedCreateJob.mockRejectedValue(networkError());

      render(<AnalysePage />);
      fireEvent.change(screen.getByLabelText(/upload audio/i), {
        target: { files: [makeFile("riff.wav")] },
      });

      await waitFor(() => expect(screen.getByText(/still waking up/i)).toBeInTheDocument());

      await act(() => jest.advanceTimersByTimeAsync(RETRY_MAX_TOTAL_MS + RETRY_ATTEMPT_TIMEOUT_MS));

      await waitFor(() =>
        expect(screen.getByText("Couldn't reach the server.")).toBeInTheDocument(),
      );
    });
  });
});
