import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import AnalysePage from "../../app/analyse/page";
import { createTranscribeJob, pollTranscribeJob } from "../../app/lib/transcribeJob";
import { uploadFile } from "../../app/lib/upload";

jest.mock("../../app/lib/upload", () => ({ uploadFile: jest.fn() }));
jest.mock("../../app/lib/transcribeJob", () => {
  const actual = jest.requireActual("../../app/lib/transcribeJob");
  return { ...actual, createTranscribeJob: jest.fn(), pollTranscribeJob: jest.fn() };
});

const mockedUploadFile = uploadFile as jest.Mock;
const mockedCreateJob = createTranscribeJob as jest.Mock;
const mockedPollJob = pollTranscribeJob as jest.Mock;

const BASE_RESULT = {
  n_notes: 5,
  duration_sec: 2.0,
  source_duration_sec: 2.0,
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

async function uploadAndStartAnalysis(user: ReturnType<typeof userEvent.setup>, file: File) {
  const input = screen.getByLabelText(/upload audio/i) as HTMLInputElement;
  await user.upload(input, file);
}

describe("Analyse page transcription flow", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.URL.createObjectURL = jest.fn().mockReturnValue("blob:fake-url");
    global.URL.revokeObjectURL = jest.fn();
  });

  it("uploads, creates a job, polls, and renders the completed analysis", async () => {
    mockedUploadFile.mockResolvedValue({ id: "up-1", filename: "upload_up-1.wav" });
    mockedCreateJob.mockResolvedValue({
      job_id: "job-1",
      status: "queued",
      progress: 0,
      result: null,
      error: null,
    });
    mockedPollJob.mockResolvedValue(BASE_RESULT);

    const user = userEvent.setup();
    render(<AnalysePage />);

    await uploadAndStartAnalysis(user, makeFile("my-riff.wav"));

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "my-riff.wav" })).toBeInTheDocument(),
    );
    expect(screen.getByText(/C major/)).toBeInTheDocument();
    // The file was already uploaded (mockedUploadFile above) -- job
    // creation must reference that upload instead of re-sending the file.
    expect(mockedCreateJob).toHaveBeenCalledWith(expect.any(File), {
      id: "up-1",
      filename: "upload_up-1.wav",
    });
    expect(mockedPollJob).toHaveBeenCalledWith(
      "job-1",
      expect.objectContaining({ isSuperseded: expect.any(Function) }),
    );
  });

  it("shows the job's error message when transcription fails", async () => {
    mockedUploadFile.mockResolvedValue({ id: "up-1", filename: "upload_up-1.wav" });
    mockedCreateJob.mockResolvedValue({
      job_id: "job-1",
      status: "queued",
      progress: 0,
      result: null,
      error: null,
    });
    mockedPollJob.mockRejectedValue(new Error("Could not decode audio"));

    const user = userEvent.setup();
    render(<AnalysePage />);

    await uploadAndStartAnalysis(user, makeFile("bad.wav"));

    await waitFor(() => expect(screen.getByText("Could not decode audio")).toBeInTheDocument());
  });

  it("only renders the result of the most recent analysis (stale request protection)", async () => {
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
    mockedPollJob.mockImplementation((jobId: string) => {
      if (jobId === "job-1") {
        return firstPoll.promise;
      }
      return Promise.resolve({ ...BASE_RESULT, key: "D minor" });
    });

    const user = userEvent.setup();
    render(<AnalysePage />);

    await uploadAndStartAnalysis(user, makeFile("first.wav"));
    await waitFor(() => expect(mockedPollJob).toHaveBeenCalledWith("job-1", expect.anything()));

    // Start a second analysis before the first job's poll has resolved.
    await uploadAndStartAnalysis(user, makeFile("second.wav"));
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "second.wav" })).toBeInTheDocument(),
    );
    expect(screen.getByText(/D minor/)).toBeInTheDocument();

    // The first (superseded) job resolving afterward must not clobber the
    // second job's already-rendered result.
    firstPoll.resolve({ ...BASE_RESULT, key: "F major" });
    await Promise.resolve();
    await Promise.resolve();

    expect(screen.getByRole("heading", { name: "second.wav" })).toBeInTheDocument();
    expect(screen.queryByText(/F major/)).not.toBeInTheDocument();
  });

  it("does not update state after unmounting mid-poll", async () => {
    mockedUploadFile.mockResolvedValue({ id: "up-1", filename: "upload_up-1.wav" });
    mockedCreateJob.mockResolvedValue({
      job_id: "job-1",
      status: "queued",
      progress: 0,
      result: null,
      error: null,
    });
    const poll = deferred<typeof BASE_RESULT>();
    mockedPollJob.mockReturnValue(poll.promise);

    const consoleError = jest.spyOn(console, "error").mockImplementation(() => {});
    const user = userEvent.setup();
    const { unmount } = render(<AnalysePage />);

    await uploadAndStartAnalysis(user, makeFile("clip.wav"));
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
