import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import GenerateVariantsPage from "../../app/generate-variants/page";
import { runBrowserTranscription } from "../../app/lib/browserTranscribe/runBrowserTranscription";
import { uploadWhenAwake } from "../../app/lib/uploadWhenAwake";

// Uploading on Variants used to send the clip with no job, so the server ran
// Basic Pitch inside the 60 s generation request and timed out ("Generation
// timed out after 60s"). The page now transcribes in the browser first --
// the same runBrowserTranscription /analyse uses, covered by its own tests --
// and generates from the resulting job, which skips server-side Basic Pitch.
jest.mock("../../app/lib/browserTranscribe/runBrowserTranscription", () => ({
  runBrowserTranscription: jest.fn(),
}));
jest.mock("../../app/lib/uploadWhenAwake", () => {
  const actual = jest.requireActual("../../app/lib/uploadWhenAwake");
  return { ...actual, uploadWhenAwake: jest.fn() };
});

const UPLOAD = { id: "0123456789abcdef0123456789abcdef", filename: "upload_0123.mp3" };
const SONG = new File([new Uint8Array(1024)], "song.mp3", { type: "audio/mpeg" });
const RECORDING = new File([new Uint8Array(64)], "recording.webm", { type: "audio/webm" });

jest.mock("../../components/upload-button", () => ({
  UploadButton: ({
    onUploadSuccess,
  }: {
    onUploadSuccess: (payload: {
      id: string;
      filename: string;
      file: File;
      pickedAt: number;
    }) => void;
  }) => (
    <button type="button" onClick={() => onUploadSuccess({ ...UPLOAD, file: SONG, pickedAt: 0 })}>
      Fake upload
    </button>
  ),
}));

jest.mock("../../components/audio-recorder", () => ({
  AudioRecorder: ({ onRecordingComplete }: { onRecordingComplete: (file: File) => void }) => (
    <button type="button" onClick={() => onRecordingComplete(RECORDING)}>
      Fake recording
    </button>
  ),
}));

jest.mock("../../hooks/use-midi-player", () => ({
  useMidiPlayer: () => ({
    state: { status: "idle", durationSec: 3 },
    toggle: jest.fn(),
    stop: jest.fn(),
  }),
}));

const mockedRun = runBrowserTranscription as jest.Mock;
const mockedUploadWhenAwake = uploadWhenAwake as jest.Mock;

function completed(jobId: string, stored = UPLOAD) {
  return {
    kind: "completed",
    stored,
    job: { job_id: jobId, status: "completed", progress: 100, result: {}, error: null },
    result: {},
  };
}

function variantsResponse() {
  return {
    ok: true,
    status: 200,
    headers: new Headers({ "content-type": "application/json" }),
    text: async () =>
      JSON.stringify({
        n_variants: 1,
        temperatures: [0.7],
        mood_idx: 0,
        mood_label: "happy",
        model_status: {
          cvae: { path: "cvae", exists: true, size_mb: 1, loaded: true },
          iddm_ppo: { path: "iddm", exists: true, size_mb: 1, loaded: true },
          device: "cpu",
          load_error: null,
          fluidsynth_available: true,
        },
        variants: [],
        job_id: "job-variants",
      }),
  };
}

function sentForm(call = 0): FormData {
  return (global.fetch as jest.Mock).mock.calls[call][1].body as FormData;
}

beforeEach(() => {
  mockedRun.mockReset();
  mockedUploadWhenAwake.mockReset();
  global.fetch = jest.fn().mockResolvedValue(variantsResponse()) as unknown as typeof fetch;
});

it("generates an uploaded file from its browser transcription's job", async () => {
  mockedRun.mockResolvedValue(completed("job-browser"));
  const user = userEvent.setup();
  render(<GenerateVariantsPage />);

  await user.click(screen.getByRole("button", { name: /fake upload/i }));
  await user.click(screen.getByRole("button", { name: /generate variants/i }));

  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
  expect(mockedRun.mock.calls[0][0].file).toBe(SONG);
  // Already uploaded by UploadButton: reused, not uploaded again.
  await expect(mockedRun.mock.calls[0][0].ensureUploaded()).resolves.toEqual(UPLOAD);
  const body = sentForm();
  expect(body.get("job_id")).toBe("job-browser");
  expect(body.get("upload_id")).toBe(UPLOAD.id);
  expect(body.get("filename")).toBe(UPLOAD.filename);
  expect(body.get("file")).toBeNull();
});

it("doesn't transcribe the same file again on a second Generate", async () => {
  mockedRun.mockResolvedValue(completed("job-browser"));
  const user = userEvent.setup();
  render(<GenerateVariantsPage />);

  await user.click(screen.getByRole("button", { name: /fake upload/i }));
  await user.click(screen.getByRole("button", { name: /generate variants/i }));
  await screen.findByRole("heading", { name: /rendered variants/i });
  await user.click(screen.getByRole("button", { name: /generate variants/i }));

  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
  expect(mockedRun).toHaveBeenCalledTimes(1);
  expect(sentForm(1).get("job_id")).toBe("job-browser");
});

it("shows the browser transcription's progress while it runs", async () => {
  mockedRun.mockImplementation(({ onProgress }: { onProgress: (fraction: number) => void }) => {
    onProgress(0.42);
    return new Promise(() => {});
  });
  const user = userEvent.setup();
  render(<GenerateVariantsPage />);

  await user.click(screen.getByRole("button", { name: /fake upload/i }));
  await user.click(screen.getByRole("button", { name: /generate variants/i }));

  expect(
    await screen.findByRole("status", { name: /transcribing song\.mp3 in your browser.*40%/i }),
  ).toBeInTheDocument();
});

it("falls back to the uploaded file when the browser can't transcribe it", async () => {
  mockedRun.mockResolvedValue({ kind: "fallback", reason: "too_long", stored: null });
  const user = userEvent.setup();
  render(<GenerateVariantsPage />);

  await user.click(screen.getByRole("button", { name: /fake upload/i }));
  await user.click(screen.getByRole("button", { name: /generate variants/i }));

  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
  const body = sentForm();
  expect(body.get("job_id")).toBeNull();
  expect(body.get("upload_id")).toBe(UPLOAD.id);
  expect(body.get("file")).toBeNull();
});

it("falls back without an error when browser transcription throws", async () => {
  mockedRun.mockRejectedValue(new Error("register exploded"));
  const user = userEvent.setup();
  render(<GenerateVariantsPage />);

  await user.click(screen.getByRole("button", { name: /fake upload/i }));
  await user.click(screen.getByRole("button", { name: /generate variants/i }));

  await screen.findByRole("heading", { name: /rendered variants/i });
  expect(sentForm().get("upload_id")).toBe(UPLOAD.id);
  expect(screen.queryByText(/register exploded/i)).not.toBeInTheDocument();
});

it("uploads a recording once and generates it from its browser job", async () => {
  mockedUploadWhenAwake.mockResolvedValue({ id: "f".repeat(32), filename: "upload_rec.webm" });
  mockedRun.mockImplementation(
    async ({ ensureUploaded }: { ensureUploaded: () => Promise<typeof UPLOAD> }) =>
      completed("job-recording", await ensureUploaded()),
  );
  const user = userEvent.setup();
  render(<GenerateVariantsPage />);

  await user.click(screen.getByRole("button", { name: /record audio/i }));
  await user.click(screen.getByRole("button", { name: /fake recording/i }));
  await user.click(screen.getByRole("button", { name: /generate variants/i }));

  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
  expect(mockedRun.mock.calls[0][0].file).toBe(RECORDING);
  expect(mockedUploadWhenAwake).toHaveBeenCalledTimes(1);
  const body = sentForm();
  expect(body.get("job_id")).toBe("job-recording");
  expect(body.get("upload_id")).toBe("f".repeat(32));
  expect(body.get("file")).toBeNull();
});

it("still sends a recording as a file when the browser can't transcribe it", async () => {
  mockedRun.mockResolvedValue({ kind: "fallback", reason: "unsupported", stored: null });
  const user = userEvent.setup();
  render(<GenerateVariantsPage />);

  await user.click(screen.getByRole("button", { name: /record audio/i }));
  await user.click(screen.getByRole("button", { name: /fake recording/i }));
  await user.click(screen.getByRole("button", { name: /generate variants/i }));

  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
  const body = sentForm();
  expect(body.get("file")).toBe(RECORDING);
  expect(body.get("job_id")).toBeNull();
  expect(mockedUploadWhenAwake).not.toHaveBeenCalled();
});

it("stops the browser transcription when leaving the page", async () => {
  let seenSignal: AbortSignal | null = null;
  mockedRun.mockImplementation(({ signal }: { signal: AbortSignal }) => {
    seenSignal = signal;
    return new Promise(() => {});
  });
  const user = userEvent.setup();
  const { unmount } = render(<GenerateVariantsPage />);

  await user.click(screen.getByRole("button", { name: /fake upload/i }));
  await user.click(screen.getByRole("button", { name: /generate variants/i }));
  await waitFor(() => expect(seenSignal).not.toBeNull());
  unmount();

  expect(seenSignal!.aborted).toBe(true);
  expect(global.fetch).not.toHaveBeenCalled();
});
