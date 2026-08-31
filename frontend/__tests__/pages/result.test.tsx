import { render, screen } from "@testing-library/react";

import ResultView from "../../app/result/[jobId]/result-view";
import { getTranscribeJob } from "../../app/lib/transcribeJob";

jest.mock("../../app/lib/transcribeJob", () => {
  const actual = jest.requireActual("../../app/lib/transcribeJob");
  return { ...actual, getTranscribeJob: jest.fn() };
});

const mockedGetJob = getTranscribeJob as jest.Mock;

function statusResponse(overrides: Record<string, unknown>) {
  return {
    job_id: "job-1",
    status: "queued",
    progress: 0,
    result: null,
    error: null,
    ...overrides,
  };
}

describe("ResultView", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("shows a running state while the job is still queued or processing", async () => {
    mockedGetJob.mockResolvedValue(statusResponse({ status: "processing" }));

    render(<ResultView jobId="job-1" />);

    expect(await screen.findByText(/still working on your result/i)).toBeInTheDocument();
  });

  it("renders a transcription result with real download links, not inline buttons", async () => {
    mockedGetJob.mockResolvedValue(
      statusResponse({
        status: "completed",
        result: {
          n_notes: 3,
          duration_sec: 1,
          source_duration_sec: 1,
          truncated: false,
          midi_b64: "AAA=",
          wav_b64: "AAA=",
          midi_filename: "transcription.mid",
          wav_filename: "transcription.wav",
          mood_label: "happy",
          mood_idx: 0,
          detected_chords: ["C"],
          key: "C major",
          pitch_histogram: [],
          tempo_bpm: 120,
          average_pitch: 60,
        },
      }),
    );

    render(<ResultView jobId="job-1" />);

    expect(
      await screen.findByRole("heading", { name: /transcription complete/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /download midi/i })).toHaveAttribute(
      "href",
      "/api/download/transcription.mid",
    );
    expect(screen.getByRole("link", { name: /download wav/i })).toHaveAttribute(
      "href",
      "/api/download/transcription.wav",
    );
  });

  it("renders a variants result with a variant picker and per-variant download links", async () => {
    mockedGetJob.mockResolvedValue(
      statusResponse({
        status: "completed",
        result: {
          n_variants: 2,
          temperatures: [0.7, 0.9],
          mood_idx: 0,
          mood_label: "happy",
          model_status: {
            cvae: { path: "cvae", exists: true, size_mb: 1, loaded: true },
            iddm_ppo: { path: "iddm", exists: true, size_mb: 1, loaded: true },
            device: "cpu",
            load_error: null,
            fluidsynth_available: true,
          },
          variants: [
            {
              index: 0,
              temperature: 0.7,
              midi_b64: "AAA=",
              midi_filename: "variant_1.mid",
              midi_download_path: "/api/download/variant_1.mid",
              wav_b64: null,
              wav_filename: "",
              wav_download_path: "",
            },
            {
              index: 1,
              temperature: 0.9,
              midi_b64: "AAA=",
              midi_filename: "variant_2.mid",
              midi_download_path: "/api/download/variant_2.mid",
              wav_b64: null,
              wav_filename: "",
              wav_download_path: "",
            },
          ],
        },
      }),
    );

    render(<ResultView jobId="job-1" />);

    expect(
      await screen.findByRole("heading", { name: /generated melody variants/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Variant 1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Variant 2" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /download midi/i })).toHaveAttribute(
      "href",
      "/api/download/variant_1.mid",
    );
  });

  it("renders a progression result with download links", async () => {
    mockedGetJob.mockResolvedValue(
      statusResponse({
        status: "completed",
        result: {
          progression: ["C", "G"],
          bpm: 120,
          instrument: 0,
          midi_b64: "AAA=",
          midi_filename: "progression.mid",
          midi_download_path: "/api/download/progression.mid",
          wav_b64: "AAA=",
          wav_filename: "progression.wav",
          wav_download_path: "/api/download/progression.wav",
          detected_chords: ["C", "G"],
        },
      }),
    );

    render(<ResultView jobId="job-1" />);

    expect(
      await screen.findByRole("heading", { name: /rendered chord progression/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /download wav/i })).toHaveAttribute(
      "href",
      "/api/download/progression.wav",
    );
  });

  it("shows the failure message when the job failed", async () => {
    mockedGetJob.mockResolvedValue(
      statusResponse({ status: "failed", error: "Could not decode audio" }),
    );

    render(<ResultView jobId="job-1" />);

    expect(await screen.findByText("Could not decode audio")).toBeInTheDocument();
  });

  it("reads an expired job as gone, not as an error", async () => {
    mockedGetJob.mockResolvedValue(statusResponse({ status: "expired" }));

    render(<ResultView jobId="job-1" />);

    expect(await screen.findByText(/this result isn't available/i)).toBeInTheDocument();
  });

  it("reads a fabricated job id (404) as gone, not as a crash or blank page", async () => {
    const notFound = new Error("Job not found") as Error & { status?: number };
    notFound.status = 404;
    mockedGetJob.mockRejectedValue(notFound);

    render(<ResultView jobId="does-not-exist" />);

    expect(await screen.findByText(/this result isn't available/i)).toBeInTheDocument();
  });
});
