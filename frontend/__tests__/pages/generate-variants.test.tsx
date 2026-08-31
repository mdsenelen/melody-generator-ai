import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import GenerateVariantsPage from "../../app/generate-variants/page";
import { useSessionStore } from "../../app/lib/session-store";

describe("GenerateVariantsPage", () => {
  beforeEach(() => {
    useSessionStore.setState({ lastUpload: null });
  });

  it("disables Generate until an audio source is selected", () => {
    render(<GenerateVariantsPage />);
    expect(screen.getByRole("button", { name: /generate variants/i })).toBeDisabled();
  });

  it("does not show the 'use my last upload' banner when no session exists", () => {
    render(<GenerateVariantsPage />);
    expect(screen.queryByText(/use my last upload/i)).not.toBeInTheDocument();
  });

  it("offers and applies a prior upload from the session store", async () => {
    const user = userEvent.setup();
    useSessionStore.getState().setLastUpload({
      uploadId: "abc123",
      filename: "upload_abc123.wav",
      sourceName: "my-riff.wav",
      transcription: {
        chords: ["C", "G"],
        key: "C major",
        moodLabel: "happy",
        pitchHistogram: [],
      },
    });

    render(<GenerateVariantsPage />);

    expect(screen.getByText("my-riff.wav")).toBeInTheDocument();
    const useButton = screen.getByRole("button", { name: /use my last upload/i });

    await user.click(useButton);

    expect(screen.getByRole("button", { name: /using this upload/i })).toBeDisabled();
    // Selecting the stored upload should be enough to enable generation
    // without requiring a fresh file upload.
    expect(screen.getByRole("button", { name: /generate variants/i })).toBeEnabled();
  });

  it("lets the user pick a variant count between 1 and 8", () => {
    render(<GenerateVariantsPage />);
    const slider = screen.getByRole("slider");
    expect(slider).toHaveAttribute("min", "1");
    expect(slider).toHaveAttribute("max", "8");
  });

  it("links the generated result to its job-id result page instead of an inline download button", async () => {
    global.fetch = jest.fn().mockResolvedValue({
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
          variants: [
            {
              index: 0,
              temperature: 0.7,
              midi_b64: "AAA=",
              midi_filename: "variant_1.mid",
              midi_download_path: "/api/download/variant_1.mid",
              wav_b64: "AAA=",
              wav_filename: "variant_1.wav",
              wav_download_path: "/api/download/variant_1.wav",
            },
          ],
          job_id: "job-variants-1",
        }),
    }) as unknown as typeof fetch;

    useSessionStore.getState().setLastUpload({
      uploadId: "abc123",
      filename: "upload_abc123.wav",
      sourceName: "my-riff.wav",
      transcription: { chords: [], key: "C major", moodLabel: "happy", pitchHistogram: [] },
    });

    const user = userEvent.setup();
    render(<GenerateVariantsPage />);

    await user.click(screen.getByRole("button", { name: /use my last upload/i }));
    await user.click(screen.getByRole("button", { name: /generate variants/i }));

    const resultLink = await screen.findByRole("link", { name: /view & download result/i });
    expect(resultLink).toHaveAttribute("href", "/result/job-variants-1");
    expect(screen.queryByRole("button", { name: /download midi/i })).not.toBeInTheDocument();
  });
});
