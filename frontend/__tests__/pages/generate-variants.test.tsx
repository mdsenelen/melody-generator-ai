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
});
