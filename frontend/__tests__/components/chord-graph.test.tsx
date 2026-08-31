import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ChordGraph } from "../../components/chord-graph";

function mockGenerateProgressionFetch() {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    status: 200,
    headers: new Headers({ "content-type": "application/json" }),
    text: async () =>
      JSON.stringify({
        audio_b64: "AAA=",
        midi_b64: "AAA=",
        midi_filename: "progression.mid",
        midi_download_path: "/api/download/progression.mid",
        wav_filename: "progression.wav",
        wav_download_path: "/api/download/progression.wav",
        bpm: 120,
        instrument: 0,
        job_id: "job-progression-1",
      }),
  }) as unknown as typeof fetch;
}

describe("ChordGraph", () => {
  beforeEach(() => {
    global.URL.createObjectURL = jest.fn().mockReturnValue("blob:fake-url");
    global.URL.revokeObjectURL = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("links a rendered progression to its job-id result page instead of an inline download button", async () => {
    mockGenerateProgressionFetch();
    const user = userEvent.setup();

    render(<ChordGraph title="Custom progression" progression={["C", "G", "Am", "F"]} />);

    await user.click(screen.getByRole("button", { name: /play progression/i }));

    const resultLink = await screen.findByRole("link", { name: /view & download result/i });
    expect(resultLink).toHaveAttribute("href", "/result/job-progression-1");
    expect(screen.queryByRole("link", { name: /download midi/i })).not.toBeInTheDocument();
  });
});
