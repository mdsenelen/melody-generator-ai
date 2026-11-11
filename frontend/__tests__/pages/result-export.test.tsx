import { screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";

import ResultView from "../../app/result/[jobId]/result-view";
import { midiBase64, variantsResponse } from "../../src/test/fixtures/variants";
import { BACKEND_API, server } from "../../src/test/msw/server";
import { renderWithQueryClient } from "../../src/test/query/render";

function completedJob(result: unknown) {
  server.use(
    http.get(`${BACKEND_API}/transcribe/job-1`, () =>
      HttpResponse.json({ job_id: "job-1", status: "completed", progress: 1, result, error: null }),
    ),
  );
}

// The browser names a same-origin download after the link's `download`
// attribute, so that is what decides the saved file's name.
describe("Result page exports", () => {
  it("saves a progression's MIDI and WAV under their own filenames", async () => {
    completedJob({
      progression: ["C", "G"],
      bpm: 120,
      instrument: 0,
      midi_b64: midiBase64(),
      midi_filename: "progression_c_g.mid",
      midi_download_path: "/api/download/progression_c_g.mid",
      wav_b64: "AAA=",
      wav_filename: "progression_c_g.wav",
      wav_download_path: "/api/download/progression_c_g.wav",
      detected_chords: ["C", "G"],
    });

    renderWithQueryClient(<ResultView jobId="job-1" />);

    const midi = await screen.findByRole("link", { name: /download midi/i });
    expect(midi).toHaveAttribute("download", "progression_c_g.mid");
    expect(midi).toHaveAttribute("href", "/api/download/progression_c_g.mid");
    const wav = screen.getByRole("link", { name: /download wav/i });
    expect(wav).toHaveAttribute("download", "progression_c_g.wav");
    expect(wav).toHaveAttribute("href", "/api/download/progression_c_g.wav");
  });

  it("saves a generated variant's MIDI under the variant's filename", async () => {
    // A variants job stores generate_iddm_variants' full return value.
    completedJob(variantsResponse());

    renderWithQueryClient(<ResultView jobId="job-1" />);

    const midi = await screen.findByRole("link", { name: /download midi/i });
    expect(midi).toHaveAttribute("download", "variant_1.mid");
    expect(midi).toHaveAttribute("href", "/api/download/variant_1.mid");
  });

  it("builds a download path that survives # and spaces in the filename", async () => {
    // A transcription job's result, as _build_transcription_result stores it.
    completedJob({
      n_notes: 0,
      duration_sec: 1,
      source_duration_sec: 1,
      truncated: false,
      midi_b64: midiBase64(),
      wav_b64: null,
      midi_filename: "take #2.mid",
      wav_filename: "",
      note_events: [],
    });

    renderWithQueryClient(<ResultView jobId="job-1" />);

    const midi = await screen.findByRole("link", { name: /download midi/i });
    expect(midi).toHaveAttribute("download", "take #2.mid");
    expect(midi).toHaveAttribute("href", "/api/download/take%20%232.mid");
  });
});
