// Response bodies for POST /api/generate-variants, shaped like the FastAPI
// route's (backend/app/inference.py generate_variants).
import { Midi } from "@tonejs/midi";

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

/** A real, playable three-note MIDI file. */
export function midiBase64(): string {
  const midi = new Midi();
  const track = midi.addTrack();
  track.addNote({ midi: 60, time: 0, duration: 0.5, velocity: 0.8 });
  track.addNote({ midi: 64, time: 0.5, duration: 0.5, velocity: 0.7 });
  track.addNote({ midi: 67, time: 1.0, duration: 1.0, velocity: 0.9 });
  return bytesToBase64(midi.toArray());
}

export function variantsResponse(jobId = "job-variants-1") {
  return {
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
        midi_b64: midiBase64(),
        midi_filename: "variant_1.mid",
        midi_download_path: "/api/download/variant_1.mid",
        wav_b64: null,
        wav_filename: "",
        wav_download_path: "",
      },
    ],
    job_id: jobId,
  };
}
