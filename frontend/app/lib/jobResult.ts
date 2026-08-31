import type { TranscriptionResult } from "./transcribeJob";

export type VariantResult = {
  index: number;
  temperature: number;
  midi_b64: string;
  midi_filename: string;
  midi_download_path: string;
  wav_b64: string | null;
  wav_filename: string;
  wav_download_path: string;
};

export type VariantsJobResult = {
  n_variants: number;
  temperatures: number[];
  mood_idx: number;
  mood_label: "happy" | "sad" | "neutral";
  model_status: {
    cvae: { path: string; exists: boolean; size_mb: number; loaded: boolean };
    iddm_ppo: { path: string; exists: boolean; size_mb: number; loaded: boolean };
    device: string;
    load_error: string | null;
    fluidsynth_available: boolean;
  };
  variants: VariantResult[];
};

export type ProgressionJobResult = {
  progression: string[];
  bpm: number;
  instrument: number;
  midi_b64: string;
  midi_filename: string;
  midi_download_path: string;
  wav_b64: string;
  wav_filename: string;
  wav_download_path: string;
  detected_chords: string[];
};

// The three shapes a completed job's `result` can take -- one per
// synchronous generation surface that now creates a job purely to get a
// shareable, job-id-addressed download link (see jobs/service.py's
// create_completed_job, GP3). The store itself doesn't tag which kind a job
// is, so callers discriminate structurally via the helpers below.
export type JobResult = TranscriptionResult | VariantsJobResult | ProgressionJobResult;

export function isVariantsJobResult(result: JobResult): result is VariantsJobResult {
  return "variants" in result;
}

export function isProgressionJobResult(result: JobResult): result is ProgressionJobResult {
  return "progression" in result;
}
