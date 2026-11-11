import * as z from "zod/mini";

// One entry of GET /api/progressions (backend/app/data/progressions.json,
// or inference.py's _PRESET_PROGRESSIONS_FALLBACK).
export const progressionSchema = z.object({
  id: z.string(),
  name: z.string(),
  genre: z.string(),
  source: z.string(),
  chords: z.array(z.string()),
  song_title: z.nullable(z.string()),
  artist: z.nullable(z.string()),
  year: z.nullable(z.number()),
});

export const progressionsResponseSchema = z.object({
  progressions: z.array(progressionSchema),
});

// POST /api/generate-progression body (backend/app/schemas.py
// GenerateProgressionRequest).
export type RenderProgressionRequest = {
  progression: string[];
  bpm: number;
  instrument: number;
};

// inference.py _generate_progression_payload, as a progression job stores it
// (jobs/service.create_completed_job) and the result page reads it.
export const progressionJobResultSchema = z.object({
  progression: z.array(z.string()),
  bpm: z.number(),
  instrument: z.number(),
  midi_b64: z.string(),
  midi_filename: z.string(),
  midi_download_path: z.string(),
  wav_b64: z.string(),
  wav_filename: z.string(),
  wav_download_path: z.string(),
  detected_chords: z.array(z.string()),
});

// POST /api/generate-progression response: the same payload plus job_id.
// Only what ChordGraph reads is required.
export const renderProgressionResponseSchema = z.object({
  audio_b64: z.string(),
  midi_b64: z.string(),
  midi_filename: z.string(),
  midi_download_path: z.string(),
  wav_filename: z.string(),
  wav_download_path: z.string(),
  bpm: z.number(),
  instrument: z.number(),
  job_id: z.string().check(z.minLength(1)),
});

export type Progression = z.infer<typeof progressionSchema>;
export type ProgressionsResponse = z.infer<typeof progressionsResponseSchema>;
export type ProgressionJobResult = z.infer<typeof progressionJobResultSchema>;
export type RenderProgressionResponse = z.infer<typeof renderProgressionResponseSchema>;
