import * as z from "zod/mini";

import { moodLabelSchema } from "./analysis";

// One generated variant (backend/app/inference.py _variant_entry). MIDI only:
// the browser synthesises playback, so the wav_* keys are kept but empty.
export const variantSchema = z.object({
  index: z.number(),
  temperature: z.number(),
  midi_b64: z.string(),
  midi_filename: z.string(),
  midi_download_path: z.string(),
  wav_b64: z.nullable(z.string()),
  wav_filename: z.string(),
  wav_download_path: z.string(),
});

const checkpointStatusSchema = z.object({
  path: z.string(),
  exists: z.boolean(),
  size_mb: z.number(),
  loaded: z.boolean(),
});

// backend/app/inference.py _variant_model_status.
export const modelStatusSchema = z.object({
  cvae: checkpointStatusSchema,
  iddm_ppo: checkpointStatusSchema,
  device: z.string(),
  load_error: z.nullable(z.string()),
  fluidsynth_available: z.boolean(),
});

// generate_iddm_variants' return value, which is also what a variants job
// stores as its result (jobs/service.create_completed_job).
export const variantsJobResultSchema = z.object({
  n_variants: z.number(),
  temperatures: z.array(z.number()),
  mood_idx: z.number(),
  mood_label: moodLabelSchema,
  model_status: modelStatusSchema,
  variants: z.array(variantSchema),
});

// POST /api/generate-variants: the result plus the job id it was stored
// under, for the shareable /result/{job_id} link.
export const variantsResponseSchema = z.extend(variantsJobResultSchema, {
  job_id: z.string().check(z.minLength(1)),
});

export type Variant = z.infer<typeof variantSchema>;
export type VariantsJobResult = z.infer<typeof variantsJobResultSchema>;
export type VariantsResponse = z.infer<typeof variantsResponseSchema>;
