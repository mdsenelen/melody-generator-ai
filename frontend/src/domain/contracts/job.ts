// Contracts use zod/mini (zod 4's functional API) rather than the classic
// `import { z } from "zod"`: the classic namespace pulls every locale and
// toJSONSchema into the client bundle (measured: ~+100 KB gzip per page vs
// ~+20 KB with mini).
import * as z from "zod/mini";

import { progressionJobResultSchema } from "./progressions";
import { variantsJobResultSchema } from "./variants";

// The async job API (backend/app/jobs/routes.py, jobs/models.py).

export const noteEventSchema = z.object({
  start: z.number(),
  end: z.number(),
  pitch: z.number(),
  velocity: z.number(),
});

export const tfjsBackendSchema = z.enum(["webgl", "wasm", "cpu"]);

// inference.py _build_transcription_result: MIDI + note events only. Mood /
// key / tempo / chords / histogram come from POST /api/analyze instead.
export const transcriptionResultSchema = z.object({
  n_notes: z.number(),
  duration_sec: z.number(),
  source_duration_sec: z.number(),
  truncated: z.boolean(),
  midi_b64: z.string(),
  wav_b64: z.nullable(z.string()),
  midi_filename: z.string(),
  wav_filename: z.string(),
  // Only set by the chunked server path.
  n_chunks: z.optional(z.number()),
  note_events: z.array(noteEventSchema),
  // Which path produced the result. Optional: jobs persisted before the
  // browser path existed have neither field.
  transcribed_by: z.optional(z.enum(["browser", "server"])),
  tfjs_backend: z.optional(tfjsBackendSchema),
});

// jobs/models.py: the internal 'creating' state is reported as 'queued';
// 'expired' is synthesised by GET /transcribe/{id} past DATA_RETENTION_HOURS.
export const jobStatusSchema = z.enum(["queued", "processing", "completed", "failed", "expired"]);

function jobStatusResponseSchema<R extends z.ZodMiniType>(result: R) {
  return z.object({
    job_id: z.string().check(z.minLength(1)),
    status: jobStatusSchema,
    progress: z.number(),
    result: z.nullable(result),
    error: z.nullable(z.string()),
    // Allowlisted on the backend (_ALLOWED_WAITING_REASONS). Optional, not
    // just nullable: the 'expired' payload and jobs stored before the field
    // existed don't send it.
    waiting_reason: z.optional(z.nullable(z.enum(["warming_up"]))),
  });
}

// GET /api/transcribe/{id} and POST /api/transcriptions, for a transcription job.
export const transcribeJobStatusSchema = jobStatusResponseSchema(transcriptionResultSchema);

// POST /api/transcribe: 202 with the new job's id and status only -- no
// progress / result / error (jobs/routes.py create_transcribe_job).
export const createTranscribeJobResponseSchema = z.object({
  job_id: z.string().check(z.minLength(1)),
  status: jobStatusSchema,
});

// GET /api/transcribe/{id} serves every job kind: generate-variants and
// generate-progression also store their result as a completed job (GP3), and
// the store doesn't tag which kind a job is. Parsing tags it, so the result
// page switches on `kind` instead of probing for keys.
export const jobResultSchema = z.union([
  z.pipe(
    variantsJobResultSchema,
    z.transform((data) => ({ kind: "variants" as const, data })),
  ),
  z.pipe(
    progressionJobResultSchema,
    z.transform((data) => ({ kind: "progression" as const, data })),
  ),
  z.pipe(
    transcriptionResultSchema,
    z.transform((data) => ({ kind: "transcription" as const, data })),
  ),
]);

export const jobResultStatusSchema = jobStatusResponseSchema(jobResultSchema);

// POST /api/transcriptions body (backend/app/schemas.py
// RegisterTranscriptionRequest). Sent, not parsed: a type is enough.
export type RegisterTranscriptionRequest = {
  upload_id: string | null;
  source_filename: string;
  source_duration_sec: number;
  tfjs_backend: TfjsBackend;
  note_events: NoteEvent[];
};

export type NoteEvent = z.infer<typeof noteEventSchema>;
export type TfjsBackend = z.infer<typeof tfjsBackendSchema>;
export type TranscriptionResult = z.infer<typeof transcriptionResultSchema>;
export type JobStatus = z.infer<typeof jobStatusSchema>;
export type TranscribeJobStatusResponse = z.infer<typeof transcribeJobStatusSchema>;
export type CreateTranscribeJobResponse = z.infer<typeof createTranscribeJobResponseSchema>;
export type JobResult = z.infer<typeof jobResultSchema>;
export type JobResultStatusResponse = z.infer<typeof jobResultStatusSchema>;
