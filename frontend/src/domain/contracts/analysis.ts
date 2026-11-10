import * as z from "zod/mini";

// backend/app/model/tokens.py MOOD_LABELS. Strict: MoodBadge styles exactly these.
export const moodLabelSchema = z.enum(["happy", "sad", "neutral"]);
export type MoodLabel = z.infer<typeof moodLabelSchema>;

// POST /api/analyze body (backend/app/schemas.py AnalyzeRequest). Omitted
// bounds mean "from the start" / "to the end".
export type AnalyzeClipRequest = {
  job_id: string;
  clip_start_sec?: number;
  clip_end_sec?: number;
};

// POST /api/analyze response (backend/app/inference.py analyze_clip): mood /
// key / tempo / chords / pitch histogram for one clip window of a completed
// transcription, computed from its stored note events.
export const clipAnalysisSchema = z.object({
  tempo_bpm: z.number(),
  average_pitch: z.number(),
  mood_idx: z.number(),
  mood_label: moodLabelSchema,
  key: z.string(),
  pitch_histogram: z.array(z.number()),
  detected_chords: z.array(z.string()),
  n_notes: z.number(),
  clip_start_sec: z.number(),
  clip_end_sec: z.number(),
});

export type ClipAnalysis = z.infer<typeof clipAnalysisSchema>;
