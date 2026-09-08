import { getPublicBackendApiUrl } from "./backendUrl";
import { requestJson } from "./request";

// POST /api/analyze -- mood / key / tempo / chords / pitch histogram for a
// clip window of a completed transcribe job, computed from its stored note
// events (no audio, no Basic Pitch). Re-runnable against any window.

export type AnalyzeClipRequest = {
  job_id: string;
  clip_start_sec?: number;
  clip_end_sec?: number;
};

export type ClipAnalysis = {
  tempo_bpm: number;
  average_pitch: number;
  mood_idx: number;
  mood_label: "happy" | "sad" | "neutral";
  key: string;
  pitch_histogram: number[];
  detected_chords: string[];
  n_notes: number;
  clip_start_sec: number;
  clip_end_sec: number;
};

export async function analyzeClip(
  request: AnalyzeClipRequest,
  init?: { signal?: AbortSignal },
): Promise<ClipAnalysis> {
  return requestJson<ClipAnalysis>(getPublicBackendApiUrl("/analyze"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
    expectedContentType: "application/json",
    signal: init?.signal,
  });
}
