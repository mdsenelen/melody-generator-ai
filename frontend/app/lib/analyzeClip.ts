import {
  clipAnalysisSchema,
  type AnalyzeClipRequest,
  type ClipAnalysis,
} from "../../src/domain/contracts/analysis";
import { getPublicBackendApiUrl } from "./backendUrl";
import { requestJson } from "./request";

// POST /api/analyze -- mood / key / tempo / chords / pitch histogram for a
// clip window of a completed transcribe job, computed from its stored note
// events (no audio, no Basic Pitch). Re-runnable against any window.
export async function analyzeClip(
  request: AnalyzeClipRequest,
  init: { signal: AbortSignal },
): Promise<ClipAnalysis> {
  return requestJson(getPublicBackendApiUrl("/analyze"), clipAnalysisSchema, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
    expectedContentType: "application/json",
    signal: init.signal,
  });
}
