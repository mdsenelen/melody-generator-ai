import {
  progressionsResponseSchema,
  renderProgressionResponseSchema,
  type Progression,
  type RenderProgressionRequest,
  type RenderProgressionResponse,
} from "../../src/domain/contracts/progressions";
import { requestJson } from "./request";

// Both go through the same-origin Next.js proxy routes (app/api/*).

/** GET /api/progressions: the curated chord-progression library. */
export async function fetchProgressions(options: { signal: AbortSignal }): Promise<Progression[]> {
  const data = await requestJson("/api/progressions", progressionsResponseSchema, {
    expectedContentType: "application/json",
    signal: options.signal,
  });
  return data.progressions;
}

/** POST /api/generate-progression: renders a progression to audio + MIDI. */
export async function renderProgression(
  request: RenderProgressionRequest,
  options: { signal: AbortSignal },
): Promise<RenderProgressionResponse> {
  const data = await requestJson("/api/generate-progression", renderProgressionResponseSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    expectedContentType: "application/json",
    signal: options.signal,
  });
  if (!data.audio_b64) {
    throw new Error("Could not render progression");
  }
  return data;
}
