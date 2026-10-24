import type { TfjsBackend } from "./transcribeJob";

export type TranscribePath = "browser" | "server";

// One console line per finished transcription, so a real run (local or
// prod) gives pick -> "Transcription ready" time without DevTools
// profiling, e.g. "[transcribe] path=browser backend=webgl notes=812
// elapsed=24.3s". backend is the TF.js backend that ran in the browser;
// "none" on the server path, which doesn't report its runtime.
export function formatTranscribeLog({
  path,
  backend,
  notes,
  elapsedMs,
}: {
  path: TranscribePath;
  backend: TfjsBackend | null;
  notes: number;
  elapsedMs: number;
}): string {
  const elapsedSec = (elapsedMs / 1000).toFixed(1);
  return `[transcribe] path=${path} backend=${backend ?? "none"} notes=${notes} elapsed=${elapsedSec}s`;
}
