import type { TfjsBackend } from "../transcribeJob";

// Messages between transcribeInBrowser (main thread) and
// transcribe.worker.ts. Type-only on purpose: importing this file must
// never pull TF.js into the main bundle.

// Basic Pitch's NoteEventTime, minus pitch bends (the server drops them too).
export type BasicPitchNote = {
  startTimeSeconds: number;
  durationSeconds: number;
  pitchMidi: number;
  amplitude: number;
};

export type WorkerRequest = {
  type: "transcribe";
  // 22050 Hz mono -- Basic Pitch's required input rate.
  audio: Float32Array;
  modelUrl: string;
  wasmPath: string;
};

export type WorkerFailureStage = "backend" | "model" | "inference";

export type WorkerResponse =
  | { type: "progress"; fraction: number }
  | { type: "done"; notes: BasicPitchNote[]; backend: TfjsBackend }
  | { type: "error"; stage: WorkerFailureStage; message: string };
