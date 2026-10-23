/// <reference lib="webworker" />
// The ONLY module that imports TF.js / Basic Pitch. Loaded via
// createWorker.ts's `new URL(..., import.meta.url)`, so all of it lands in
// the worker chunk and never in a page bundle.
import { BasicPitch, noteFramesToTime, outputToNotesPoly } from "@spotify/basic-pitch";
import * as tf from "@tensorflow/tfjs";
// Pinned to the exact @tensorflow/tfjs version @spotify/basic-pitch resolves
// (package.json) -- a core/backend version mismatch fails at runtime.
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import type { TfjsBackend } from "../transcribeJob";
import type { BasicPitchNote, WorkerFailureStage, WorkerRequest, WorkerResponse } from "./protocol";

// Python basic-pitch==0.4.0's predict() defaults, so browser and server
// notes stay close: onset_threshold 0.5, frame_threshold 0.3,
// minimum_note_length 127.70 ms = round(0.1277 s x 22050/256 fps) = 11 frames
// (the TS default is 5 frames, which would keep many more short notes).
const ONSET_THRESHOLD = 0.5;
const FRAME_THRESHOLD = 0.3;
const MIN_NOTE_LENGTH_FRAMES = 11;

const scope = self as unknown as DedicatedWorkerGlobalScope;

function post(message: WorkerResponse) {
  scope.postMessage(message);
}

class StageError extends Error {
  constructor(
    readonly stage: WorkerFailureStage,
    cause: unknown,
  ) {
    super(cause instanceof Error ? cause.message : String(cause));
  }
}

// Explicit order: WebGL needs OffscreenCanvas inside a worker; wasm is the
// next-fastest everywhere else; cpu is the always-available floor.
// tf.setBackend resolves false (rather than throwing) when a backend fails
// to initialise, so both outcomes move on to the next candidate.
async function selectBackend(wasmPath: string): Promise<TfjsBackend> {
  setWasmPaths(wasmPath);
  const candidates: TfjsBackend[] =
    typeof OffscreenCanvas !== "undefined" ? ["webgl", "wasm", "cpu"] : ["wasm", "cpu"];
  for (const name of candidates) {
    try {
      if (await tf.setBackend(name)) {
        await tf.ready();
        return name;
      }
    } catch {
      // try the next backend
    }
  }
  throw new StageError("backend", "No TF.js backend could be initialised");
}

async function transcribe({ audio, modelUrl, wasmPath }: WorkerRequest) {
  const backend = await selectBackend(wasmPath);
  console.info(`[basic-pitch worker] tfjs backend: ${backend}`);

  let model: tf.GraphModel;
  try {
    model = await tf.loadGraphModel(modelUrl);
  } catch (error) {
    throw new StageError("model", error);
  }

  const frames: number[][] = [];
  const onsets: number[][] = [];
  try {
    await new BasicPitch(Promise.resolve(model)).evaluateModel(
      audio,
      (f, o) => {
        frames.push(...f);
        onsets.push(...o);
      },
      (fraction) => post({ type: "progress", fraction }),
    );
  } catch (error) {
    throw new StageError("inference", error);
  } finally {
    model.dispose();
  }

  // Pitch bends are skipped: the server drops them from note events too.
  const notes: BasicPitchNote[] = noteFramesToTime(
    outputToNotesPoly(frames, onsets, ONSET_THRESHOLD, FRAME_THRESHOLD, MIN_NOTE_LENGTH_FRAMES),
  ).map(({ startTimeSeconds, durationSeconds, pitchMidi, amplitude }) => ({
    startTimeSeconds,
    durationSeconds,
    pitchMidi,
    amplitude,
  }));
  post({ type: "done", notes, backend });
}

scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
  transcribe(event.data).catch((error: unknown) => {
    post({
      type: "error",
      stage: error instanceof StageError ? error.stage : "inference",
      message: error instanceof Error ? error.message : String(error),
    });
  });
};
