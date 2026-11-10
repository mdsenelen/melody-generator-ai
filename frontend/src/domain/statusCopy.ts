// Every pipeline status line, derived from state. Pure, so the copy is
// unit-tested without timers, workers, or fetch.

import { sourceName, type PipelineState, type Transcribing } from "./pipeline";

export const WAKING_STATUS_MESSAGE =
  "Waking the server — this can take a minute or two on the free tier…";

function assertNever(value: never): never {
  throw new Error(`Unhandled transcribe step: ${JSON.stringify(value)}`);
}

/** The /analyse spinner label while a clip is transcribing. */
export function describeTranscription(state: Transcribing): string {
  const name = sourceName(state.source);
  const step = state.step;
  switch (step.kind) {
    case "starting":
      return `Transcribing ${name}...`;
    case "uploading":
      return step.phase === "waking" ? WAKING_STATUS_MESSAGE : `Uploading ${name}...`;
    case "stillWorking":
      return `Still transcribing ${name}... this can take a minute or two if the server was idle.`;
    case "browser":
      return step.step === 10
        ? `Transcribed ${name} in your browser. Saving the notes...`
        : `Transcribing ${name} in your browser... ${step.step * 10}%`;
    case "savingRetry":
      return `Server was still waking up — saving the notes for ${name}...`;
    case "submitRetry":
      return `Server was still waking up — retrying ${name}...`;
    case "queued":
      return step.slow
        ? `Still queued for ${name}... this can take a minute or two if the server was idle.`
        : `Queued for transcription...`;
    case "processing":
      return step.slow
        ? `Still transcribing ${name}... longer clips take longer, and a cold server can add a minute or two on top.`
        : `Transcribing ${name}...`;
    case "warmingUp":
      // The worker bounces a job queued during warm-up without failing it
      // (backend/app/jobs/worker.py): calm, specific copy, never an error.
      return (
        "The server is starting up. Your clip will be transcribed automatically, " +
        "this can take about 2 to 4 minutes."
      );
    case "restarting":
      // The job survives the restart in Postgres; polling continues.
      return "The server restarted and is coming back up. Your clip is safe -- we'll keep checking automatically.";
    default:
      return assertNever(step);
  }
}

/**
 * The /generate-variants button label while busy. Variants only transcribes
 * in the browser, so every transcribing step reads that way, with a
 * percentage once Basic Pitch reports one.
 */
export function describeVariantsProgress<A>(state: PipelineState<A>): string {
  if (state.status !== "transcribing") {
    return "Generating variants";
  }
  const name = sourceName(state.source);
  return state.step.kind === "browser"
    ? `Transcribing ${name} in your browser... ${state.step.step * 10}%`
    : `Transcribing ${name} in your browser...`;
}

/** The /analyse clip-analysis spinner while POST /api/analyze retries a cold server. */
export const ANALYSIS_RETRY_MESSAGE = "Server was still waking up — retrying analysis...";
