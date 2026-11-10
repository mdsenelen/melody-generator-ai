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
 * The /generate-variants button label while busy. Only browser progress has
 * its own copy; everything else (including the moment before the first
 * progress tick) reads "Generating variants", as it did before ADR 0001.
 */
export function describeVariantsProgress<A>(state: PipelineState<A>): string {
  if (state.status === "transcribing" && state.step.kind === "browser") {
    return `Transcribing ${sourceName(state.source)} in your browser... ${state.step.step * 10}%`;
  }
  return "Generating variants";
}
