// The audio → melody pipeline as one typed state machine (ADR 0001).
// Framework-free: no React import, no I/O. Pages dispatch events; this
// module decides what state they lead to.

import type { DomainError } from "./errors";

export type StoredUpload = { id: string; filename: string };

export type AudioSource =
  | { kind: "upload"; file: File; stored: StoredUpload }
  | { kind: "recording"; file: File }
  // The last upload kept in the session store (already on the backend).
  | { kind: "session"; sourceName: string };

export function sourceName(source: AudioSource): string {
  return source.kind === "session" ? source.sourceName : source.file.name;
}

export function sourceFile(source: AudioSource): File | null {
  return source.kind === "session" ? null : source.file;
}

export type UploadPhase = "uploading" | "waking";

/** A server status still counts as slow after this long. */
export const SLOW_AFTER_MS = 8000;

/** What the status line currently says, before it's turned into copy. */
export type TranscribeStep =
  | { kind: "starting" }
  | { kind: "uploading"; phase: UploadPhase }
  | { kind: "stillWorking" }
  | { kind: "browser"; step: number } // 0–10, in 10% steps
  | { kind: "savingRetry" }
  | { kind: "submitRetry" }
  | { kind: "queued"; slow: boolean }
  | { kind: "processing"; slow: boolean }
  | { kind: "warmingUp" }
  | { kind: "restarting" };

export type Transcribing = {
  status: "transcribing";
  runId: number;
  source: AudioSource;
  // Basic Pitch runs in the browser first; "server" once it fell back.
  path: "browser" | "server";
  // A recording's upload runs alongside; its phase is shown only on the
  // server path, and "waking" holds off the still-working copy.
  upload: UploadPhase | null;
  step: TranscribeStep;
};

export type PipelineState<A> =
  | { status: "idle" }
  | Transcribing
  | { status: "generating"; runId: number; source: AudioSource }
  | { status: "completed"; runId: number; source: AudioSource; artifact: A }
  | {
      status: "failed";
      runId: number;
      source: AudioSource;
      error: DomainError;
      dismissed: boolean;
    };

export type PipelineEvent<A> =
  | { type: "start"; runId: number; source: AudioSource }
  | { type: "browserProgress"; runId: number; fraction: number }
  | { type: "savingRetry"; runId: number }
  | { type: "fellBack"; runId: number }
  | { type: "uploadPhase"; runId: number; phase: UploadPhase }
  | { type: "uploaded"; runId: number }
  | { type: "stillWorking"; runId: number }
  | { type: "submitRetry"; runId: number }
  | { type: "submitted"; runId: number }
  | {
      type: "serverStatus";
      runId: number;
      status: string;
      elapsedMs: number;
      waitingReason?: "warming_up" | null;
    }
  | { type: "restarting"; runId: number }
  | { type: "generationStarted"; runId: number }
  | { type: "completed"; runId: number; artifact: A }
  | { type: "failed"; runId: number; error: DomainError }
  | { type: "cancel"; runId: number }
  | { type: "dismissError" };

export const IDLE = { status: "idle" } as const;

function assertNever(value: never): never {
  throw new Error(`Unhandled pipeline event: ${JSON.stringify(value)}`);
}

function withStep<A>(state: Transcribing, step: TranscribeStep): PipelineState<A> {
  return { ...state, step };
}

/**
 * Pure transition function. Stale-run rule: an event carrying a runId that
 * isn't the current run's is ignored, which is what keeps a superseded
 * transcription from overwriting a newer one's progress.
 */
export function transition<A>(state: PipelineState<A>, event: PipelineEvent<A>): PipelineState<A> {
  if (event.type === "start") {
    return {
      status: "transcribing",
      runId: event.runId,
      source: event.source,
      path: "browser",
      upload: null,
      step: { kind: "starting" },
    };
  }
  if (event.type === "dismissError") {
    return state.status === "failed" && !state.dismissed ? { ...state, dismissed: true } : state;
  }
  if (state.status === "idle" || event.runId !== state.runId) {
    return state;
  }

  // Events that end or advance a run, from either running stage.
  const running = state.status === "transcribing" || state.status === "generating";
  switch (event.type) {
    case "completed":
      return running
        ? {
            status: "completed",
            runId: state.runId,
            source: state.source,
            artifact: event.artifact,
          }
        : state;
    case "failed":
      return running
        ? {
            status: "failed",
            runId: state.runId,
            source: state.source,
            error: event.error,
            dismissed: false,
          }
        : state;
    case "cancel":
      return running ? IDLE : state;
    default:
      break;
  }

  if (state.status !== "transcribing") {
    return state;
  }

  switch (event.type) {
    case "browserProgress": {
      const step = Math.min(10, Math.floor(event.fraction * 10));
      return state.step.kind === "browser" && state.step.step === step
        ? state
        : withStep(state, { kind: "browser", step });
    }
    case "savingRetry":
      return withStep(state, { kind: "savingRetry" });
    case "fellBack":
      return { ...state, path: "server" };
    case "uploadPhase":
      return state.path === "server"
        ? { ...state, upload: event.phase, step: { kind: "uploading", phase: event.phase } }
        : { ...state, upload: event.phase };
    case "uploaded":
      return { ...state, upload: null };
    case "stillWorking":
      return state.path === "server" && state.upload !== "waking"
        ? withStep(state, { kind: "stillWorking" })
        : state;
    case "submitRetry":
      return withStep(state, { kind: "submitRetry" });
    case "submitted":
      return withStep(state, { kind: "queued", slow: false });
    case "serverStatus": {
      const slow = event.elapsedMs > SLOW_AFTER_MS;
      if (event.status === "queued" && event.waitingReason === "warming_up") {
        return withStep(state, { kind: "warmingUp" });
      }
      if (event.status === "queued") return withStep(state, { kind: "queued", slow });
      if (event.status === "processing") return withStep(state, { kind: "processing", slow });
      return state;
    }
    case "restarting":
      return withStep(state, { kind: "restarting" });
    case "generationStarted":
      return { status: "generating", runId: state.runId, source: state.source };
    default:
      return assertNever(event);
  }
}
