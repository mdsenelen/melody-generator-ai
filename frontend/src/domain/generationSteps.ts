// What the "generating variants" progress panel shows, derived from the
// pipeline state. Framework-free. Only transcription has a real percentage
// (Basic Pitch reports one in the browser); generation is a single server
// request with no progress feed, so it stays indeterminate (percent: null)
// rather than showing a made-up number.

import type { PipelineState } from "./pipeline";

export type GenerationStepStatus = "pending" | "active" | "done";

export type GenerationStep = {
  id: "transcribe" | "generate";
  label: string;
  status: GenerationStepStatus;
  /** 0-100 when the work reports real progress, null when it can't. */
  percent: number | null;
};

export function generationSteps<A>(state: PipelineState<A>): GenerationStep[] {
  if (state.status === "transcribing") {
    return [
      {
        id: "transcribe",
        label: "Transcribe your clip",
        status: "active",
        percent: state.step.kind === "browser" ? state.step.step * 10 : null,
      },
      { id: "generate", label: "Generate variants", status: "pending", percent: null },
    ];
  }
  if (state.status === "generating") {
    return [
      { id: "transcribe", label: "Transcribe your clip", status: "done", percent: 100 },
      { id: "generate", label: "Generate variants", status: "active", percent: null },
    ];
  }
  return [];
}
