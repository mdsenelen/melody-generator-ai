import { generationSteps } from "../../src/domain/generationSteps";
import { IDLE, transition, type AudioSource, type PipelineState } from "../../src/domain/pipeline";

const file = new File(["x"], "take.wav", { type: "audio/wav" });
const upload: AudioSource = { kind: "upload", file, stored: { id: "u1", filename: "take.wav" } };

function transcribing(): PipelineState<unknown> {
  return transition(IDLE, { type: "start", runId: 1, source: upload });
}

describe("generationSteps", () => {
  it("is empty when nothing is running", () => {
    expect(generationSteps(IDLE)).toEqual([]);
  });

  it("starts with an indeterminate transcribe step and a pending generate step", () => {
    expect(generationSteps(transcribing())).toEqual([
      { id: "transcribe", label: "Transcribe your clip", status: "active", percent: null },
      { id: "generate", label: "Generate variants", status: "pending", percent: null },
    ]);
  });

  it("reports the real browser transcription percentage, nothing invented", () => {
    const state = transition(transcribing(), { type: "browserProgress", runId: 1, fraction: 0.4 });
    expect(generationSteps(state)[0]).toMatchObject({ status: "active", percent: 40 });
  });

  it("marks transcription done and generation active, with no percentage, once generating", () => {
    const state = transition(transcribing(), { type: "generationStarted", runId: 1 });
    expect(generationSteps(state)).toEqual([
      { id: "transcribe", label: "Transcribe your clip", status: "done", percent: 100 },
      { id: "generate", label: "Generate variants", status: "active", percent: null },
    ]);
  });

  it("is empty once the run has completed or failed", () => {
    const generating = transition(transcribing(), { type: "generationStarted", runId: 1 });
    expect(
      generationSteps(transition(generating, { type: "completed", runId: 1, artifact: 1 })),
    ).toEqual([]);
  });
});
