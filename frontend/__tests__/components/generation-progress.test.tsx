import { act, render, screen, within } from "@testing-library/react";

import { GenerationProgress } from "../../components/generation-progress";
import { IDLE, transition, type AudioSource, type PipelineState } from "../../src/domain/pipeline";

const file = new File(["x"], "take.wav", { type: "audio/wav" });
const source: AudioSource = { kind: "upload", file, stored: { id: "u1", filename: "take.wav" } };

const transcribing = (): PipelineState<unknown> =>
  transition(IDLE, { type: "start", runId: 1, source });

describe("GenerationProgress", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = jest.fn();
  });
  afterEach(() => jest.useRealTimers());

  it("renders nothing when no run is in progress", () => {
    const { container } = render(<GenerationProgress state={IDLE} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("lists both steps and marks the current one", () => {
    render(<GenerationProgress state={transcribing()} />);
    const steps = screen.getAllByRole("listitem");
    expect(steps).toHaveLength(2);
    expect(steps[0]).toHaveAttribute("aria-current", "step");
    expect(steps[1]).not.toHaveAttribute("aria-current");
  });

  it("shows a determinate bar with the real percentage while transcribing in the browser", () => {
    const state = transition(transcribing(), { type: "browserProgress", runId: 1, fraction: 0.4 });
    render(<GenerationProgress state={state} />);
    const bar = screen.getByRole("progressbar", { name: /transcribe your clip/i });
    expect(bar).toHaveAttribute("aria-valuenow", "40");
    expect(screen.getByText("40%")).toBeInTheDocument();
  });

  it("shows an indeterminate bar (no aria-valuenow, no percent) while generating", () => {
    const state = transition(transcribing(), { type: "generationStarted", runId: 1 });
    render(<GenerationProgress state={state} />);
    const bar = screen.getByRole("progressbar", { name: /generate variants/i });
    expect(bar).not.toHaveAttribute("aria-valuenow");
    expect(screen.queryByText(/\d+%/)).not.toBeInTheDocument();
    const done = screen.getAllByRole("listitem")[0];
    expect(within(done).getByText(/done/i)).toBeInTheDocument();
  });

  it("counts elapsed time so a long wait doesn't look frozen", () => {
    jest.useFakeTimers();
    const state = transition(transcribing(), { type: "generationStarted", runId: 1 });
    render(<GenerationProgress state={state} />);
    expect(screen.getByText(/elapsed 0:00/i)).toBeInTheDocument();
    act(() => {
      jest.advanceTimersByTime(65_000);
    });
    expect(screen.getByText(/elapsed 1:05/i)).toBeInTheDocument();
  });

  it("scrolls itself into view when it appears, since it can stack below the controls", () => {
    const scrollIntoView = jest.fn();
    Element.prototype.scrollIntoView = scrollIntoView;
    render(<GenerationProgress state={transcribing()} />);
    expect(scrollIntoView).toHaveBeenCalledTimes(1);
    expect(scrollIntoView).toHaveBeenCalledWith(expect.objectContaining({ block: "nearest" }));
  });

  it("is not a live region (the Generate button's spinner already announces status)", () => {
    const { container } = render(<GenerationProgress state={transcribing()} />);
    expect(container.querySelector("[aria-live], [role='status'], [role='alert']")).toBeNull();
  });
});
