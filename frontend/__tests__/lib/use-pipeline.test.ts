import { act, renderHook } from "@testing-library/react";

import { usePipeline } from "../../hooks/use-pipeline";
import type { AudioSource } from "../../src/domain/pipeline";

const source: AudioSource = { kind: "recording", file: new File(["x"], "hum.webm") };

describe("usePipeline", () => {
  it("starts idle", () => {
    const { result } = renderHook(() => usePipeline<string>());
    expect(result.current.state).toEqual({ status: "idle" });
  });

  it("gives each run a fresh id and signal, aborting the previous run", () => {
    const { result } = renderHook(() => usePipeline<string>());
    let first!: { runId: number; signal: AbortSignal };
    let second!: { runId: number; signal: AbortSignal };
    act(() => {
      first = result.current.start(source);
    });
    expect(result.current.state).toMatchObject({ status: "transcribing", runId: first.runId });
    act(() => {
      second = result.current.start(source);
    });
    expect(second.runId).toBeGreaterThan(first.runId);
    expect(first.signal.aborted).toBe(true);
    expect(second.signal.aborted).toBe(false);
  });

  it("drops events from a superseded run", () => {
    const { result } = renderHook(() => usePipeline<string>());
    let first!: { runId: number };
    act(() => {
      first = result.current.start(source);
      result.current.start(source);
    });
    act(() => {
      result.current.dispatch({ type: "completed", runId: first.runId, artifact: "stale" });
    });
    expect(result.current.state.status).toBe("transcribing");
  });

  it("aborts the running run on unmount", () => {
    const { result, unmount } = renderHook(() => usePipeline<string>());
    let run!: { signal: AbortSignal };
    act(() => {
      run = result.current.start(source);
    });
    unmount();
    expect(run.signal.aborted).toBe(true);
  });

  it("dismisses a failure", () => {
    const { result } = renderHook(() => usePipeline<string>());
    act(() => {
      const { runId } = result.current.start(source);
      result.current.dispatch({
        type: "failed",
        runId,
        error: { code: "failed", message: "boom", retryable: true },
      });
    });
    act(() => result.current.dismissError());
    expect(result.current.state).toMatchObject({ status: "failed", dismissed: true });
  });
});
