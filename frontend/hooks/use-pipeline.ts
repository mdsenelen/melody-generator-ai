"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";

import {
  IDLE,
  transition,
  type AudioSource,
  type PipelineEvent,
  type PipelineState,
} from "../src/domain/pipeline";

export type PipelineRun = { runId: number; signal: AbortSignal };

/**
 * Wraps the pure pipeline machine in React: one reducer for state, and one
 * AbortController per run. Starting a run aborts the previous one, and
 * unmounting aborts whatever is running. Events from an aborted run carry a
 * stale runId, so the reducer drops them.
 */
export function usePipeline<A>() {
  const [state, dispatch] = useReducer(
    (current: PipelineState<A>, event: PipelineEvent<A>) => transition(current, event),
    IDLE as PipelineState<A>,
  );
  const runIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const start = useCallback((source: AudioSource): PipelineRun => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const runId = ++runIdRef.current;
    dispatch({ type: "start", runId, source });
    return { runId, signal: controller.signal };
  }, []);

  const dismissError = useCallback(() => dispatch({ type: "dismissError" }), []);

  return { state, dispatch, start, dismissError };
}
