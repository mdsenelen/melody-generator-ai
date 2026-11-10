"use client";

import { useMutation } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";

import { renderProgression } from "../app/lib/progressions";
import type {
  RenderProgressionRequest,
  RenderProgressionResponse,
} from "../src/domain/contracts/progressions";

/**
 * "Play progression": POST /api/generate-progression as a mutation. Each
 * render gets its own AbortSignal; starting another, or unmounting, aborts
 * the one still in flight. The last successful render stays available while
 * a new one runs or after it fails, as ChordGraph has always shown it.
 */
export function useRenderProgression() {
  const abortRef = useRef<AbortController | null>(null);
  const [lastResult, setLastResult] = useState<RenderProgressionResponse | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const mutation = useMutation({
    mutationFn: (request: RenderProgressionRequest) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      return renderProgression(request, { signal: controller.signal });
    },
  });

  const { mutate } = mutation;
  const render = useCallback(
    (request: RenderProgressionRequest) => {
      // Per-call callbacks only fire for the latest mutate call, so a
      // superseded (aborted) render can't overwrite a newer result.
      mutate(request, { onSuccess: setLastResult });
    },
    [mutate],
  );

  return {
    render,
    isRendering: mutation.isPending,
    error: mutation.error ? mutation.error.message || "Could not render progression" : null,
    lastResult,
  };
}
