"use client";

import { skipToken, useQuery } from "@tanstack/react-query";

import { analyzeClip } from "../app/lib/analyzeClip";
import { withAttemptTimeout } from "../app/lib/attemptTimeout";
import type { ClipAnalysis } from "../src/domain/contracts/analysis";
import { queryKeys } from "../src/domain/queryKeys";
import { isRetryingTransient } from "../src/domain/retryPolicy";
import { ANALYSIS_RETRY_MESSAGE } from "../src/domain/statusCopy";

/** One clip window (seconds) of a completed transcription job. */
export type ClipAnalysisRequest = { jobId: string; window: { start: number; end: number } };

export type AnalysisState =
  | { status: "idle" }
  | { status: "loading"; message?: string }
  | { status: "ready"; data: ClipAnalysis }
  | { status: "error"; error: string };

/**
 * POST /api/analyze for `request`, or nothing while it's null. Each window
 * is its own query: changing the window (or the job) moves to a new key and
 * aborts the request still running for the old one, so a stale analysis
 * can never land over a newer one.
 */
export function useClipAnalysis(request: ClipAnalysisRequest | null): {
  state: AnalysisState;
  reanalyse: () => void;
} {
  const query = useQuery({
    queryKey: request
      ? queryKeys.clipAnalysis(request.jobId, request.window.start, request.window.end)
      : queryKeys.noClipAnalysis(),
    queryFn: request
      ? ({ signal }) =>
          withAttemptTimeout(
            (attemptSignal) =>
              analyzeClip(
                {
                  job_id: request.jobId,
                  clip_start_sec: request.window.start,
                  clip_end_sec: request.window.end,
                },
                { signal: attemptSignal },
              ),
            { signal },
          )
      : skipToken,
    // A window's analysis is a pure function of the stored note events.
    staleTime: Infinity,
  });

  const { refetch } = query;
  const reanalyse = () => {
    void refetch();
  };

  if (!request) {
    return { state: { status: "idle" }, reanalyse };
  }
  // Any request in flight shows as loading, including an explicit re-run of
  // a window already analysed -- as every run did before this was a query.
  if (query.fetchStatus === "fetching") {
    return {
      state: isRetryingTransient(query)
        ? { status: "loading", message: ANALYSIS_RETRY_MESSAGE }
        : { status: "loading" },
      reanalyse,
    };
  }
  if (query.status === "success") {
    return { state: { status: "ready", data: query.data }, reanalyse };
  }
  if (query.status === "error") {
    return {
      state: { status: "error", error: query.error.message || "Analysis failed" },
      reanalyse,
    };
  }
  return { state: { status: "idle" }, reanalyse };
}
