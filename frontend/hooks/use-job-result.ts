"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { getJob } from "../app/lib/transcribeJob";
import type { JobResult, JobResultStatusResponse } from "../src/domain/contracts/job";
import { queryKeys } from "../src/domain/queryKeys";
import { isRetryingTransient } from "../src/domain/retryPolicy";

export type JobResultState =
  | { status: "loading" }
  // A transient failure (502/503/network) is being retried: the free-tier
  // backend is waking up or restarting. The job itself survives that.
  | { status: "waking" }
  | { status: "running"; jobStatus: "queued" | "processing" }
  | { status: "succeeded"; result: JobResult }
  | { status: "failed"; error: string }
  // Covers both a fabricated/unknown job id (404) and a real job whose
  // generated files are past DATA_RETENTION_HOURS (backend reports this as
  // status "expired" -- see jobs/routes.py) -- from the user's side there's
  // no meaningful difference between "never existed" and "gone now".
  | { status: "gone" };

const POLL_INTERVAL_MS = 2000;
// How long this page keeps polling a queued/processing job. Note: shorter
// than transcribeJob.ts's POLL_MAX_TOTAL_MS (20 min), which was raised to
// cover the worst-case restart-recovery chain; this one was not.
const POLL_MAX_TOTAL_MS = 15 * 60 * 1000;

const TIMEOUT_MESSAGE = "This is taking longer than expected. Please try again in a moment.";

type PollOptions = { pollIntervalMs?: number; pollMaxTotalMs?: number };

function isRunning(
  response: JobResultStatusResponse | undefined,
): response is JobResultStatusResponse & { status: "queued" | "processing" } {
  return response?.status === "queued" || response?.status === "processing";
}

function hasStatus(error: unknown, status: number): boolean {
  return (
    typeof error === "object" && error !== null && "status" in error && error.status === status
  );
}

/**
 * Watches any job (transcription, variants, progression) until it settles.
 * Polls every `pollIntervalMs` while queued/processing, and gives up after
 * `pollMaxTotalMs` of watching.
 */
export function useJobResult(
  jobId: string,
  { pollIntervalMs = POLL_INTERVAL_MS, pollMaxTotalMs = POLL_MAX_TOTAL_MS }: PollOptions = {},
): JobResultState {
  // When this page started watching this job; reset when the id changes.
  const [watch, setWatch] = useState(() => ({ jobId, startedAt: Date.now() }));
  if (watch.jobId !== jobId) {
    setWatch({ jobId, startedAt: Date.now() });
  }
  const overBudget = (updatedAt: number) => updatedAt - watch.startedAt > pollMaxTotalMs;

  const query = useQuery({
    queryKey: queryKeys.job(jobId),
    queryFn: ({ signal }) => getJob(jobId, { signal }),
    refetchInterval: (current) => {
      const { data, dataUpdatedAt, status } = current.state;
      if (status === "error" || !isRunning(data) || overBudget(dataUpdatedAt)) {
        return false;
      }
      return pollIntervalMs;
    },
    // The old hand-rolled poll kept going in a background tab; so does this.
    refetchIntervalInBackground: true,
  });

  if (isRetryingTransient(query)) {
    return { status: "waking" };
  }
  if (query.status === "pending") {
    return { status: "loading" };
  }
  if (query.status === "error") {
    if (hasStatus(query.error, 404)) {
      return { status: "gone" };
    }
    return { status: "failed", error: query.error.message || "Could not load this result" };
  }

  const response = query.data;
  switch (response.status) {
    case "completed":
      return response.result
        ? { status: "succeeded", result: response.result }
        : { status: "failed", error: "Job completed with no result" };
    case "failed":
      return { status: "failed", error: response.error || "Generation failed" };
    case "expired":
      return { status: "gone" };
    case "queued":
    case "processing":
      return overBudget(query.dataUpdatedAt)
        ? { status: "failed", error: TIMEOUT_MESSAGE }
        : { status: "running", jobStatus: response.status };
  }
}
