"use client";

import { useEffect, useState } from "react";

import type { JobResult } from "../app/lib/jobResult";
import { getTranscribeJob } from "../app/lib/transcribeJob";

export type JobResultState =
  | { status: "loading" }
  | { status: "running"; jobStatus: "queued" | "processing" }
  | { status: "succeeded"; result: JobResult }
  | { status: "failed"; error: string }
  // Covers both a fabricated/unknown job id (404) and a real job whose
  // generated files are past DATA_RETENTION_HOURS (backend reports this as
  // status "expired" -- see jobs/routes.py) -- from the user's side there's
  // no meaningful difference between "never existed" and "gone now".
  | { status: "gone" };

const POLL_INTERVAL_MS = 2000;
// Matches transcribeJob.ts's POLL_MAX_TOTAL_MS: no legitimate job should
// still be queued/processing this long after this page started watching it.
const POLL_MAX_TOTAL_MS = 15 * 60 * 1000;

export function useJobResult(jobId: string): JobResultState {
  const [state, setState] = useState<JobResultState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const startedAt = Date.now();
    setState({ status: "loading" });

    async function poll() {
      try {
        const response = await getTranscribeJob<JobResult>(jobId);
        if (cancelled) return;

        if (response.status === "completed") {
          if (!response.result) {
            setState({ status: "failed", error: "Job completed with no result" });
            return;
          }
          setState({ status: "succeeded", result: response.result });
          return;
        }
        if (response.status === "failed") {
          setState({ status: "failed", error: response.error || "Generation failed" });
          return;
        }
        if (response.status === "expired") {
          setState({ status: "gone" });
          return;
        }

        if (Date.now() - startedAt > POLL_MAX_TOTAL_MS) {
          setState({
            status: "failed",
            error: "This is taking longer than expected. Please try again in a moment.",
          });
          return;
        }
        setState({ status: "running", jobStatus: response.status });
        timer = setTimeout(() => void poll(), POLL_INTERVAL_MS);
      } catch (error) {
        if (cancelled) return;
        const status = (error as { status?: number }).status;
        if (status === 404) {
          setState({ status: "gone" });
          return;
        }
        setState({
          status: "failed",
          error: error instanceof Error ? error.message : "Could not load this result",
        });
      }
    }

    void poll();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  return state;
}
