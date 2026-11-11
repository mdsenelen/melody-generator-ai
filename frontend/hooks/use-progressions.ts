"use client";

import { useQuery } from "@tanstack/react-query";

import { fetchProgressions } from "../app/lib/progressions";
import type { Progression } from "../src/domain/contracts/progressions";
import { queryKeys } from "../src/domain/queryKeys";
import { isRetryingTransient } from "../src/domain/retryPolicy";

export type ProgressionsState =
  | { status: "loading" }
  // Retrying a transient failure: the free-tier backend is waking up.
  | { status: "waking" }
  | { status: "failed"; error: string }
  | { status: "ready"; progressions: Progression[] };

// The library only changes on deploy; no need to refetch it per visit.
const PROGRESSIONS_STALE_MS = 5 * 60 * 1000;

export function useProgressions(): ProgressionsState {
  const query = useQuery({
    queryKey: queryKeys.progressions(),
    queryFn: ({ signal }) => fetchProgressions({ signal }),
    staleTime: PROGRESSIONS_STALE_MS,
  });

  if (query.status === "success") {
    return { status: "ready", progressions: query.data };
  }
  if (query.status === "error") {
    return { status: "failed", error: query.error.message || "Failed to load progressions" };
  }
  return isRetryingTransient(query) ? { status: "waking" } : { status: "loading" };
}
