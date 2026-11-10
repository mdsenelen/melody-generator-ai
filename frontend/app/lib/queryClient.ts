import { QueryClient } from "@tanstack/react-query";

import { retryDelayMs, shouldRetry } from "../../src/domain/retryPolicy";

/**
 * The app's one QueryClient configuration (app/providers.tsx creates the
 * instance). Queries retry transient failures only, per
 * src/domain/retryPolicy.ts; mutations never retry on their own.
 *
 * networkMode "always": a request goes out even when the browser reports
 * itself offline, and fails into the retry policy like any other network
 * error -- the cold-start handling already covers that path, and a query
 * silently paused on `navigator.onLine` would show neither waking nor error.
 */
export function createQueryClient(options: { retry?: boolean } = {}): QueryClient {
  const retry = options.retry ?? true;
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: retry ? shouldRetry : false,
        retryDelay: retryDelayMs,
        networkMode: "always",
        refetchOnWindowFocus: false,
      },
      mutations: {
        retry: false,
        networkMode: "always",
      },
    },
  });
}
