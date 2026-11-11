// Renders with a fresh QueryClient per call, so no cache leaks between tests.
// Retries are off by default (a failing request fails at once, keeping tests
// fast and deterministic); pass { retry: true } to run the app's real retry
// policy (src/domain/retryPolicy.ts), e.g. for cold-start tests.
import { QueryClientProvider, type QueryClient } from "@tanstack/react-query";
import { render, renderHook, type RenderOptions } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";

import { createQueryClient } from "../../../app/lib/queryClient";

type QueryOptions = { retry?: boolean; queryClient?: QueryClient };

export function createTestQueryClient({ retry = false }: { retry?: boolean } = {}) {
  return createQueryClient({ retry });
}

function wrapperFor(queryClient: QueryClient) {
  return function QueryWrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
}

export function renderWithQueryClient(
  ui: ReactElement,
  { retry, queryClient, ...options }: QueryOptions & Omit<RenderOptions, "wrapper"> = {},
) {
  const client = queryClient ?? createTestQueryClient({ retry });
  return { queryClient: client, ...render(ui, { wrapper: wrapperFor(client), ...options }) };
}

export function renderHookWithQueryClient<Result, Props>(
  hook: (props: Props) => Result,
  { retry, queryClient, initialProps }: QueryOptions & { initialProps?: Props } = {},
) {
  const client = queryClient ?? createTestQueryClient({ retry });
  return {
    queryClient: client,
    ...renderHook(hook, { wrapper: wrapperFor(client), initialProps }),
  };
}
