"use client";

import type { ReactNode } from "react";
import { ErrorBoundary as ReactErrorBoundary } from "react-error-boundary";

function ErrorFallback({ error }: { error: Error }) {
  return (
    <div
      role="alert"
      className="border-danger/40 bg-danger/8 text-foreground rounded-[var(--radius)] border p-4 text-sm"
    >
      <p className="font-semibold">Something went wrong.</p>
      <pre className="text-foreground mt-2 text-xs whitespace-pre-wrap">{error.message}</pre>
    </div>
  );
}

const ErrorBoundary = ({ children }: { children: ReactNode }) => (
  <ReactErrorBoundary FallbackComponent={ErrorFallback}>{children}</ReactErrorBoundary>
);

export default ErrorBoundary;
