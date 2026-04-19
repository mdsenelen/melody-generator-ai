"use client";

import type { ReactNode } from "react";
import { ErrorBoundary as ReactErrorBoundary } from "react-error-boundary";

function ErrorFallback({ error }: { error: Error }) {
  return (
    <div
      role="alert"
      className="rounded-2xl border border-red-500/50 bg-red-950/40 p-4 text-sm text-red-100"
    >
      <p className="font-semibold">Something went wrong.</p>
      <pre className="mt-2 whitespace-pre-wrap text-xs text-red-200">
        {error.message}
      </pre>
    </div>
  );
}

const ErrorBoundary = ({ children }: { children: ReactNode }) => (
  <ReactErrorBoundary FallbackComponent={ErrorFallback}>
    {children}
  </ReactErrorBoundary>
);

export default ErrorBoundary;
