"use client";

import { useId } from "react";

type SpinnerSize = "sm" | "md" | "lg";

type SpinnerProps = {
  size?: SpinnerSize;
  label: string;
  className?: string;
};

const SIZE_CLASSES: Record<SpinnerSize, string> = {
  sm: "h-4 w-4 border-2",
  md: "h-6 w-6 border-2",
  lg: "h-8 w-8 border-2",
};

export function Spinner({ size = "md", label, className = "" }: SpinnerProps) {
  const labelId = useId();

  return (
    <span
      role="status"
      aria-labelledby={labelId}
      className={`inline-flex items-center gap-2 ${className}`}
    >
      <span
        aria-hidden="true"
        className={`border-primary/40 motion-reduce:border-t-primary/40 inline-block animate-spin rounded-full border-t-transparent motion-reduce:animate-none ${SIZE_CLASSES[size]}`}
      />
      <span id={labelId}>{label}</span>
    </span>
  );
}
