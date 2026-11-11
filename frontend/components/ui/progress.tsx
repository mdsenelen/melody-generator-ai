/**
 * Progress: a read-only meter for long operations (transcription, generation) on Radix Progress.
 * Conventions in ./variants.
 *
 * Usage
 *   <Progress label="Transcribing" value={chunksDone} max={chunkCount} showValue />
 *   <Progress aria-label="Generating melody" indeterminate />
 *   <Progress aria-labelledby="job-title" value={40} tone="danger" />
 *
 * Name        required by the type, one of: `label` (renders visible text and wires it up through
 *             aria-labelledby, preferred), `aria-label`, or `aria-labelledby`. The mode is a
 *             discriminated union: `value` (determinate, optional `max`, default 100) OR
 *             `indeterminate`; both or neither is a compile error. Out-of-range and NaN values
 *             are clamped to [0, max]. `aria-valuetext` is the percentage ("25%").
 * Live region NOT one. Progress never sets role=status or aria-live and does not announce ticks.
 *             The page owns the live-region status text ("Transcribing, 3 of 12 sections").
 * Motion      indeterminate is a sweeping bar; under prefers-reduced-motion it is replaced by a
 *             static diagonal-hatched full-width bar, which still reads as "in progress, amount
 *             unknown" (it is neither empty nor a completed solid fill).
 * Tone        accent (default), success, warning, danger; exposed as `data-tone`. Danger adds a
 *             "!" glyph so failure is not colour alone.
 *
 * States
 *   default         track + tone-coloured fill
 *   error           `tone="danger"` (above). Not keyed on aria-invalid: progressbar does not
 *                   support it.
 *   hover / focus-visible / active / disabled / loading
 *                   do not apply: Progress is not interactive and is not focusable; it IS the
 *                   loading indicator.
 */
import * as React from "react";
import * as ProgressPrimitive from "@radix-ui/react-progress";
import { cva } from "class-variance-authority";

import { ButtonErrorMark } from "./button";
import { cn } from "./cn";
import type { Exhaustive } from "./variants";

export type ProgressTone = "accent" | "success" | "warning" | "danger";

const FILL_TONES: Exhaustive<ProgressTone> = {
  accent: "bg-accent",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-danger",
};

const TEXT_TONES: Exhaustive<ProgressTone> = {
  accent: "text-accent-text",
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
};

const fill = cva("h-full", {
  variants: { tone: FILL_TONES },
  defaultVariants: { tone: "accent" },
});

type NameProps =
  | { label: string; "aria-label"?: never; "aria-labelledby"?: never }
  | { label?: undefined; "aria-label": string; "aria-labelledby"?: never }
  | { label?: undefined; "aria-label"?: never; "aria-labelledby": string };

type ModeProps =
  | { value: number; max?: number; indeterminate?: never }
  | { indeterminate: true; value?: never; max?: never };

export type ProgressProps = NameProps &
  ModeProps & {
    tone?: ProgressTone;
    /** Show the percentage as mono text beside the bar (determinate only). */
    showValue?: boolean;
    className?: string;
  };

function clamp(value: number, max: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(Math.max(value, 0), max);
}

export const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(function Progress(
  {
    label,
    value,
    max: maxProp,
    indeterminate,
    tone = "accent",
    showValue = false,
    className,
    "aria-label": ariaLabel,
    "aria-labelledby": ariaLabelledBy,
  },
  ref,
) {
  const labelId = React.useId();
  const max = maxProp !== undefined && maxProp > 0 ? maxProp : 100;
  const current = indeterminate ? null : clamp(value ?? 0, max);
  const percent = current === null ? null : Math.round((current / max) * 100);

  return (
    <div className={cn("w-full space-y-1", className)}>
      {label || showValue || tone === "danger" ? (
        <div className="flex items-baseline justify-between gap-2">
          {label ? (
            <span
              id={labelId}
              className="text-text-muted text-2xs font-mono tracking-widest uppercase"
            >
              {label}
            </span>
          ) : (
            <span />
          )}
          <span
            className={cn(
              "text-label inline-flex items-center gap-1 font-mono tabular-nums",
              TEXT_TONES[tone],
            )}
          >
            {tone === "danger" ? (
              <span data-progress-error="" className="inline-flex">
                <ButtonErrorMark className="size-3" />
              </span>
            ) : null}
            {showValue && percent !== null ? <span>{percent}%</span> : null}
          </span>
        </div>
      ) : null}
      <ProgressPrimitive.Root
        ref={ref}
        value={current}
        max={max}
        data-tone={tone}
        aria-label={ariaLabel}
        aria-labelledby={label ? labelId : ariaLabelledBy}
        aria-valuetext={percent === null ? undefined : `${percent}%`}
        className="bg-border-control/40 relative h-1 w-full overflow-hidden rounded-[var(--radius)]"
      >
        {indeterminate ? (
          <ProgressPrimitive.Indicator
            data-progress-indeterminate=""
            className={cn(
              fill({ tone }),
              "w-2/5 animate-[ui-progress-sweep_1.4s_var(--ease-standard)_infinite]",
              // Reduced motion: no sweep; a static hatched bar across the full width.
              "motion-reduce:w-full motion-reduce:animate-none motion-reduce:bg-transparent",
              "motion-reduce:bg-[repeating-linear-gradient(135deg,currentColor_0_4px,transparent_4px_8px)]",
              TEXT_TONES[tone],
            )}
          />
        ) : (
          <ProgressPrimitive.Indicator
            className={cn(
              fill({ tone }),
              "ease-standard transition-[width] duration-(--duration-base)",
            )}
            style={{ width: `${percent ?? 0}%` }}
          />
        )}
      </ProgressPrimitive.Root>
    </div>
  );
});
