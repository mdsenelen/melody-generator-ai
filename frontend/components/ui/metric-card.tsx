/**
 * MetricCard: one labelled analysis read-out ("Key: A minor", "Tempo: 96 BPM"). Conventions in
 * ./variants.
 *
 * Usage
 *   <MetricGroup label="Analysis">
 *     <MetricCard label="Key" value="A minor" detail="confidence 82%" />
 *     <MetricCard label="Tempo" value={96} unit="BPM" tone="accent" />
 *     <MetricCard label="Mood" status="loading" />
 *     <MetricCard label="Chords" status="unavailable" reason="Too few notes in this window" />
 *   </MetricGroup>
 *
 * Semantics     Each MetricCard is its own `<dl>` holding exactly one `<dt>` (label) and `<dd>`
 *               (value, unit, detail). That is the programmatic association, it is valid HTML
 *               anywhere (a lone card needs no wrapper, unlike a card that must live inside
 *               someone else's `<dl>`), and screen readers announce "Key, term / A minor,
 *               definition". `MetricGroup` is an optional named `role="group"` grid for a set of
 *               read-outs. A single shared `<dl>` was rejected: it forces every card to be a
 *               direct child of one wrapper, which breaks as soon as cards sit in separate cells
 *               or Cards.
 * Name          `label` is a required string; the value cannot render without it.
 * Status        Discriminated union, default "ready":
 *                 ready        { value, unit?, detail? }
 *                 loading      the label stays, a Skeleton replaces the value, the `<dl>` gets
 *                              aria-busy, the `<dd>` holds a visually hidden "Loading"
 *                 unavailable  { reason? }: the text "Not available" with a dash glyph (never an
 *                              empty box), the reason as the secondary line
 *               Contradictory props (a value while loading, a reason while ready) do not compile.
 * Tone          neutral (default), accent, success, warning, danger; exposed as `data-tone`.
 *               Tone colours the value only; the meaning must also be in the text (the label,
 *               value or detail), because colour is never the sole carrier.
 *
 * States
 *   default         label + value read-out
 *   loading         `status="loading"` (above)
 *   error           `status="unavailable"` (above). A read-out has no invalid input, so no
 *                   aria-invalid
 *   hover / focus-visible / active / disabled
 *                   do not apply: MetricCard is static content, not a control, and is not
 *                   focusable. If a read-out must become interactive, wrap it in a real button.
 */
import * as React from "react";
import { cva } from "class-variance-authority";

import { cn } from "./cn";
import { Skeleton } from "./skeleton";
import type { Exhaustive } from "./variants";

export type MetricTone = "neutral" | "accent" | "success" | "warning" | "danger";

const VALUE_TONES: Exhaustive<MetricTone> = {
  neutral: "text-text",
  accent: "text-accent-text",
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
};

const valueStyle = cva("font-mono text-lg leading-tight font-medium tabular-nums", {
  variants: { tone: VALUE_TONES },
  defaultVariants: { tone: "neutral" },
});

type StatusProps =
  | {
      status?: "ready";
      /** The read-out. Strings and numbers; use `unit` rather than baking units into it. */
      value: React.ReactNode;
      reason?: never;
    }
  | { status: "loading"; value?: never; unit?: never; detail?: never; reason?: never }
  | { status: "unavailable"; value?: never; unit?: never; reason?: string; detail?: never };

export type MetricCardProps = StatusProps & {
  label: string;
  /** Unit shown small beside the value ("BPM", "s", "Hz"). */
  unit?: string;
  /** Secondary line: a confidence, a range. */
  detail?: React.ReactNode;
  tone?: MetricTone;
  className?: string;
};

export const MetricCard = React.forwardRef<HTMLDListElement, MetricCardProps>(function MetricCard(
  { label, value, unit, detail, reason, status = "ready", tone = "neutral", className },
  ref,
) {
  const secondary = status === "unavailable" ? reason : detail;
  return (
    <dl
      ref={ref}
      data-tone={tone}
      data-status={status}
      aria-busy={status === "loading" ? true : undefined}
      className={cn(
        "border-border bg-surface min-w-0 rounded-[var(--radius)] border px-3 py-2",
        className,
      )}
    >
      <dt className="text-text-muted text-2xs font-mono tracking-widest uppercase">{label}</dt>
      <dd className="mt-1 min-w-0">
        {status === "loading" ? (
          <>
            <span className="sr-only">Loading</span>
            <Skeleton shape="text" className="h-5 w-16" />
          </>
        ) : status === "unavailable" ? (
          <span
            data-metric-unavailable=""
            className="text-text-subtle inline-flex items-center gap-1.5 text-xs"
          >
            <span aria-hidden="true" className="font-mono">
              &ndash;
            </span>
            Not available
          </span>
        ) : (
          <span className={valueStyle({ tone })}>
            {value}
            {unit ? (
              <span className="text-text-subtle text-label ml-1 font-normal tracking-wide">
                {unit}
              </span>
            ) : null}
          </span>
        )}
        {secondary ? (
          <span className="text-text-subtle text-label mt-0.5 block font-mono tabular-nums">
            {secondary}
          </span>
        ) : null}
      </dd>
    </dl>
  );
});

export type MetricGroupProps = {
  /** Names the set ("Analysis"). Required. */
  label: string;
  children: React.ReactNode;
  className?: string;
};

/** A named, responsive grid for a set of MetricCards. */
export function MetricGroup({ label, children, className }: MetricGroupProps) {
  return (
    <div
      role="group"
      aria-label={label}
      className={cn("grid grid-cols-[repeat(auto-fit,minmax(8rem,1fr))] gap-2", className)}
    >
      {children}
    </div>
  );
}
