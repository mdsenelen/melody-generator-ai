/**
 * Slider: a single-thumb value control on Radix Slider. Conventions in ./variants.
 *
 * Usage
 *   <Slider label="Tempo" value={bpm} min={60} max={160} unit=" BPM" onChange={setBpm} />
 *   <Slider label="Number of variants" value={n} min={1} max={8} disabled={busy} />
 *   <Slider label="Tempo" aria-invalid aria-describedby="tempo-err" ... />
 *
 * Name        the visible label is the accessible name (`aria-labelledby`). An explicit
 *             `aria-label` overrides it. `aria-valuetext` is `${value}${unit}`.
 * Keyboard    Arrow keys step by `step`, Shift+Arrow and PageUp/PageDown by 10 steps, Home/End to
 *             min/max (Radix).
 * Hit area    the thumb element is 24x24 CSS px; the visible dot is 12px inside it.
 *
 * States
 *   default         hairline track, accent range, accent dot, mono readout in `text-accent-text`
 *   hover           dot grows (pointer devices)
 *   focus-visible   FOCUS_RING on the 24px thumb (keyboard only)
 *   active          dot grows further while dragging
 *   disabled        Radix `disabled`: not focusable, no changes, `aria-disabled`, DISABLED-level dim
 *   error           `aria-invalid`: danger range/dot/readout plus an "!" glyph before the readout
 *                   (never colour alone). Put the message in an element and set `aria-describedby`.
 *   loading         does not apply: a slider only holds a value, it does not start work. The
 *                   control that consumes the value (e.g. "Generate melody") owns loading.
 * Single thumb only; there is no range variant.
 */
import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";

import { ButtonErrorMark, isInvalid } from "./button";
import { cn } from "./cn";
import { FOCUS_RING } from "./variants";

export type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange?: (value: number) => void;
  disabled?: boolean;
  className?: string;
} & Pick<React.AriaAttributes, "aria-label" | "aria-invalid" | "aria-describedby">;

export const Slider = React.forwardRef<HTMLSpanElement, SliderProps>(function Slider(
  {
    label,
    value,
    min,
    max,
    step = 1,
    unit = "",
    onChange,
    disabled = false,
    className,
    "aria-label": ariaLabel,
    "aria-invalid": ariaInvalid,
    "aria-describedby": describedBy,
  },
  ref,
) {
  const labelId = React.useId();
  const invalid = isInvalid(ariaInvalid);

  return (
    <div className={cn("space-y-1", disabled && "opacity-30", className)}>
      <div className="flex items-baseline justify-between gap-2">
        <span id={labelId} className="text-text-muted text-2xs font-mono tracking-widest uppercase">
          {label}
        </span>
        <span
          className={cn(
            "text-label inline-flex items-center gap-1 font-mono tabular-nums",
            invalid ? "text-danger" : "text-accent-text",
          )}
        >
          {invalid ? (
            <span data-slider-error="" className="inline-flex">
              <ButtonErrorMark className="size-3" />
            </span>
          ) : null}
          <span>
            {value}
            {unit}
          </span>
        </span>
      </div>
      <SliderPrimitive.Root
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onValueChange={([next]) => {
          if (next !== undefined) onChange?.(next);
        }}
        className={cn(
          "relative flex h-6 w-full touch-none items-center select-none",
          disabled ? "cursor-not-allowed" : "cursor-pointer",
        )}
      >
        <SliderPrimitive.Track className="bg-border-control/50 relative h-px grow">
          <SliderPrimitive.Range
            className={cn("absolute h-full", invalid ? "bg-danger" : "bg-accent")}
          />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          ref={ref}
          aria-label={ariaLabel}
          aria-labelledby={ariaLabel ? undefined : labelId}
          aria-valuetext={`${value}${unit}`}
          aria-invalid={invalid ? true : undefined}
          aria-describedby={describedBy}
          aria-disabled={disabled ? true : undefined}
          className={cn(
            // No `outline-none` here: in Tailwind v4 it also sets --tw-outline-style, which
            // FOCUS_RING's `focus-visible:outline-2` reads, so the ring would never render.
            "group flex size-6 items-center justify-center rounded-full",
            FOCUS_RING,
          )}
        >
          <span
            aria-hidden="true"
            className={cn(
              "ease-standard size-3 rounded-full border-2 transition-transform duration-(--duration-fast)",
              "border-surface group-hover:scale-125 group-active:scale-150 group-aria-invalid:rounded-xs",
              invalid ? "bg-danger" : "bg-accent",
            )}
          />
        </SliderPrimitive.Thumb>
      </SliderPrimitive.Root>
    </div>
  );
});

/** A compact 0.3-2.0 temperature dial, one per generated variant. */
export function TemperatureInput({
  label,
  value,
  onChange,
  disabled = false,
}: {
  label?: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  const id = React.useId();
  const pct = ((value - 0.3) / 1.7) * 100;
  const warmth = value > 1.2 ? "text-amber-400" : value < 0.7 ? "text-sky-400" : "text-primary";

  return (
    <div className={disabled ? "flex items-center gap-2 opacity-40" : "flex items-center gap-2"}>
      {label ? (
        <label htmlFor={id} className="text-muted-foreground w-5 shrink-0 font-mono text-[10px]">
          {label}
        </label>
      ) : null}
      <span className={`w-7 shrink-0 text-right font-mono text-[11px] ${warmth}`}>
        {value.toFixed(1)}
      </span>
      <div className="bg-border relative h-px flex-1">
        <div className="bg-primary absolute top-0 left-0 h-full" style={{ width: `${pct}%` }} />
        <input
          id={id}
          type="range"
          min={0.3}
          max={2.0}
          step={0.1}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(Number(event.target.value))}
          className="absolute inset-0 w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
          style={{ height: 20, top: -10 }}
        />
      </div>
    </div>
  );
}
