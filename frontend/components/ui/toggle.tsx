/**
 * Toggle: a two-state pressed button ("Loop", "Metronome", "Mute") on Radix Toggle.
 * Conventions in ./variants. Use Button for "do this now"; use Toggle for "this is on/off".
 *
 * Usage
 *   <Toggle defaultPressed>Loop</Toggle>
 *   <Toggle pressed={muted} onPressedChange={setMuted} label="Mute track" icon={<MuteIcon />} />
 *   <Toggle loading={switching}>Metronome</Toggle>
 *
 * Name        text children, or the icon-only form: `label` (required, becomes `aria-label`) plus
 *             `icon` (wrapped in an `aria-hidden` span). `icon` without `label` is a compile error.
 * Pressed     `aria-pressed` + `data-state="on"`. Not colour alone: pressed fills the control,
 *             strengthens its border, and shows a filled bar indicator (`data-toggle-indicator`).
 *
 * States
 *   default         outlined, muted text
 *   hover           stronger border and text (pointer devices)
 *   focus-visible   FOCUS_RING
 *   active          surface-raised fill while pressed down
 *   disabled        native `disabled`; no activation
 *   loading         `aria-busy="true"`, stays focusable, activation (click, Space, Enter) is
 *                   swallowed so the state cannot change. The page owns any status announcement.
 *   error           `aria-invalid="true"`: danger border and text plus an "!" glyph. The message
 *                   goes in an element referenced by `aria-describedby`. (aria-invalid is not an
 *                   ARIA attribute for role=button, so it is styling/convention only here.)
 */
import * as React from "react";
import * as TogglePrimitive from "@radix-ui/react-toggle";
import { cva } from "class-variance-authority";

import { ButtonErrorMark, ButtonSpinner, isInvalid, type ButtonSize } from "./button";
import { cn } from "./cn";
import { DISABLED, ERROR, FOCUS_RING, LOADING, type Exhaustive } from "./variants";

export type ToggleVariant = "outline" | "ghost";
export type ToggleSize = ButtonSize;

const TOGGLE_VARIANTS: Exhaustive<ToggleVariant> = {
  outline:
    "border border-border-control text-text-muted hover:border-text-muted hover:text-text active:bg-surface-raised data-[state=on]:border-accent data-[state=on]:bg-accent/15 data-[state=on]:text-text data-[state=on]:font-semibold",
  ghost:
    "border border-transparent text-text-muted hover:text-text hover:bg-surface-raised active:bg-surface-muted data-[state=on]:border-border-strong data-[state=on]:bg-surface-overlay data-[state=on]:text-text data-[state=on]:font-semibold",
};

const TOGGLE_SIZES: Exhaustive<ToggleSize> = {
  sm: "min-h-7 px-2.5 text-xs",
  md: "min-h-10 px-3.5 text-sm",
};

const toggleVariants = cva(
  [
    "group relative inline-flex items-center justify-center gap-2 rounded-[var(--radius)] font-medium tracking-wide",
    "transition-colors duration-(--duration-fast) ease-standard",
    FOCUS_RING,
    DISABLED,
    LOADING,
    ERROR,
    "aria-invalid:bg-danger-surface",
  ],
  {
    variants: { variant: TOGGLE_VARIANTS, size: TOGGLE_SIZES },
    defaultVariants: { variant: "outline", size: "md" },
  },
);

type ToggleBaseProps = Omit<
  React.ComponentPropsWithoutRef<typeof TogglePrimitive.Root>,
  "children" | "aria-label" | "aria-labelledby" | "asChild"
> & {
  variant?: ToggleVariant;
  size?: ToggleSize;
  loading?: boolean;
};

/** Text form: children are the accessible name. */
export type TextToggleProps = ToggleBaseProps & {
  children: React.ReactNode;
  label?: never;
  icon?: never;
};

/** Icon-only form: `label` is required and is the accessible name. */
export type IconToggleProps = ToggleBaseProps & {
  label: string;
  icon: React.ReactNode;
  children?: never;
};

export type ToggleProps = TextToggleProps | IconToggleProps;

export const Toggle = React.forwardRef<HTMLButtonElement, ToggleProps>(function Toggle(
  { variant, size, loading = false, className, children, label, icon, onClick, ...rest },
  ref,
) {
  const showError = isInvalid(rest["aria-invalid"]) && !loading;

  return (
    <TogglePrimitive.Root
      ref={ref}
      aria-label={label}
      aria-busy={loading ? true : undefined}
      className={cn(toggleVariants({ variant, size }), className)}
      {...rest}
      onClick={(event) => {
        if (loading) {
          // Radix toggles in its own click handler; preventing default makes it skip.
          event.preventDefault();
          return;
        }
        onClick?.(event);
      }}
    >
      <span
        aria-hidden="true"
        data-toggle-indicator=""
        className={cn(
          "h-3 w-0.5 shrink-0 rounded-full bg-current opacity-0 transition-opacity duration-(--duration-fast)",
          "group-data-[state=on]:opacity-100",
          loading && "group-data-[state=on]:opacity-0",
        )}
      />
      {showError ? <ButtonErrorMark /> : null}
      <span
        data-toggle-label=""
        className={cn("inline-flex items-center gap-2", loading && "opacity-0")}
      >
        {icon !== undefined ? (
          <span aria-hidden="true" className="inline-flex">
            {icon}
          </span>
        ) : (
          children
        )}
      </span>
      {loading ? <ButtonSpinner /> : null}
    </TogglePrimitive.Root>
  );
});
