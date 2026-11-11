/**
 * Button: the one control for "do this now". Built on cva + Radix Slot; conventions in ./variants.
 *
 * Usage
 *   <Button onClick={generate}>Generate melody</Button>
 *   <Button variant="secondary" size="sm" loading={busy}>Analyse this section</Button>
 *   <Button variant="destructive" aria-invalid={failed} aria-describedby="err">Retry</Button>
 *   <Button asChild variant="ghost"><a href="/result/1">View result</a></Button>
 *
 * States
 *   focus-visible  FOCUS_RING (full-strength outline token)
 *   disabled       native `disabled`; no activation
 *   loading        `aria-busy="true"`, stays focusable, onClick is swallowed (click, Enter, Space),
 *                  label stays in the DOM (invisible) so width and accessible name are unchanged.
 *                  Button does NOT announce: the caller owns the live region (its status text,
 *                  e.g. "Analysing…"), because a role=status inside a <button> is presentational
 *                  to assistive tech and a sibling would break `asChild`.
 *   error          `aria-invalid="true"`: danger colours plus an "!" glyph (never colour alone).
 *                  Put the message in an element and point `aria-describedby` at it.
 *
 * `buttonClass()` stays for call sites that must render a `<Link>`/`<a>` with the same classes.
 * `small` is the legacy spelling of `size="sm"`; prefer `size`.
 */
import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";

import { cn } from "./cn";
import {
  DISABLED,
  ERROR,
  FOCUS_RING,
  LOADING,
  type Exhaustive,
  type VariantPropsOf,
} from "./variants";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "destructive";
export type ButtonSize = "sm" | "md";

export const BUTTON_BASE = [
  "relative inline-flex items-center justify-center gap-2 rounded-[var(--radius)] font-medium tracking-wide",
  "transition-colors duration-(--duration-fast)",
  FOCUS_RING,
  DISABLED,
  LOADING,
  ERROR,
];

/** Shared with IconButton so the two never drift. */
export const BUTTON_VARIANTS: Exhaustive<ButtonVariant> = {
  primary:
    "bg-accent text-accent-foreground hover:bg-accent/85 active:bg-accent/70 font-display aria-invalid:bg-danger-surface aria-invalid:text-danger-foreground aria-invalid:shadow-[inset_0_0_0_1px_var(--color-danger)]",
  secondary:
    "bg-accent/12 text-accent border border-accent/30 hover:bg-accent/20 hover:border-accent/50 active:bg-accent/25 font-display aria-invalid:bg-danger-surface",
  ghost:
    "border border-border text-text-muted hover:border-border-strong hover:text-text active:bg-surface-raised aria-invalid:bg-danger-surface",
  destructive:
    "border border-danger-border text-danger hover:bg-danger/10 hover:border-danger active:bg-danger/20 aria-invalid:bg-danger-surface",
};

const BUTTON_SIZES: Exhaustive<ButtonSize> = {
  sm: "px-3 py-1.5 text-xs",
  md: "px-4 py-2.5 text-sm",
};

const buttonVariants = cva(BUTTON_BASE, {
  variants: { variant: BUTTON_VARIANTS, size: BUTTON_SIZES },
  defaultVariants: { variant: "primary", size: "md" },
});

/**
 * Class list for the button variants, for call sites that must render a real `<a>`/`<Link>`.
 */
export function buttonClass(
  variant: ButtonVariant = "primary",
  { small = false, className = "" }: { small?: boolean; className?: string } = {},
): string {
  return cn(buttonVariants({ variant, size: small ? "sm" : "md" }), className);
}

/** Centered over the (invisible) label so the button never changes width. */
export function ButtonSpinner() {
  return (
    <span
      aria-hidden="true"
      data-button-spinner=""
      className="pointer-events-none absolute inset-0 flex items-center justify-center"
    >
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent motion-reduce:animate-none" />
    </span>
  );
}

/** Non-colour error cue: an exclamation glyph. */
export function ButtonErrorMark({ className }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      data-button-error=""
      viewBox="0 0 16 16"
      className={cn("size-3.5 shrink-0", className)}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    >
      <circle cx="8" cy="8" r="6.5" />
      <path d="M8 4.5v4M8 11v.01" />
    </svg>
  );
}

export function isInvalid(value: React.AriaAttributes["aria-invalid"]): boolean {
  return value === true || value === "true";
}

type ButtonOwnProps = {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** @deprecated use `size="sm"`. */
  small?: boolean;
  loading?: boolean;
  /** Render the single child element (e.g. `<a>`) with button styling instead of a `<button>`. */
  asChild?: boolean;
};

export type ButtonProps = React.ComponentPropsWithoutRef<"button"> &
  ButtonOwnProps &
  Omit<VariantPropsOf<typeof buttonVariants>, "variant" | "size">;

/** Swallows activation while loading. Click covers Enter and Space, which browsers map to click. */
export function guardClick<E extends React.MouseEvent>(
  loading: boolean,
  onClick: ((event: E) => void) | undefined,
) {
  return (event: E) => {
    if (loading) {
      event.preventDefault();
      return;
    }
    onClick?.(event);
  };
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant,
    size,
    small = false,
    loading = false,
    asChild = false,
    className,
    children,
    type,
    onClick,
    ...rest
  },
  ref,
) {
  const classes = cn(buttonVariants({ variant, size: size ?? (small ? "sm" : "md") }), className);
  const invalid = isInvalid(rest["aria-invalid"]);
  const showError = invalid && !loading;

  const decorate = (content: React.ReactNode) => (
    <>
      {showError ? <ButtonErrorMark /> : null}
      <span data-button-label="" className={loading ? "opacity-0" : undefined}>
        {content}
      </span>
      {loading ? <ButtonSpinner /> : null}
    </>
  );

  const common = {
    className: classes,
    "aria-busy": loading ? (true as const) : undefined,
    onClick: guardClick(loading, onClick),
    ...rest,
  };

  if (asChild && React.isValidElement<{ children?: React.ReactNode }>(children)) {
    return (
      <Slot ref={ref} {...common}>
        {React.cloneElement(children, undefined, decorate(children.props.children))}
      </Slot>
    );
  }

  return (
    <button ref={ref} type={type ?? "button"} {...common}>
      {decorate(children)}
    </button>
  );
});
