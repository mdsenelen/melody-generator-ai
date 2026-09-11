import type { ButtonHTMLAttributes } from "react";

import { cn } from "./cn";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "destructive";

const FOCUS_RING =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:ring-offset-1 focus-visible:ring-offset-background";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary:
    "bg-primary text-primary-foreground hover:bg-primary/85 active:bg-primary/70 font-display",
  secondary:
    "bg-primary/12 text-primary border border-primary/30 hover:bg-primary/20 hover:border-primary/50 active:bg-primary/25 font-display",
  ghost:
    "border border-border text-secondary-foreground hover:border-border-strong hover:text-foreground active:bg-card",
  destructive:
    "border border-red-500/30 text-red-400 hover:bg-red-500/8 hover:border-red-500/50 active:bg-red-500/15",
};

/**
 * Shared class list for the four button variants -- for call sites that must
 * render a real `<a>`/`<Link>` (e.g. a download or "view result" link) rather
 * than a `<button>`, so the visual language stays identical either way.
 */
export function buttonClass(
  variant: ButtonVariant = "primary",
  { small = false, className = "" }: { small?: boolean; className?: string } = {},
): string {
  return cn(
    "inline-flex items-center justify-center gap-2 rounded-[var(--radius)] font-medium tracking-wide transition-all duration-150",
    "disabled:cursor-not-allowed disabled:opacity-30",
    FOCUS_RING,
    VARIANT_CLASSES[variant],
    small ? "px-3 py-1.5 text-xs" : "px-4 py-2.5 text-sm",
    className,
  );
}

function ButtonSpinner() {
  return (
    <span
      aria-hidden="true"
      className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border border-current border-t-transparent motion-reduce:animate-none"
    />
  );
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  small?: boolean;
  loading?: boolean;
};

export function Button({
  variant = "primary",
  small = false,
  loading = false,
  disabled,
  className = "",
  children,
  type = "button",
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      className={buttonClass(variant, { small, className })}
      {...rest}
    >
      {loading ? <ButtonSpinner /> : null}
      {children}
    </button>
  );
}
