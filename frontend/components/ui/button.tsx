import type { ButtonHTMLAttributes } from "react";

import { cn } from "./cn";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "destructive";

const FOCUS_RING =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary:
    "bg-primary text-primary-foreground hover:bg-primary-bright active:bg-primary-bright shadow-[0_1px_0_rgba(255,255,255,0.18)_inset]",
  secondary:
    "bg-primary-deep text-primary-bright border border-primary/40 hover:border-primary active:bg-primary-deep/80",
  ghost: "border border-border-strong text-foreground hover:bg-raised active:bg-raised-hover",
  destructive:
    "border border-danger/50 text-danger hover:bg-danger/10 hover:border-danger active:bg-danger/15",
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
    "inline-flex items-center justify-center gap-2 rounded-[var(--radius)] font-sans font-medium transition-colors duration-150",
    "disabled:cursor-not-allowed disabled:opacity-40",
    FOCUS_RING,
    VARIANT_CLASSES[variant],
    small ? "h-8 px-3 text-xs" : "h-11 px-5 text-sm",
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
