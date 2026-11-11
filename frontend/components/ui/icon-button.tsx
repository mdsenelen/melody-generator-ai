/**
 * IconButton: a square, icon-only Button. Shares Button's variants, states and focus ring.
 *
 * Usage
 *   <IconButton label="Play" icon={<PlayIcon />} onClick={play} />
 *   <IconButton label="Stop" icon={<StopIcon />} variant="ghost" size="sm" loading={busy} />
 *
 * `label` is required and becomes the accessible name (`aria-label`); omitting it, or passing only
 * `aria-label`, is a compile error. The icon is wrapped in an `aria-hidden` span.
 * Loading, disabled and error behave exactly as on Button (see button.tsx). The error state tints
 * the button and adds an "!" badge in the corner.
 */
import * as React from "react";
import { cva } from "class-variance-authority";

import {
  BUTTON_BASE,
  BUTTON_VARIANTS,
  ButtonErrorMark,
  ButtonSpinner,
  guardClick,
  isInvalid,
  type ButtonSize,
  type ButtonVariant,
} from "./button";
import { cn } from "./cn";
import type { Exhaustive } from "./variants";

const ICON_BUTTON_SIZES: Exhaustive<ButtonSize> = {
  sm: "size-7 text-xs",
  md: "size-10 text-sm",
};

const iconButtonVariants = cva(BUTTON_BASE, {
  variants: { variant: BUTTON_VARIANTS, size: ICON_BUTTON_SIZES },
  defaultVariants: { variant: "ghost", size: "md" },
});

export type IconButtonProps = Omit<
  React.ComponentPropsWithoutRef<"button">,
  "children" | "aria-label" | "aria-labelledby"
> & {
  /** Accessible name. Required. */
  label: string;
  icon: React.ReactNode;
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
};

export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, variant, size, loading = false, className, type, onClick, ...rest },
  ref,
) {
  const showError = isInvalid(rest["aria-invalid"]) && !loading;
  return (
    <button
      ref={ref}
      type={type ?? "button"}
      aria-label={label}
      aria-busy={loading ? true : undefined}
      onClick={guardClick(loading, onClick)}
      className={cn(iconButtonVariants({ variant, size }), className)}
      {...rest}
    >
      <span aria-hidden="true" className={cn("inline-flex", loading && "opacity-0")}>
        {icon}
      </span>
      {loading ? <ButtonSpinner /> : null}
      {showError ? (
        <ButtonErrorMark className="bg-surface text-danger absolute -top-1 -right-1 size-3 rounded-full" />
      ) : null}
    </button>
  );
});
