/**
 * Card: the bordered surface panels and read-outs sit on. On tokens; the status variants use the
 * tinted-surface and border tokens, so ErrorState and the feedback banners share one definition.
 *
 * Usage
 *   <Card variant="muted" className="space-y-4">...</Card>
 *
 * States: static container. hover / focus-visible / active / disabled / loading do not apply
 * (it is not a control); `error` is the `error` variant. Any div attribute (role, data-*, id)
 * is forwarded.
 */
import type { HTMLAttributes } from "react";

import { cn } from "./cn";
import type { Exhaustive } from "./variants";

export type CardVariant = "default" | "accent" | "muted" | "error" | "warning" | "success";

const CARD_STYLES: Exhaustive<CardVariant> = {
  default: "bg-surface-raised border-border",
  accent: "bg-surface-raised border-accent/20",
  muted: "bg-surface border-border",
  error: "bg-danger-surface border-danger-border",
  warning: "bg-warning-surface border-warning-border",
  success: "bg-success-surface border-success-border",
};

export type CardProps = HTMLAttributes<HTMLDivElement> & {
  variant?: CardVariant;
};

export function Card({ children, className, variant = "default", ...rest }: CardProps) {
  return (
    <div
      {...rest}
      className={cn("rounded-[var(--radius)] border p-4", CARD_STYLES[variant], className)}
    >
      {children}
    </div>
  );
}
