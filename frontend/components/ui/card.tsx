import type { ReactNode } from "react";

import { cn } from "./cn";

type CardVariant = "default" | "accent" | "muted" | "error" | "warning" | "success";

const CARD_STYLES: Record<CardVariant, string> = {
  default: "bg-card border-border",
  accent: "bg-card border-primary/40 shadow-[var(--shadow-panel)]",
  muted: "bg-background border-border",
  error: "bg-danger/8 border-danger/30",
  warning: "bg-warning/8 border-warning/30",
  success: "bg-success/8 border-success/30",
};

export function Card({
  children,
  className = "",
  variant = "default",
}: {
  children: ReactNode;
  className?: string;
  variant?: CardVariant;
}) {
  return (
    <div className={cn("rounded-[var(--radius-lg)] border p-5", CARD_STYLES[variant], className)}>
      {children}
    </div>
  );
}
