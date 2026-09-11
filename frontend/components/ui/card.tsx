import type { ReactNode } from "react";

import { cn } from "./cn";

type CardVariant = "default" | "accent" | "muted" | "error" | "warning" | "success";

const CARD_STYLES: Record<CardVariant, string> = {
  default: "bg-card border-border",
  accent: "bg-card border-primary/20",
  muted: "bg-background border-border",
  error: "bg-[#120808] border-red-500/20",
  warning: "bg-[#120d04] border-amber-500/20",
  success: "bg-[#040f06] border-green-500/20",
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
    <div className={cn("rounded-[var(--radius)] border p-4", CARD_STYLES[variant], className)}>
      {children}
    </div>
  );
}
