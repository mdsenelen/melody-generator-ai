import type { ReactNode } from "react";

import { cn } from "./cn";

type BadgeVariant = "default" | "accent" | "success" | "warning" | "error" | "muted";

const BADGE_STYLES: Record<BadgeVariant, string> = {
  default: "border-border text-muted-foreground",
  accent: "border-primary/50 text-primary-bright bg-primary-deep/60",
  success: "border-success/40 text-success bg-success/8",
  warning: "border-warning/40 text-warning bg-warning/8",
  error: "border-danger/40 text-danger bg-danger/8",
  muted: "border-border text-muted-foreground",
};

const DOT_STYLES: Record<BadgeVariant, string> = {
  default: "bg-muted-foreground",
  accent: "bg-primary",
  success: "bg-success",
  warning: "bg-warning",
  error: "bg-danger",
  muted: "bg-border-strong",
};

export function StatusBadge({
  children,
  variant = "default",
  dot = false,
}: {
  children: ReactNode;
  variant?: BadgeVariant;
  dot?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[10px] tracking-[0.14em] uppercase",
        BADGE_STYLES[variant],
      )}
    >
      {dot ? (
        <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", DOT_STYLES[variant])} />
      ) : null}
      {children}
    </span>
  );
}

export function PlannedBadge({ label = "Planned" }: { label?: string }) {
  return <StatusBadge variant="muted">{label}</StatusBadge>;
}

const MOOD_COLORS: Record<string, string> = {
  happy: "border-warning/40 text-warning bg-warning/8",
  sad: "border-primary/50 text-primary-bright bg-primary-deep/60",
  neutral: "border-border text-muted-foreground bg-muted",
};

const MOOD_EMOJI: Record<string, string> = {
  happy: "😄",
  sad: "😢",
  neutral: "😐",
};

export function MoodBadge({ mood, label }: { mood: string; label?: string }) {
  const key = mood.toLowerCase();
  const cls = MOOD_COLORS[key] ?? BADGE_STYLES.default;
  const emoji = MOOD_EMOJI[key];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-[10px] tracking-[0.14em] uppercase",
        cls,
      )}
    >
      {emoji ? <span aria-hidden="true">{emoji}</span> : null}
      {label ?? mood}
    </span>
  );
}

// Genres are labels, not states: one neutral treatment keeps the steel accent meaningful.
const GENRE_CLASS = "border-border-strong text-secondary-foreground";

export function GenreBadge({ genre }: { genre: string }) {
  return (
    <span
      className={cn(
        "rounded-full border px-2.5 py-0.5 font-mono text-[10px] tracking-[0.14em] uppercase",
        GENRE_CLASS,
      )}
    >
      {genre}
    </span>
  );
}
