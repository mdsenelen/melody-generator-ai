import type { ReactNode } from "react";

import { cn } from "./cn";

type BadgeVariant = "default" | "accent" | "success" | "warning" | "error" | "muted";

const BADGE_STYLES: Record<BadgeVariant, string> = {
  default: "border-border text-muted-foreground",
  accent: "border-primary/40 text-primary bg-primary/8",
  success: "border-green-500/30 text-green-400 bg-green-500/8",
  warning: "border-amber-500/30 text-amber-400 bg-amber-500/8",
  error: "border-red-500/30 text-red-400 bg-red-500/8",
  muted: "border-border text-border-strong",
};

const DOT_STYLES: Record<BadgeVariant, string> = {
  default: "bg-muted-foreground",
  accent: "bg-primary",
  success: "bg-green-400",
  warning: "bg-amber-400",
  error: "bg-red-400",
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
        "inline-flex items-center gap-1.5 rounded-[var(--radius)] border px-1.5 py-0.5 font-mono text-[9px] tracking-widest uppercase",
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
  happy: "border-amber-500/40 text-amber-300 bg-amber-950/30",
  sad: "border-sky-500/40 text-sky-300 bg-sky-950/30",
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
        "inline-flex items-center gap-1.5 rounded-[var(--radius)] border px-2 py-0.5 font-mono text-[9px] tracking-widest uppercase",
        cls,
      )}
    >
      {emoji ? <span aria-hidden="true">{emoji}</span> : null}
      {label ?? mood}
    </span>
  );
}

const GENRE_COLORS: Record<string, string> = {
  Ambient: "border-emerald-900/50 text-emerald-400",
  Jazz: "border-amber-900/50 text-amber-400",
  Classical: "border-indigo-900/50 text-indigo-400",
  Electronic: "border-primary/40 text-primary/85",
  Folk: "border-orange-900/50 text-orange-400",
  Cinematic: "border-slate-800/60 text-slate-400",
  Pop: "border-primary/40 text-primary/85",
  Blues: "border-sky-900/50 text-sky-400",
  Rock: "border-red-900/50 text-red-400",
  Flamenco: "border-orange-900/50 text-orange-400",
  "J-Pop": "border-fuchsia-900/50 text-fuchsia-400",
  General: "border-border text-muted-foreground",
};

export function GenreBadge({ genre }: { genre: string }) {
  const cls = GENRE_COLORS[genre] ?? GENRE_COLORS.General;
  return (
    <span
      className={cn(
        "rounded-[var(--radius)] border px-2 py-0.5 font-mono text-[9px] tracking-widest uppercase",
        cls,
      )}
    >
      {genre}
    </span>
  );
}
