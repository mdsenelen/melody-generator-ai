"use client";

import { NOTE_NAMES, CLARITY_THRESHOLD } from "../utils/pitch";
import type { AudioAnalyzerState } from "../hooks/use-audio-analyzer";

type LivePitchHistogramProps = Pick<
  AudioAnalyzerState,
  "noteHistogram" | "pitchClass" | "currentNote" | "currentFrequency" | "clarity"
>;

export function LivePitchHistogram({
  noteHistogram,
  pitchClass,
  currentNote,
  currentFrequency,
  clarity,
}: LivePitchHistogramProps) {
  const maxCount = Math.max(...noteHistogram, 1); // avoid divide-by-zero

  const isPitched = clarity >= CLARITY_THRESHOLD && currentNote !== null;

  return (
    <div className="border-border bg-background mt-4 rounded-[var(--radius)] border p-4">
      {/* Header row: current note name + frequency */}
      <div className="mb-3 flex items-center justify-between">
        <p className="text-muted-foreground font-mono text-xs tracking-widest uppercase">
          Live pitch
        </p>
        <div className="flex items-center gap-2">
          {isPitched ? (
            <>
              {/* Pulsing dot — indicates live signal */}
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
              <span className="text-foreground text-sm font-semibold">{currentNote}</span>
              <span className="text-muted-foreground text-xs">{currentFrequency} Hz</span>
            </>
          ) : (
            <>
              <span className="bg-border-strong inline-block h-2 w-2 rounded-full" />
              <span className="text-muted-foreground text-xs">listening…</span>
            </>
          )}
        </div>
      </div>

      {/* 12-bar histogram — one bar per pitch class */}
      <div className="flex items-end justify-between gap-1" style={{ height: "72px" }}>
        {NOTE_NAMES.map((name, i) => {
          const isActive = pitchClass === i && isPitched;
          const normalized = noteHistogram[i] / maxCount; // 0–1
          // Active bar gets full height; inactive bars scale by accumulated count
          const heightPct = isActive ? 100 : Math.max(normalized * 100, 8);

          return (
            <div key={name} className="flex flex-1 flex-col items-center gap-1">
              {/* The bar itself */}
              <div className="relative flex w-full items-end" style={{ height: "56px" }}>
                <div
                  className={[
                    "w-full rounded-t-[2px] transition-all duration-75",
                    isActive
                      ? "bg-primary animate-pulse shadow-[0_0_12px_rgba(172,32,232,0.5)]"
                      : normalized > 0
                        ? "bg-primary/40"
                        : "bg-border",
                  ].join(" ")}
                  style={{ height: `${heightPct}%` }}
                />
              </div>
              {/* Note label below the bar */}
              <span
                className={[
                  "text-[10px] leading-none font-semibold",
                  isActive ? "text-primary" : "text-muted-foreground",
                ].join(" ")}
              >
                {name}
              </span>
            </div>
          );
        })}
      </div>

      {/* Clarity meter — a thin strip showing signal confidence */}
      <div className="mt-3">
        <div className="bg-border h-px w-full overflow-hidden">
          <div
            className="h-full bg-emerald-400/70 transition-all duration-150"
            style={{ width: `${Math.round(clarity * 100)}%` }}
          />
        </div>
        <p className="text-muted-foreground mt-1 text-right font-mono text-[10px]">
          clarity {Math.round(clarity * 100)}%
        </p>
      </div>
    </div>
  );
}
