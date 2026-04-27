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
    <div className="mt-4 rounded-2xl border border-white/10 bg-black/30 p-4 backdrop-blur-sm">
      {/* Header row: current note name + frequency */}
      <div className="mb-3 flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-widest text-white/50">
          Live pitch
        </p>
        <div className="flex items-center gap-2">
          {isPitched ? (
            <>
              {/* Pulsing dot — indicates live signal */}
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
              <span className="text-sm font-bold text-white">
                {currentNote}
              </span>
              <span className="text-xs text-white/50">{currentFrequency} Hz</span>
            </>
          ) : (
            <>
              <span className="inline-block h-2 w-2 rounded-full bg-white/20" />
              <span className="text-xs text-white/40">listening…</span>
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
                    "w-full rounded-t-md transition-all duration-75",
                    isActive
                      ? "animate-pulse bg-gradient-to-t from-purple-500 to-sky-300 shadow-lg shadow-sky-400/40"
                      : normalized > 0
                        ? "bg-gradient-to-t from-purple-700/60 to-sky-600/40"
                        : "bg-white/10",
                  ].join(" ")}
                  style={{ height: `${heightPct}%` }}
                />
              </div>
              {/* Note label below the bar */}
              <span
                className={[
                  "text-[10px] font-semibold leading-none",
                  isActive ? "text-sky-300" : "text-white/40",
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
        <div className="h-1 w-full overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-emerald-400/70 transition-all duration-150"
            style={{ width: `${Math.round(clarity * 100)}%` }}
          />
        </div>
        <p className="mt-1 text-right text-[10px] text-white/30">
          clarity {Math.round(clarity * 100)}%
        </p>
      </div>
    </div>
  );
}
