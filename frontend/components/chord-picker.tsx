"use client";

import { useEffect, useRef, useState } from "react";

import { CHORD_QUALITIES, ROOTS, formatChordName, type ChordQuality } from "../data/chords";
import { cn } from "./ui/cn";

type ChordPickerProps = {
  /** The current chord name (e.g. "Am7"), or null when the slot is empty. */
  value: string | null;
  onChange: (chord: string | null) => void;
  placeholder: string;
};

/** Root x quality picker that emits a plain chord-name string, same shape as the old <select>. */
export function ChordPicker({ value, onChange, placeholder }: ChordPickerProps) {
  const [open, setOpen] = useState(false);
  const [root, setRoot] = useState("C");
  const [quality, setQuality] = useState<ChordQuality>("major");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const pickQuality = (nextQuality: ChordQuality) => {
    setQuality(nextQuality);
    onChange(formatChordName(root, nextQuality));
    setOpen(false);
  };

  const clear = (event: React.MouseEvent) => {
    event.stopPropagation();
    onChange(null);
  };

  const categories = [...new Set(CHORD_QUALITIES.map((q) => q.category))];

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className={cn(
          "flex w-full items-center justify-between rounded-[var(--radius)] border px-3 py-2.5 text-sm transition-colors",
          value
            ? "border-primary/40 bg-card text-foreground"
            : "border-border bg-background text-muted-foreground hover:border-border-strong",
        )}
      >
        <span className={value ? "font-display" : "font-mono"}>{value ?? placeholder}</span>
        <span className="flex items-center gap-1.5">
          {value ? (
            <span
              role="button"
              tabIndex={0}
              aria-label="Clear chord"
              onClick={clear}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onChange(null);
                }
              }}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
                <path
                  d="M1 1l8 8M9 1L1 9"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            </span>
          ) : null}
          <svg
            width="10"
            height="10"
            viewBox="0 0 10 10"
            fill="none"
            aria-hidden="true"
            className={cn("text-muted-foreground transition-transform", open && "rotate-180")}
          >
            <path
              d="M2 3.5l4 4 4-4"
              stroke="currentColor"
              strokeWidth="1.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>

      {open ? (
        <div className="border-border bg-card absolute top-full left-0 z-30 mt-1 w-64 rounded-[var(--radius)] border shadow-[0_16px_40px_rgba(0,0,0,0.6)]">
          <div className="flex max-h-72">
            <div className="border-border w-16 shrink-0 overflow-y-auto border-r p-1.5">
              <p className="text-muted-foreground mb-1 px-1 font-mono text-[9px] tracking-widest uppercase">
                Root
              </p>
              {ROOTS.map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setRoot(r)}
                  className={cn(
                    "w-full rounded-[var(--radius)] px-2 py-1 text-left font-mono text-xs transition-colors",
                    root === r
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {r}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto p-1.5">
              {categories.map((category) => (
                <div key={category} className="mb-2">
                  <p className="text-muted-foreground mb-0.5 px-1 font-mono text-[9px] tracking-widest uppercase">
                    {category}
                  </p>
                  {CHORD_QUALITIES.filter((q) => q.category === category).map((q) => (
                    <button
                      key={q.quality}
                      type="button"
                      onClick={() => pickQuality(q.quality)}
                      aria-label={q.label}
                      className="text-muted-foreground hover:text-foreground w-full rounded-[var(--radius)] px-2 py-1 text-left text-xs transition-colors"
                    >
                      <span aria-hidden="true">{q.label}</span>
                      <span aria-hidden="true" className="text-muted-foreground ml-1.5 font-mono">
                        {formatChordName(root, q.quality)}
                      </span>
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
