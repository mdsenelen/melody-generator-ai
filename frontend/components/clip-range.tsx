"use client";

import { useId, useState } from "react";

export type ClipWindow = { start: number; end: number };

const MIN_SPAN_SEC = 1;

function formatTime(seconds: number) {
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

type Props = {
  /** Full source clip length; the range is [0, sourceDurationSec]. */
  sourceDurationSec: number;
  /** Committed window (what the shown analysis was computed for). */
  value: ClipWindow;
  /** Fires when the user commits a new window (button / Enter). */
  onCommit: (window: ClipWindow) => void;
  /** True while an analysis request for this component is in flight. */
  busy?: boolean;
};

/**
 * Two native range sliders picking the [start, end] window analysis runs on.
 * No waveform (deferred to roadmap phase 7) -- just the numbers, keyboard
 * operable, with a live region announcing the pending selection.
 */
export function ClipRange({ sourceDurationSec, value, onCommit, busy = false }: Props) {
  const max = Math.max(MIN_SPAN_SEC, Math.floor(sourceDurationSec));
  const [draft, setDraft] = useState<ClipWindow>(value);
  const startId = useId();
  const endId = useId();

  // Re-sync the draft when the committed value changes from outside (e.g. a
  // fresh transcription resets it to [0, 60]).
  const [lastValue, setLastValue] = useState(value);
  if (lastValue.start !== value.start || lastValue.end !== value.end) {
    setLastValue(value);
    setDraft(value);
  }

  const setStart = (next: number) => {
    const start = Math.min(Math.max(0, next), max - MIN_SPAN_SEC);
    setDraft((d) => ({ start, end: Math.max(d.end, start + MIN_SPAN_SEC) }));
  };
  const setEnd = (next: number) => {
    const end = Math.min(Math.max(MIN_SPAN_SEC, next), max);
    setDraft((d) => ({ end, start: Math.min(d.start, end - MIN_SPAN_SEC) }));
  };

  const dirty = draft.start !== value.start || draft.end !== value.end;
  const spanLabel = `${formatTime(draft.start)}–${formatTime(draft.end)}`;

  const commit = () => {
    if (dirty && !busy) onCommit(draft);
  };

  return (
    <form
      className="space-y-3"
      onSubmit={(event) => {
        event.preventDefault();
        commit();
      }}
    >
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs tracking-[0.2em] text-white/45 uppercase">Analysis window</p>
        <p className="text-sm font-semibold text-white" aria-hidden="true">
          {spanLabel}
        </p>
      </div>

      <label htmlFor={startId} className="block text-xs text-white/55">
        Start
      </label>
      <input
        id={startId}
        type="range"
        min={0}
        max={max - MIN_SPAN_SEC}
        step={1}
        value={draft.start}
        disabled={busy}
        onChange={(event) => setStart(Number(event.target.value))}
        aria-label="Analysis window start"
        aria-valuetext={formatTime(draft.start)}
        className="w-full accent-[#8b5cf6]"
      />

      <label htmlFor={endId} className="block text-xs text-white/55">
        End
      </label>
      <input
        id={endId}
        type="range"
        min={MIN_SPAN_SEC}
        max={max}
        step={1}
        value={draft.end}
        disabled={busy}
        onChange={(event) => setEnd(Number(event.target.value))}
        aria-label="Analysis window end"
        aria-valuetext={formatTime(draft.end)}
        className="w-full accent-[#8b5cf6]"
      />

      <p aria-live="polite" className="sr-only">
        {dirty
          ? `Pending analysis window ${spanLabel}. Press analyse this section to apply.`
          : `Analysis window ${spanLabel}.`}
      </p>

      <button
        type="submit"
        disabled={!dirty || busy}
        className="rounded-full border border-[#8b5cf6]/50 bg-[rgba(139,92,246,0.12)] px-4 py-2 text-sm font-semibold text-[#f1e9ff] transition hover:border-[#b18aff] hover:bg-[rgba(139,92,246,0.2)] disabled:cursor-not-allowed disabled:opacity-40"
      >
        {busy ? "Analysing…" : "Analyse this section"}
      </button>
    </form>
  );
}
