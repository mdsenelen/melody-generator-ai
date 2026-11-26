"use client";

import { useEffect, useMemo, useRef } from "react";

import type { NoteEvent } from "../src/domain/contracts/job";
import { layoutNotes, pitchBounds } from "../src/domain/noteRoll";

const ROLL_HEIGHT = 220;
const WAVE_HEIGHT = 64;
const KEY_WIDTH = 36;

const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

type Props = {
  notes: NoteEvent[];
  durationSec: number;
  /** Waveform envelope (0..1 per bucket); null while it is still decoding. */
  peaks: Float32Array | null;
  /** The analysis window, shaded on both lanes. */
  window?: { start: number; end: number };
};

function cssVar(name: string, fallback: string) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/**
 * Basic Pitch's notes as a piano roll with the audio waveform underneath
 * (the layout of basicpitch.spotify.com). Pure paint: geometry comes from
 * src/domain/noteRoll.ts.
 */
export function NoteRoll({ notes, durationSec, peaks, window: clip }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bounds = useMemo(() => pitchBounds(notes), [notes]);
  const rects = useMemo(
    () => layoutNotes(notes, { ...bounds, durationSec }),
    [notes, bounds, durationSec],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const paint = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = globalThis.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = ROLL_HEIGHT + WAVE_HEIGHT;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const primary = cssVar("--color-primary", "#ac20e8");
      const border = cssVar("--color-border", "#1e1e22");
      const card = cssVar("--color-card", "#111114");
      const fg = cssVar("--color-foreground", "#f0ede8");
      const plotX = KEY_WIDTH;
      const plotW = Math.max(1, width - KEY_WIDTH);
      const rows = bounds.high - bounds.low + 1;
      const rowH = ROLL_HEIGHT / rows;

      ctx.clearRect(0, 0, width, height);

      // Piano keys + row stripes (black-key rows are darker, like the reference).
      for (let r = 0; r < rows; r += 1) {
        const pitch = bounds.high - r;
        const black = BLACK_KEYS.has(pitch % 12);
        const y = r * rowH;
        ctx.fillStyle = black ? "#050506" : "#0d0d10";
        ctx.fillRect(plotX, y, plotW, rowH);
        ctx.fillStyle = black ? "#2a2a2f" : "#d8d4cc";
        ctx.fillRect(0, y, KEY_WIDTH - 1, rowH - 0.5);
        if (pitch % 12 === 0) {
          ctx.fillStyle = "#0a0a0b";
          ctx.font = "9px monospace";
          ctx.fillText(`${NOTE_NAMES[0]}${Math.floor(pitch / 12) - 1}`, 4, y + rowH - 2);
        }
      }

      // Analysis window shading, behind the notes.
      if (clip && durationSec > 0) {
        const x0 = plotX + (clip.start / durationSec) * plotW;
        const x1 = plotX + (Math.min(clip.end, durationSec) / durationSec) * plotW;
        ctx.fillStyle = `${primary}22`;
        ctx.fillRect(x0, 0, x1 - x0, height);
        ctx.fillStyle = `${primary}aa`;
        ctx.fillRect(x0, 0, 1, height);
        ctx.fillRect(x1 - 1, 0, 1, height);
      }

      // Notes.
      ctx.fillStyle = primary;
      for (const rect of rects) {
        const y = (rect.rows - 1 - rect.row) * rowH;
        ctx.globalAlpha = rect.opacity;
        ctx.fillRect(
          plotX + rect.x * plotW,
          y + 0.5,
          Math.max(2, rect.width * plotW),
          Math.max(1, rowH - 1),
        );
      }
      ctx.globalAlpha = 1;

      // Waveform lane.
      ctx.fillStyle = card;
      ctx.fillRect(0, ROLL_HEIGHT, width, WAVE_HEIGHT);
      ctx.fillStyle = border;
      ctx.fillRect(0, ROLL_HEIGHT, width, 1);
      const mid = ROLL_HEIGHT + WAVE_HEIGHT / 2;
      if (peaks && peaks.length > 0) {
        ctx.fillStyle = fg;
        ctx.globalAlpha = 0.75;
        const cols = Math.floor(plotW);
        for (let x = 0; x < cols; x += 1) {
          const peak = peaks[Math.min(peaks.length - 1, Math.floor((x / cols) * peaks.length))];
          const h = Math.max(1, peak * (WAVE_HEIGHT - 6));
          ctx.fillRect(plotX + x, mid - h / 2, 1, h);
        }
        ctx.globalAlpha = 1;
      } else {
        ctx.fillStyle = border;
        ctx.fillRect(plotX, mid, plotW, 1);
      }
    };

    paint();
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => paint());
    observer?.observe(canvas);
    return () => observer?.disconnect();
  }, [bounds, rects, peaks, clip, durationSec]);

  return (
    <div className="border-border bg-background overflow-hidden rounded-[var(--radius)] border">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Piano roll of ${notes.length} detected notes with the audio waveform below`}
        className="block w-full"
        style={{ height: ROLL_HEIGHT + WAVE_HEIGHT }}
      />
    </div>
  );
}
