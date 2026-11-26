// Framework-free geometry for the analysis page's piano roll + waveform.
// Everything here returns plain numbers so it can be unit-tested without a
// canvas; components/note-roll.tsx only paints the results.

import type { NoteEvent } from "./contracts/job";

const DEFAULT_BOUNDS = { low: 48, high: 84 } as const;
const PITCH_PADDING = 2;
const MIN_SPAN_SEMITONES = 12;

export type PitchBounds = { low: number; high: number };

/** Max |amplitude| per bucket -- the envelope drawn as the waveform. */
export function waveformPeaks(samples: Float32Array, buckets: number): Float32Array {
  const peaks = new Float32Array(buckets);
  if (samples.length === 0 || buckets <= 0) return peaks;
  const perBucket = samples.length / buckets;
  for (let b = 0; b < buckets; b += 1) {
    const from = Math.floor(b * perBucket);
    const to = Math.min(samples.length, Math.max(from + 1, Math.floor((b + 1) * perBucket)));
    let max = 0;
    for (let i = from; i < to; i += 1) {
      const v = Math.abs(samples[i]);
      if (v > max) max = v;
    }
    peaks[b] = max;
  }
  return peaks;
}

/** Visible MIDI pitch range: played range +/- 2 semitones, at least an octave. */
export function pitchBounds(notes: NoteEvent[]): PitchBounds {
  if (notes.length === 0) return { ...DEFAULT_BOUNDS };
  let low = Infinity;
  let high = -Infinity;
  for (const note of notes) {
    if (note.pitch < low) low = note.pitch;
    if (note.pitch > high) high = note.pitch;
  }
  low -= PITCH_PADDING;
  high += PITCH_PADDING;
  const missing = MIN_SPAN_SEMITONES - (high - low);
  if (missing > 0) {
    low -= Math.floor(missing / 2);
    high += Math.ceil(missing / 2);
  }
  return { low: Math.max(0, low), high: Math.min(127, high) };
}

export type NoteRect = {
  /** 0..1 of the time axis */
  x: number;
  width: number;
  /** pitch - low, i.e. rows up from the bottom */
  row: number;
  /** total rows (high - low + 1) */
  rows: number;
  /** 0.35..1, from velocity */
  opacity: number;
};

export function layoutNotes(
  notes: NoteEvent[],
  view: PitchBounds & { durationSec: number },
): NoteRect[] {
  const rows = view.high - view.low + 1;
  if (view.durationSec <= 0) return [];
  const rects: NoteRect[] = [];
  for (const note of notes) {
    if (note.pitch < view.low || note.pitch > view.high || note.start >= view.durationSec) {
      continue;
    }
    const end = Math.min(note.end, view.durationSec);
    rects.push({
      x: note.start / view.durationSec,
      width: Math.max(0, end - note.start) / view.durationSec,
      row: note.pitch - view.low,
      rows,
      opacity: 0.35 + 0.65 * Math.min(1, Math.max(0, note.velocity / 127)),
    });
  }
  return rects;
}
