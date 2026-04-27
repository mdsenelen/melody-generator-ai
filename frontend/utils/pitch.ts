// Chromatic note names starting at C (pitch class 0)
export const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;

export type NoteName = (typeof NOTE_NAMES)[number];

/**
 * Convert a frequency in Hz to the nearest MIDI note number.
 * Formula: MIDI = 69 + 12 * log2(freq / 440)
 * A4 = 440 Hz = MIDI 69 is the anchor point.
 */
export function frequencyToMidi(freq: number): number {
  return Math.round(69 + 12 * Math.log2(freq / 440));
}

/**
 * Convert a MIDI note number to a pitch class index (0 = C, 11 = B).
 * Uses modulo 12 so it works across all octaves.
 */
export function midiToPitchClass(midi: number): number {
  return ((midi % 12) + 12) % 12; // double-mod keeps negatives positive
}

/**
 * Convert a MIDI note number to a human-readable name like "A4" or "C#3".
 * Octave number = floor(midi / 12) - 1  (MIDI 60 = C4, middle C).
 */
export function midiToNoteName(midi: number): string {
  const pitchClass = midiToPitchClass(midi);
  const octave = Math.floor(midi / 12) - 1;
  return `${NOTE_NAMES[pitchClass]}${octave}`;
}

/**
 * Check whether a frequency is in the range a human voice or instrument
 * can realistically produce (C2 = ~65 Hz to C7 = ~2093 Hz).
 */
export function isInPlayableRange(freq: number): boolean {
  return freq >= 65 && freq <= 2100;
}

// Minimum Pitchy clarity score to treat a reading as real pitched audio (not noise/breath).
// Shared by the analyzer hook and the histogram display so both stay in sync.
export const CLARITY_THRESHOLD = 0.85;
