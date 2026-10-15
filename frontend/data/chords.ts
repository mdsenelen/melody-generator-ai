// Root × quality data for the Compose page's chord picker. Output is always
// a plain chord-name string (e.g. "C#m7") -- the same shape `ChordGraph` and
// `ChordDiagram` already consume, so this only changes how a chord name is
// picked, not what gets sent anywhere.

export type ChordQuality =
  "major" | "minor" | "dim" | "aug" | "power" | "7" | "maj7" | "min7" | "dim7" | "sus2" | "sus4";

export const ROOTS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];

export const CHORD_QUALITIES: { quality: ChordQuality; label: string; category: string }[] = [
  { quality: "major", label: "Major", category: "Basic" },
  { quality: "minor", label: "Minor", category: "Basic" },
  { quality: "power", label: "Power (5)", category: "Basic" },
  { quality: "dim", label: "Diminished", category: "Basic" },
  { quality: "aug", label: "Augmented", category: "Basic" },
  { quality: "7", label: "Dominant 7th", category: "Sevenths" },
  { quality: "maj7", label: "Major 7th", category: "Sevenths" },
  { quality: "min7", label: "Minor 7th", category: "Sevenths" },
  { quality: "dim7", label: "Diminished 7th", category: "Sevenths" },
  { quality: "sus2", label: "Sus2", category: "Suspended" },
  { quality: "sus4", label: "Sus4", category: "Suspended" },
];

const SUFFIXES: Record<ChordQuality, string> = {
  major: "",
  minor: "m",
  dim: "dim",
  aug: "aug",
  power: "5",
  "7": "7",
  maj7: "maj7",
  min7: "m7",
  dim7: "dim7",
  sus2: "sus2",
  sus4: "sus4",
};

export function formatChordName(root: string, quality: ChordQuality): string {
  return root + SUFFIXES[quality];
}

const ROOT_PITCH_CLASS: Record<string, number> = {
  C: 0,
  "C#": 1,
  Db: 1,
  D: 2,
  "D#": 3,
  Eb: 3,
  E: 4,
  F: 5,
  "F#": 6,
  Gb: 6,
  G: 7,
  "G#": 8,
  Ab: 8,
  A: 9,
  "A#": 10,
  Bb: 10,
  B: 11,
};

// Intervals above the root, keyed by the suffix `formatChordName` writes.
const SUFFIX_INTERVALS: Record<string, number[]> = {
  "": [0, 4, 7],
  m: [0, 3, 7],
  dim: [0, 3, 6],
  aug: [0, 4, 8],
  "5": [0, 7],
  "7": [0, 4, 7, 10],
  maj7: [0, 4, 7, 11],
  m7: [0, 3, 7, 10],
  dim7: [0, 3, 6, 9],
  sus2: [0, 2, 7],
  sus4: [0, 5, 7],
};

/** Pitch classes (0 = C) of a chord name, root first; [] when the name isn't one we spell. */
export function chordPitchClasses(chord: string): number[] {
  const match = /^([A-G][#b]?)(.*)$/.exec(chord);
  if (!match) return [];
  const root = ROOT_PITCH_CLASS[match[1]];
  const intervals = SUFFIX_INTERVALS[match[2]];
  if (root === undefined || !intervals) return [];
  return intervals.map((interval) => (root + interval) % 12);
}
