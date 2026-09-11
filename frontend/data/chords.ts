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
