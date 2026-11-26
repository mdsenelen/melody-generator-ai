// Enharmonic chord naming. Db and C# are the same chord on the instrument, so
// the label shows the more usual spelling plainly and the other one in italics
// (see components/ui/chord-name.tsx).
//
// "More usual" follows the key-signature convention, not usage counts (none
// exist): a chord is spelled the way the key it most often lives in spells it.
//   major-type: Db (5 flats, vs C# major's 7 sharps), Eb, Ab, Bb; F# (6 sharps
//               and Gb's 6 flats tie, F# is the guitar-chart default)
//   minor-type: C#m, F#m, G#m (3-5 sharps, vs 7-8 flats); Bbm (5 flats, vs 7
//               sharps); Ebm/D#m is a 6-6 tie, Ebm is the pick.

export type ChordNames = { primary: string; alternate: string | null };

// pitch class -> [major-type spelling, minor-type spelling] and the other name
const PAIRS: Record<string, { major: string; minor: string; other: Record<string, string> }> = {
  "C#": { major: "Db", minor: "C#", other: { Db: "C#", "C#": "Db" } },
  Db: { major: "Db", minor: "C#", other: { Db: "C#", "C#": "Db" } },
  "D#": { major: "Eb", minor: "Eb", other: { Eb: "D#", "D#": "Eb" } },
  Eb: { major: "Eb", minor: "Eb", other: { Eb: "D#", "D#": "Eb" } },
  "F#": { major: "F#", minor: "F#", other: { "F#": "Gb", Gb: "F#" } },
  Gb: { major: "F#", minor: "F#", other: { "F#": "Gb", Gb: "F#" } },
  "G#": { major: "Ab", minor: "G#", other: { Ab: "G#", "G#": "Ab" } },
  Ab: { major: "Ab", minor: "G#", other: { Ab: "G#", "G#": "Ab" } },
  "A#": { major: "Bb", minor: "Bb", other: { Bb: "A#", "A#": "Bb" } },
  Bb: { major: "Bb", minor: "Bb", other: { Bb: "A#", "A#": "Bb" } },
};

function isMinorType(suffix: string) {
  const lower = suffix.toLowerCase();
  return lower.startsWith("m") && !lower.startsWith("maj");
}

export function chordNames(chord: string): ChordNames {
  const match = chord.match(/^([A-G][#b]?)(.*)$/);
  const pair = match ? PAIRS[match[1]] : undefined;
  if (!match || !pair) return { primary: chord, alternate: null };

  const suffix = match[2];
  const root = isMinorType(suffix) ? pair.minor : pair.major;
  return { primary: root + suffix, alternate: pair.other[root] + suffix };
}
