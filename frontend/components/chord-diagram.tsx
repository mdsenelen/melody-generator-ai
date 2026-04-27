"use client";

type FretDot = {
  string: number;
  fret: number;
  finger: number;
};

type Barre = {
  fret: number;
  fromString: number;
  toString: number;
  finger: number;
};

type ChordShape = {
  label: string;
  positions: FretDot[];
  muted?: number[];
  open?: number[];
  barre?: Barre;
  baseFret?: number;
};

const ENHARMONIC_ROOTS: Record<string, string> = {
  "C#": "Db", "D#": "Eb", "G#": "Ab", "A#": "Bb", "Gb": "F#",
};

function resolveChordShape(chord: string): ChordShape | undefined {
  if (CHORD_SHAPES[chord]) return CHORD_SHAPES[chord];
  const rootMatch = chord.match(/^([A-G][#b]?)/);
  if (!rootMatch) return undefined;
  const root = rootMatch[1];
  const quality = chord.slice(root.length) === "maj" ? "" : chord.slice(root.length);
  if (CHORD_SHAPES[root + quality]) return CHORD_SHAPES[root + quality];
  const altRoot = ENHARMONIC_ROOTS[root];
  if (altRoot && CHORD_SHAPES[altRoot + quality]) return CHORD_SHAPES[altRoot + quality];
  return undefined;
}

const CHORD_SHAPES: Record<string, ChordShape> = {
  // ── Triads ──────────────────────────────────────────────────────────────────
  C:   { label: "C",   positions: [{ string: 5, fret: 3, finger: 3 }, { string: 4, fret: 2, finger: 2 }, { string: 2, fret: 1, finger: 1 }], muted: [6], open: [3, 1] },
  Cm:  { label: "Cm",  positions: [{ string: 5, fret: 3, finger: 3 }, { string: 4, fret: 5, finger: 4 }, { string: 3, fret: 5, finger: 4 }, { string: 2, fret: 4, finger: 2 }, { string: 1, fret: 3, finger: 1 }], muted: [6], barre: { fret: 3, fromString: 1, toString: 5, finger: 1 }, baseFret: 3 },
  D:   { label: "D",   positions: [{ string: 3, fret: 2, finger: 1 }, { string: 2, fret: 3, finger: 3 }, { string: 1, fret: 2, finger: 2 }], muted: [6, 5], open: [4] },
  Dm:  { label: "Dm",  positions: [{ string: 3, fret: 2, finger: 2 }, { string: 2, fret: 3, finger: 3 }, { string: 1, fret: 1, finger: 1 }], muted: [6, 5], open: [4] },
  E:   { label: "E",   positions: [{ string: 5, fret: 2, finger: 2 }, { string: 4, fret: 2, finger: 3 }, { string: 3, fret: 1, finger: 1 }], open: [6, 2, 1] },
  Em:  { label: "Em",  positions: [{ string: 5, fret: 2, finger: 2 }, { string: 4, fret: 2, finger: 3 }], open: [6, 3, 2, 1] },
  F:   { label: "F",   positions: [{ string: 4, fret: 3, finger: 3 }, { string: 3, fret: 2, finger: 2 }], barre: { fret: 1, fromString: 1, toString: 6, finger: 1 }, baseFret: 1 },
  G:   { label: "G",   positions: [{ string: 6, fret: 3, finger: 2 }, { string: 5, fret: 2, finger: 1 }, { string: 1, fret: 3, finger: 3 }], open: [4, 3, 2] },
  Am:  { label: "Am",  positions: [{ string: 4, fret: 2, finger: 2 }, { string: 3, fret: 2, finger: 3 }, { string: 2, fret: 1, finger: 1 }], muted: [6], open: [5, 1] },
  Bm:  { label: "Bm",  positions: [{ string: 4, fret: 4, finger: 3 }, { string: 3, fret: 4, finger: 4 }, { string: 2, fret: 3, finger: 2 }], muted: [6], barre: { fret: 2, fromString: 1, toString: 5, finger: 1 }, baseFret: 2 },
  A:   { label: "A",   positions: [{ string: 4, fret: 2, finger: 1 }, { string: 3, fret: 2, finger: 2 }, { string: 2, fret: 2, finger: 3 }], muted: [6], open: [5, 1] },
  B:   { label: "B",   positions: [{ string: 4, fret: 4, finger: 4 }, { string: 3, fret: 4, finger: 3 }, { string: 2, fret: 4, finger: 2 }], muted: [6], barre: { fret: 2, fromString: 1, toString: 5, finger: 1 }, baseFret: 2 },
  Bb:  { label: "Bb",  positions: [{ string: 4, fret: 3, finger: 3 }, { string: 3, fret: 3, finger: 4 }, { string: 2, fret: 3, finger: 2 }], muted: [6], barre: { fret: 1, fromString: 1, toString: 5, finger: 1 }, baseFret: 1 },
  Bbm: { label: "Bbm", positions: [{ string: 4, fret: 3, finger: 3 }, { string: 3, fret: 3, finger: 4 }, { string: 2, fret: 2, finger: 2 }], muted: [6], barre: { fret: 1, fromString: 1, toString: 5, finger: 1 }, baseFret: 1 },
  Fm:  { label: "Fm",  positions: [{ string: 5, fret: 3, finger: 3 }, { string: 4, fret: 3, finger: 4 }], barre: { fret: 1, fromString: 1, toString: 6, finger: 1 }, baseFret: 1 },
  Gm:  { label: "Gm",  positions: [{ string: 5, fret: 5, finger: 3 }, { string: 4, fret: 5, finger: 4 }], muted: [6], barre: { fret: 3, fromString: 1, toString: 5, finger: 1 }, baseFret: 3 },
  "F#": { label: "F#", positions: [{ string: 4, fret: 4, finger: 3 }, { string: 3, fret: 4, finger: 4 }], barre: { fret: 2, fromString: 1, toString: 6, finger: 1 }, baseFret: 2 },
  "F#m": { label: "F#m", positions: [{ string: 5, fret: 4, finger: 3 }, { string: 4, fret: 4, finger: 4 }], barre: { fret: 2, fromString: 1, toString: 6, finger: 1 }, baseFret: 2 },
  Ab:  { label: "Ab",  positions: [{ string: 5, fret: 6, finger: 3 }, { string: 4, fret: 6, finger: 4 }, { string: 3, fret: 5, finger: 2 }], barre: { fret: 4, fromString: 1, toString: 6, finger: 1 }, baseFret: 4 },
  Abm: { label: "Abm", positions: [{ string: 5, fret: 6, finger: 3 }, { string: 4, fret: 6, finger: 4 }], barre: { fret: 4, fromString: 1, toString: 6, finger: 1 }, baseFret: 4 },
  Db:  { label: "Db",  positions: [{ string: 4, fret: 6, finger: 4 }, { string: 3, fret: 6, finger: 3 }, { string: 2, fret: 6, finger: 2 }], muted: [6], barre: { fret: 4, fromString: 1, toString: 5, finger: 1 }, baseFret: 4 },
  Dbm: { label: "Dbm", positions: [{ string: 4, fret: 6, finger: 3 }, { string: 3, fret: 6, finger: 4 }], muted: [6], barre: { fret: 4, fromString: 1, toString: 5, finger: 1 }, baseFret: 4 },
  Eb:  { label: "Eb",  positions: [{ string: 4, fret: 5, finger: 3 }, { string: 3, fret: 5, finger: 4 }], barre: { fret: 3, fromString: 1, toString: 6, finger: 1 }, baseFret: 3 },
  Ebm: { label: "Ebm", positions: [{ string: 5, fret: 8, finger: 3 }, { string: 4, fret: 8, finger: 4 }], barre: { fret: 6, fromString: 1, toString: 6, finger: 1 }, baseFret: 6 },
  // ── Dominant 7ths ─────────────────────────────────────────────────────────
  G7:    { label: "G7",    positions: [{ string: 6, fret: 3, finger: 2 }, { string: 5, fret: 2, finger: 1 }, { string: 1, fret: 1, finger: 1 }], open: [4, 3, 2] },
  E7:    { label: "E7",    positions: [{ string: 5, fret: 2, finger: 2 }, { string: 3, fret: 1, finger: 1 }], open: [6, 4, 2, 1] },
  A7:    { label: "A7",    positions: [{ string: 4, fret: 2, finger: 2 }, { string: 2, fret: 2, finger: 3 }], muted: [6], open: [5, 3, 1] },
  D7:    { label: "D7",    positions: [{ string: 3, fret: 2, finger: 2 }, { string: 2, fret: 1, finger: 1 }, { string: 1, fret: 2, finger: 3 }], muted: [6, 5], open: [4] },
  B7:    { label: "B7",    positions: [{ string: 5, fret: 2, finger: 2 }, { string: 4, fret: 1, finger: 1 }, { string: 3, fret: 2, finger: 3 }, { string: 1, fret: 2, finger: 4 }], muted: [6], open: [2] },
  C7:    { label: "C7",    positions: [{ string: 5, fret: 3, finger: 3 }, { string: 4, fret: 2, finger: 2 }, { string: 2, fret: 1, finger: 1 }, { string: 3, fret: 3, finger: 4 }], muted: [6], open: [1] },
  F7:    { label: "F7",    positions: [{ string: 5, fret: 3, finger: 3 }, { string: 3, fret: 2, finger: 2 }], barre: { fret: 1, fromString: 1, toString: 6, finger: 1 }, baseFret: 1 },
  Bb7:   { label: "Bb7",  positions: [{ string: 4, fret: 3, finger: 3 }, { string: 3, fret: 3, finger: 4 }], muted: [6], barre: { fret: 1, fromString: 1, toString: 5, finger: 1 }, baseFret: 1 },
  // ── Major 7ths ────────────────────────────────────────────────────────────
  Cmaj7: { label: "Cmaj7", positions: [{ string: 5, fret: 3, finger: 3 }, { string: 4, fret: 2, finger: 2 }], muted: [6], open: [3, 2, 1] },
  Amaj7: { label: "Amaj7", positions: [{ string: 4, fret: 2, finger: 2 }, { string: 3, fret: 1, finger: 1 }, { string: 2, fret: 2, finger: 3 }], muted: [6], open: [5, 1] },
  Dmaj7: { label: "Dmaj7", positions: [{ string: 3, fret: 2, finger: 1 }, { string: 2, fret: 2, finger: 2 }, { string: 1, fret: 2, finger: 3 }], muted: [6, 5], open: [4] },
  Emaj7: { label: "Emaj7", positions: [{ string: 5, fret: 2, finger: 2 }, { string: 4, fret: 1, finger: 1 }, { string: 3, fret: 1, finger: 1 }], open: [6, 2, 1] },
  Fmaj7: { label: "Fmaj7", positions: [{ string: 4, fret: 3, finger: 3 }, { string: 3, fret: 2, finger: 2 }, { string: 2, fret: 1, finger: 1 }], muted: [6, 5], open: [1] },
  Gmaj7: { label: "Gmaj7", positions: [{ string: 6, fret: 3, finger: 2 }, { string: 5, fret: 2, finger: 1 }, { string: 1, fret: 2, finger: 3 }], open: [4, 3, 2] },
  Bmaj7: { label: "Bmaj7", positions: [{ string: 4, fret: 4, finger: 4 }, { string: 3, fret: 4, finger: 3 }, { string: 2, fret: 3, finger: 2 }], muted: [6], barre: { fret: 2, fromString: 1, toString: 5, finger: 1 }, baseFret: 2 },
  // ── Minor 7ths ────────────────────────────────────────────────────────────
  Am7:   { label: "Am7",   positions: [{ string: 4, fret: 2, finger: 2 }, { string: 2, fret: 1, finger: 1 }], muted: [6], open: [5, 3, 1] },
  Dm7:   { label: "Dm7",   positions: [{ string: 3, fret: 2, finger: 2 }, { string: 2, fret: 1, finger: 1 }], muted: [6, 5], open: [4, 1] },
  Em7:   { label: "Em7",   positions: [{ string: 5, fret: 2, finger: 2 }, { string: 4, fret: 2, finger: 3 }, { string: 2, fret: 3, finger: 4 }], open: [6, 3, 1] },
  Bm7:   { label: "Bm7",   positions: [{ string: 4, fret: 4, finger: 3 }, { string: 2, fret: 3, finger: 2 }], muted: [6], barre: { fret: 2, fromString: 1, toString: 5, finger: 1 }, baseFret: 2 },
  "F#m7": { label: "F#m7", positions: [{ string: 5, fret: 4, finger: 3 }], barre: { fret: 2, fromString: 1, toString: 6, finger: 1 }, baseFret: 2 },
  // ── Sus chords ───────────────────────────────────────────────────────────
  Csus2: { label: "Csus2", positions: [{ string: 4, fret: 5, finger: 3 }, { string: 3, fret: 5, finger: 4 }], muted: [6], barre: { fret: 3, fromString: 1, toString: 5, finger: 1 }, baseFret: 3 },
  Csus4: { label: "Csus4", positions: [{ string: 5, fret: 3, finger: 3 }, { string: 4, fret: 3, finger: 4 }, { string: 2, fret: 1, finger: 1 }, { string: 1, fret: 1, finger: 2 }], muted: [6], open: [3] },
  Dsus2: { label: "Dsus2", positions: [{ string: 3, fret: 2, finger: 2 }, { string: 2, fret: 3, finger: 3 }], muted: [6, 5], open: [4, 1] },
  Dsus4: { label: "Dsus4", positions: [{ string: 3, fret: 2, finger: 1 }, { string: 2, fret: 3, finger: 3 }, { string: 1, fret: 3, finger: 4 }], muted: [6, 5], open: [4] },
  Esus4: { label: "Esus4", positions: [{ string: 5, fret: 2, finger: 2 }, { string: 4, fret: 2, finger: 3 }, { string: 3, fret: 2, finger: 4 }], open: [6, 2, 1] },
  Gsus2: { label: "Gsus2", positions: [{ string: 6, fret: 3, finger: 2 }, { string: 3, fret: 2, finger: 1 }, { string: 2, fret: 3, finger: 3 }, { string: 1, fret: 3, finger: 4 }], open: [5, 4] },
  Gsus4: { label: "Gsus4", positions: [{ string: 6, fret: 3, finger: 2 }, { string: 5, fret: 3, finger: 3 }, { string: 2, fret: 1, finger: 1 }, { string: 1, fret: 3, finger: 4 }], open: [4, 3] },
};

const QUALITY_LABELS: Record<string, string> = {
  "":      "",
  "m":     " Minor",
  "7":     " 7",
  "maj7":  " Maj7",
  "m7":    " Min7",
  "dim":   " Dim",
  "aug":   " Aug",
  "5":     " 5",
  "sus2":  " Sus2",
  "sus4":  " Sus4",
  "9":     " 9",
  "maj9":  " Maj9",
  "m9":    " Min9",
  "min9":  " Min9",
  "add9":  " Add9",
};

function formatChordLabel(chord: string): string {
  const rootMatch = chord.match(/^([A-G][#b]?)/);
  if (!rootMatch) return chord;
  const root = rootMatch[1];
  const quality = chord.slice(root.length).toLowerCase();
  const suffix = QUALITY_LABELS[quality];
  return suffix !== undefined ? root + suffix : chord;
}

// ── Chord playback via Web Audio API ────────────────────────────────────────

let _audioCtx: AudioContext | null = null;

function getAudioContext(): AudioContext | null {
  try {
    if (!_audioCtx || _audioCtx.state === "closed") {
      _audioCtx = new AudioContext();
    }
    return _audioCtx;
  } catch {
    return null;
  }
}

const ROOT_MIDI: Record<string, number> = {
  C: 60, "C#": 61, Db: 61, D: 62, "D#": 63, Eb: 63,
  E: 64, F: 65, "F#": 66, Gb: 66, G: 67, "G#": 68,
  Ab: 68, A: 69, "A#": 70, Bb: 70, B: 71,
};

function chordToMidiNotes(chord: string): number[] {
  const rootMatch = chord.match(/^([A-G][#b]?)/);
  if (!rootMatch) return [60, 64, 67];
  const root = rootMatch[1];
  const quality = chord.slice(root.length);
  const base = (ROOT_MIDI[root] ?? 60) + 12;

  if (quality === "m") return [base, base + 3, base + 7];
  if (quality === "7") return [base, base + 4, base + 7, base + 10];
  if (quality === "maj7") return [base, base + 4, base + 7, base + 11];
  if (quality === "m7") return [base, base + 3, base + 7, base + 10];
  if (quality === "sus2") return [base, base + 2, base + 7];
  if (quality === "sus4") return [base, base + 5, base + 7];
  if (quality === "dim") return [base, base + 3, base + 6];
  if (quality === "aug") return [base, base + 4, base + 8];
  return [base, base + 4, base + 7];
}

function playChord(chord: string) {
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    if (ctx.state === "suspended") ctx.resume();
    const notes = chordToMidiNotes(chord);
    const now = ctx.currentTime;
    notes.forEach((midi) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "triangle";
      osc.frequency.value = 440 * Math.pow(2, (midi - 69) / 12);
      gain.gain.setValueAtTime(0.15, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 1.6);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 1.6);
    });
  } catch {
    // AudioContext not available (e.g. SSR)
  }
}

// ── Diagram rendering helpers ────────────────────────────────────────────────

function stringX(stringNumber: number) {
  return 24 + (6 - stringNumber) * 20;
}

function fretY(fret: number) {
  return 32 + (fret - 0.5) * 22;
}

export function ChordDiagram({ chord }: { chord: string }) {
  const shape = resolveChordShape(chord);
  const frets = 5;
  const strings = 6;

  return (
    <div className="group relative inline-flex">
      <span
        className="cursor-pointer rounded-full border border-purple-400/30 bg-purple-500/10 px-3 py-1 text-xs font-semibold text-purple-100 transition group-hover:border-purple-300 group-hover:bg-purple-500/20 select-none whitespace-nowrap"
        onClick={() => playChord(chord)}
        title="Click to play"
      >
        {formatChordLabel(chord)}
      </span>
      <div className="pointer-events-none absolute bottom-full left-1/2 z-30 hidden -translate-x-1/2 pb-3 group-hover:block">
        <div className="rounded-2xl border border-white/10 bg-gray-950/95 p-3 shadow-2xl shadow-black/30">
          {shape ? (
            <svg width="168" height="190" viewBox="0 0 168 190" className="text-white">
              <text x="84" y="18" textAnchor="middle" className="fill-white text-sm font-semibold">
                {shape.label}
              </text>
              {shape.baseFret && shape.baseFret > 1 ? (
                <text x="144" y="48" textAnchor="middle" className="fill-gray-300 text-[10px]">
                  {shape.baseFret}fr
                </text>
              ) : null}
              {Array.from({ length: strings }).map((_, index) => {
                const stringNumber = 6 - index;
                const x = stringX(stringNumber);
                return <line key={`string-${stringNumber}`} x1={x} y1="36" x2={x} y2="146" stroke="rgba(255,255,255,0.5)" strokeWidth="2" />;
              })}
              {Array.from({ length: frets + 1 }).map((_, index) => {
                const y = 36 + index * 22;
                return (
                  <line
                    key={`fret-${index}`}
                    x1="24"
                    y1={y}
                    x2="124"
                    y2={y}
                    stroke="rgba(255,255,255,0.55)"
                    strokeWidth={index === 0 && (!shape.baseFret || shape.baseFret === 1) ? "5" : "2"}
                  />
                );
              })}
              {shape.barre ? (
                <>
                  <line
                    x1={stringX(shape.barre.toString)}
                    y1={fretY(shape.barre.fret)}
                    x2={stringX(shape.barre.fromString)}
                    y2={fretY(shape.barre.fret)}
                    stroke="#f59e0b"
                    strokeWidth="10"
                    strokeLinecap="round"
                  />
                  <text x={(stringX(shape.barre.toString) + stringX(shape.barre.fromString)) / 2} y={fretY(shape.barre.fret) + 4} textAnchor="middle" className="fill-black text-[9px] font-bold">
                    {shape.barre.finger}
                  </text>
                </>
              ) : null}
              {shape.positions.map((position) => (
                <g key={`${position.string}-${position.fret}`}>
                  <circle cx={stringX(position.string)} cy={fretY(position.fret)} r="8" fill="#60a5fa" />
                  <text x={stringX(position.string)} y={fretY(position.fret) + 3} textAnchor="middle" className="fill-black text-[9px] font-bold">
                    {position.finger}
                  </text>
                </g>
              ))}
              {(shape.open ?? []).map((stringNumber) => (
                <g key={`open-${stringNumber}`}>
                  <circle cx={stringX(stringNumber)} cy="20" r="5" fill="none" stroke="#e5e7eb" strokeWidth="2" />
                </g>
              ))}
              {(shape.muted ?? []).map((stringNumber) => (
                <text key={`mute-${stringNumber}`} x={stringX(stringNumber)} y="24" textAnchor="middle" className="fill-red-300 text-xs font-bold">
                  ×
                </text>
              ))}
            </svg>
          ) : (
            <div className="w-36 rounded-xl border border-dashed border-white/10 bg-black/20 px-4 py-8 text-center text-xs text-gray-400">
              Diagram unavailable for {chord}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
