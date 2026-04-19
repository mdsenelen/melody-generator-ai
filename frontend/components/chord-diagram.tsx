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

const CHORD_SHAPES: Record<string, ChordShape> = {
  C: { label: "C", positions: [{ string: 5, fret: 3, finger: 3 }, { string: 4, fret: 2, finger: 2 }, { string: 2, fret: 1, finger: 1 }], muted: [6], open: [3, 1] },
  Cm: { label: "Cm", positions: [{ string: 5, fret: 3, finger: 3 }, { string: 4, fret: 5, finger: 4 }, { string: 3, fret: 5, finger: 4 }, { string: 2, fret: 4, finger: 2 }, { string: 1, fret: 3, finger: 1 }], muted: [6], barre: { fret: 3, fromString: 1, toString: 5, finger: 1 }, baseFret: 3 },
  D: { label: "D", positions: [{ string: 3, fret: 2, finger: 1 }, { string: 2, fret: 3, finger: 3 }, { string: 1, fret: 2, finger: 2 }], muted: [6, 5], open: [4] },
  Dm: { label: "Dm", positions: [{ string: 3, fret: 2, finger: 2 }, { string: 2, fret: 3, finger: 3 }, { string: 1, fret: 1, finger: 1 }], muted: [6, 5], open: [4] },
  E: { label: "E", positions: [{ string: 5, fret: 2, finger: 2 }, { string: 4, fret: 2, finger: 3 }, { string: 3, fret: 1, finger: 1 }], open: [6, 2, 1] },
  Em: { label: "Em", positions: [{ string: 5, fret: 2, finger: 2 }, { string: 4, fret: 2, finger: 3 }], open: [6, 3, 2, 1] },
  F: { label: "F", positions: [{ string: 4, fret: 3, finger: 3 }, { string: 3, fret: 2, finger: 2 }, { string: 2, fret: 1, finger: 1 }], barre: { fret: 1, fromString: 1, toString: 6, finger: 1 }, baseFret: 1 },
  G: { label: "G", positions: [{ string: 6, fret: 3, finger: 2 }, { string: 5, fret: 2, finger: 1 }, { string: 1, fret: 3, finger: 3 }], open: [4, 3, 2] },
  Am: { label: "Am", positions: [{ string: 4, fret: 2, finger: 2 }, { string: 3, fret: 2, finger: 3 }, { string: 2, fret: 1, finger: 1 }], muted: [6], open: [5, 1] },
  Bm: { label: "Bm", positions: [{ string: 4, fret: 4, finger: 3 }, { string: 3, fret: 4, finger: 4 }, { string: 2, fret: 3, finger: 2 }], muted: [6], barre: { fret: 2, fromString: 1, toString: 5, finger: 1 }, baseFret: 2 },
  G7: { label: "G7", positions: [{ string: 6, fret: 3, finger: 2 }, { string: 5, fret: 2, finger: 1 }, { string: 1, fret: 1, finger: 1 }], open: [4, 3, 2] },
  Cmaj7: { label: "Cmaj7", positions: [{ string: 5, fret: 3, finger: 3 }, { string: 4, fret: 2, finger: 2 }], muted: [6], open: [3, 2, 1] },
  Am7: { label: "Am7", positions: [{ string: 4, fret: 2, finger: 2 }, { string: 2, fret: 1, finger: 1 }], muted: [6], open: [5, 3, 1] },
  Dm7: { label: "Dm7", positions: [{ string: 3, fret: 2, finger: 2 }, { string: 2, fret: 1, finger: 1 }], muted: [6, 5], open: [4, 1] },
};

function stringX(stringNumber: number) {
  return 24 + (6 - stringNumber) * 20;
}

function fretY(fret: number) {
  return 32 + (fret - 0.5) * 22;
}

export function ChordDiagram({ chord }: { chord: string }) {
  const shape = CHORD_SHAPES[chord];
  const frets = 5;
  const strings = 6;

  return (
    <div className="group relative inline-flex">
      <span className="cursor-default rounded-full border border-purple-400/30 bg-purple-500/10 px-3 py-1 text-xs font-semibold text-purple-100 transition group-hover:border-purple-300 group-hover:bg-purple-500/20">
        {chord}
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
