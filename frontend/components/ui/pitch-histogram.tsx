const NOTES_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/**
 * 12-bar pitch-class histogram. `values` are the fractional weights returned
 * by `/api/analyze` (`pitch_histogram`), already normalised so each bar's
 * height is `value * 100`.
 */
export function PitchHistogram({
  values,
  highlightNote,
}: {
  values: number[];
  highlightNote?: string;
}) {
  return (
    <div className="grid grid-cols-12 gap-2">
      {NOTES_12.map((note, index) => {
        const value = values[index] ?? 0;
        const active = note === highlightNote;
        return (
          <div key={note} className="flex flex-col items-center gap-2">
            <div className="border-border bg-background flex h-28 w-full items-end rounded-[var(--radius)] border p-2">
              <div
                className="w-full rounded-[2px] transition-all duration-300"
                style={{
                  height: `${Math.max(value * 100, 8)}%`,
                  backgroundColor: active ? "#AC20E8" : "#2e2e34",
                }}
              />
            </div>
            <span
              className={`font-mono text-[10px] ${active ? "text-primary" : "text-muted-foreground"}`}
            >
              {note}
            </span>
          </div>
        );
      })}
    </div>
  );
}
