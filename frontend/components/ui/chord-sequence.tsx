import { cn } from "./cn";

/** A row of chord/note-name chips. Plain display chips unless `onSelect` is given. */
export function ChordSequence({
  chords,
  onSelect,
}: {
  chords: string[];
  onSelect?: (chord: string, index: number) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {chords.map((chord, index) => {
        const className = cn(
          "rounded-[var(--radius)] border border-primary/25 px-2.5 py-0.5 font-mono text-xs text-primary",
          onSelect && "cursor-pointer hover:border-primary/60 hover:bg-primary/8",
        );
        if (!onSelect) {
          return (
            <span key={`${chord}-${index}`} className={className}>
              {chord}
            </span>
          );
        }
        return (
          <button
            key={`${chord}-${index}`}
            type="button"
            onClick={() => onSelect(chord, index)}
            className={className}
          >
            {chord}
          </button>
        );
      })}
    </div>
  );
}
