import { NOTE_NAMES } from "../utils/pitch";
import { cn } from "./ui/cn";

const OCTAVES = 2;
const WHITE_PCS = [0, 2, 4, 5, 7, 9, 11];
// Left edge of each black key, in white-key widths from the octave's C.
const BLACK_OFFSETS: Record<number, number> = { 1: 0.65, 3: 1.7, 6: 3.65, 8: 4.68, 10: 5.72 };

/** A two-octave keyboard that lights the given pitch classes in steel. Display only. */
export function KeyboardStrip({ pitchClasses }: { pitchClasses: number[] }) {
  const lit = new Set(pitchClasses);
  const whiteCount = WHITE_PCS.length * OCTAVES;
  const label =
    pitchClasses.length > 0
      ? `Keyboard showing ${pitchClasses.map((pc) => NOTE_NAMES[pc]).join(", ")}`
      : "Keyboard, no chord selected";

  return (
    <div
      role="img"
      aria-label={label}
      className="border-border relative h-24 w-full overflow-hidden rounded-[var(--radius)] border bg-[#d9dde3]"
    >
      {Array.from({ length: OCTAVES }, (_, octave) =>
        WHITE_PCS.map((pc, index) => {
          const position = octave * WHITE_PCS.length + index;
          const active = lit.has(pc);
          return (
            <span
              key={`w-${octave}-${pc}`}
              data-key
              data-active={active}
              className={cn(
                "absolute top-0 bottom-0 border-r border-[#14171c]/35 transition-colors",
                active ? "bg-primary" : "bg-[#eceef2]",
              )}
              style={{
                left: `${(position / whiteCount) * 100}%`,
                width: `${100 / whiteCount}%`,
              }}
            />
          );
        }),
      )}
      {Array.from({ length: OCTAVES }, (_, octave) =>
        Object.entries(BLACK_OFFSETS).map(([pcText, offset]) => {
          const pc = Number(pcText);
          const active = lit.has(pc);
          return (
            <span
              key={`b-${octave}-${pc}`}
              data-key
              data-active={active}
              className={cn(
                "absolute top-0 h-[60%] rounded-b-[3px] transition-colors",
                active ? "bg-primary-bright" : "bg-[#14171c]",
              )}
              style={{
                left: `${((octave * WHITE_PCS.length + offset) / whiteCount) * 100}%`,
                width: `${(0.6 / whiteCount) * 100}%`,
              }}
            />
          );
        }),
      )}
    </div>
  );
}
