import { chordNames } from "../../src/domain/chordName";

/**
 * A chord name with its enharmonic twin: the usual spelling plain, the other
 * one italic ("Eb Min7 / *D# Min7*"). `format` turns a raw chord name into its
 * display text (e.g. "Am7" -> "A Min7") and is applied to both.
 */
export function ChordName({
  chord,
  format = (name) => name,
}: {
  chord: string;
  format?: (name: string) => string;
}) {
  const { primary, alternate } = chordNames(chord);
  return (
    <>
      {format(primary)}
      {alternate ? (
        <>
          {" / "}
          <i>{format(alternate)}</i>
        </>
      ) : null}
    </>
  );
}
