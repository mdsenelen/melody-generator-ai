import { chordNames } from "../../src/domain/chordName";

describe("chordNames", () => {
  it("leaves chords with no enharmonic twin alone", () => {
    expect(chordNames("C")).toEqual({ primary: "C", alternate: null });
    expect(chordNames("Am7")).toEqual({ primary: "Am7", alternate: null });
    expect(chordNames("Fmaj7")).toEqual({ primary: "Fmaj7", alternate: null });
  });

  // Major-type chords take the flat-key spelling (Db, Eb, Ab, Bb) except F#,
  // which is as common as Gb and is the guitar-chart default.
  it.each([
    ["C#", "Db", "C#"],
    ["Db", "Db", "C#"],
    ["D#7", "Eb7", "D#7"],
    ["Ebmaj7", "Ebmaj7", "D#maj7"],
    ["Gb", "F#", "Gb"],
    ["F#sus4", "F#sus4", "Gbsus4"],
    ["G#", "Ab", "G#"],
    ["A#7", "Bb7", "A#7"],
  ])("names the major-type chord %s as %s (alt %s)", (input, primary, alternate) => {
    expect(chordNames(input)).toEqual({ primary, alternate });
  });

  // Minor-type chords take the key-signature-friendly spelling: C#m, F#m,
  // G#m (few sharps) but Bbm (5 flats vs 7 sharps). Ebm/D#m is a coin flip;
  // Ebm is the pick.
  it.each([
    ["Dbm", "C#m", "Dbm"],
    ["C#m7", "C#m7", "Dbm7"],
    ["D#m", "Ebm", "D#m"],
    ["Ebm7", "Ebm7", "D#m7"],
    ["Gbm", "F#m", "Gbm"],
    ["Abm7", "G#m7", "Abm7"],
    ["A#m", "Bbm", "A#m"],
    ["Bbmin7", "Bbmin7", "A#min7"],
    ["C#m7b5", "C#m7b5", "Dbm7b5"],
  ])("names the minor-type chord %s as %s (alt %s)", (input, primary, alternate) => {
    expect(chordNames(input)).toEqual({ primary, alternate });
  });

  it("does not mistake maj for minor", () => {
    expect(chordNames("Dbmaj7").primary).toBe("Dbmaj7");
    expect(chordNames("C#maj7")).toEqual({ primary: "Dbmaj7", alternate: "C#maj7" });
  });

  it("returns unparseable input untouched", () => {
    expect(chordNames("N.C.")).toEqual({ primary: "N.C.", alternate: null });
    expect(chordNames("")).toEqual({ primary: "", alternate: null });
  });
});
