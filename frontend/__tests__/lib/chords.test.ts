import { chordPitchClasses } from "../../data/chords";

describe("chordPitchClasses", () => {
  it.each([
    ["C", [0, 4, 7]],
    ["Am", [9, 0, 4]],
    ["F#dim", [6, 9, 0]],
    ["Ebaug", [3, 7, 11]],
    ["G5", [7, 2]],
    ["Bb7", [10, 2, 5, 8]],
    ["Cmaj7", [0, 4, 7, 11]],
    ["C#m7", [1, 4, 8, 11]],
    ["Bdim7", [11, 2, 5, 8]],
    ["Dsus2", [2, 4, 9]],
    ["Asus4", [9, 2, 4]],
  ])("spells %s", (chord, expected) => {
    expect(chordPitchClasses(chord)).toEqual(expected);
  });

  it("returns an empty list for something that is not a chord", () => {
    expect(chordPitchClasses("H7")).toEqual([]);
    expect(chordPitchClasses("Cwhatever")).toEqual([]);
  });
});
