import { layoutNotes, pitchBounds, waveformPeaks } from "../../src/domain/noteRoll";

describe("waveformPeaks", () => {
  it("returns the max absolute amplitude per bucket", () => {
    const samples = Float32Array.from([0.1, -0.5, 0.2, 0.9, -0.3, 0.05, 0, 0]);
    expect(Array.from(waveformPeaks(samples, 4))).toEqual([
      expect.closeTo(0.5),
      expect.closeTo(0.9),
      expect.closeTo(0.3),
      0,
    ]);
  });

  it("returns silence for empty input and never more buckets than asked", () => {
    expect(waveformPeaks(new Float32Array(0), 3)).toEqual(new Float32Array(3));
    expect(waveformPeaks(Float32Array.from([1, 1]), 5)).toHaveLength(5);
  });
});

describe("pitchBounds", () => {
  it("pads the played range by two semitones and keeps at least an octave", () => {
    expect(pitchBounds([{ start: 0, end: 1, pitch: 60, velocity: 80 }])).toEqual({
      low: 54,
      high: 66,
    });
    expect(
      pitchBounds([
        { start: 0, end: 1, pitch: 48, velocity: 80 },
        { start: 1, end: 2, pitch: 72, velocity: 80 },
      ]),
    ).toEqual({ low: 46, high: 74 });
  });

  it("falls back to a default range with no notes", () => {
    expect(pitchBounds([])).toEqual({ low: 48, high: 84 });
  });
});

describe("layoutNotes", () => {
  it("maps notes to fractional x/width and row-from-top y", () => {
    const rects = layoutNotes([{ start: 5, end: 10, pitch: 66, velocity: 100 }], {
      durationSec: 20,
      low: 60,
      high: 71,
    });
    expect(rects).toHaveLength(1);
    expect(rects[0].x).toBeCloseTo(0.25);
    expect(rects[0].width).toBeCloseTo(0.25);
    // 12 rows (60..71); pitch 66 is 6 rows up from the bottom
    expect(rects[0].row).toBe(6);
    expect(rects[0].rows).toBe(12);
    expect(rects[0].opacity).toBeGreaterThan(0.3);
  });

  it("drops notes entirely outside the visible bounds or past the duration", () => {
    const rects = layoutNotes(
      [
        { start: 0, end: 1, pitch: 20, velocity: 80 },
        { start: 30, end: 31, pitch: 65, velocity: 80 },
      ],
      { durationSec: 20, low: 60, high: 71 },
    );
    expect(rects).toEqual([]);
  });
});
