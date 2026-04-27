import {
  frequencyToMidi,
  midiToPitchClass,
  midiToNoteName,
  isInPlayableRange,
  NOTE_NAMES,
} from "../../utils/pitch";

describe("pitch utils", () => {
  describe("frequencyToMidi", () => {
    it("should convert A4 (440 Hz) to MIDI 69", () => {
      expect(frequencyToMidi(440)).toBe(69);
    });

    it("should convert C4 (middle C, ~261.63 Hz) to MIDI 60", () => {
      expect(frequencyToMidi(261.63)).toBe(60);
    });

    it("should round to nearest semitone", () => {
      // A4 is 440 Hz, A#4 is ~466 Hz. 450 is closer to A4 (69) than A#4 (70)
      expect(frequencyToMidi(450)).toBe(69);
    });

    it("should handle low frequencies", () => {
      // C2 is ~65.41 Hz = MIDI 36
      expect(frequencyToMidi(65.41)).toBe(36);
    });

    it("should handle high frequencies", () => {
      // C7 is ~2093 Hz = MIDI 96
      expect(frequencyToMidi(2093)).toBe(96);
    });
  });

  describe("midiToPitchClass", () => {
    it("should convert MIDI 0 (C-1) to pitch class 0", () => {
      expect(midiToPitchClass(0)).toBe(0);
    });

    it("should convert MIDI 69 (A4) to pitch class 9", () => {
      expect(midiToPitchClass(69)).toBe(9);
    });

    it("should wrap octaves — MIDI 12 is another C, pitch class 0", () => {
      expect(midiToPitchClass(12)).toBe(0);
    });

    it("should handle negative MIDI values without returning negative pitch class", () => {
      // MIDI -3 wraps to pitch class 9 (A)
      expect(midiToPitchClass(-3)).toBe(9);
    });

    it("should always return a value in 0–11 for any MIDI note", () => {
      for (let midi = -12; midi <= 127; midi++) {
        const pc = midiToPitchClass(midi);
        expect(pc).toBeGreaterThanOrEqual(0);
        expect(pc).toBeLessThan(12);
      }
    });
  });

  describe("midiToNoteName", () => {
    it("should convert MIDI 60 (middle C) to C4", () => {
      expect(midiToNoteName(60)).toBe("C4");
    });

    it("should convert MIDI 69 (A4) to A4", () => {
      expect(midiToNoteName(69)).toBe("A4");
    });

    it("should convert MIDI 61 to C#4", () => {
      expect(midiToNoteName(61)).toBe("C#4");
    });

    it("should handle octave boundary — B4 to C5", () => {
      expect(midiToNoteName(71)).toBe("B4");
      expect(midiToNoteName(72)).toBe("C5");
    });

    it("should produce names whose pitch prefix is always in NOTE_NAMES", () => {
      for (let midi = 0; midi <= 127; midi++) {
        const name = midiToNoteName(midi);
        // strip the trailing octave digit(s) — note name is everything before the last run of digits
        const pitchClass = name.replace(/-?\d+$/, "");
        expect(NOTE_NAMES).toContain(pitchClass);
      }
    });
  });

  describe("isInPlayableRange", () => {
    it("should accept A4 (440 Hz)", () => {
      expect(isInPlayableRange(440)).toBe(true);
    });

    it("should accept the lower boundary — 65 Hz", () => {
      expect(isInPlayableRange(65)).toBe(true);
    });

    it("should accept the upper boundary — 2100 Hz", () => {
      expect(isInPlayableRange(2100)).toBe(true);
    });

    it("should reject frequencies below 65 Hz", () => {
      expect(isInPlayableRange(64)).toBe(false);
      expect(isInPlayableRange(20)).toBe(false);
    });

    it("should reject frequencies above 2100 Hz", () => {
      expect(isInPlayableRange(2101)).toBe(false);
      expect(isInPlayableRange(5000)).toBe(false);
    });
  });
});
