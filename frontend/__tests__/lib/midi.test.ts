import { Midi } from "@tonejs/midi";

import { base64ToArrayBuffer, MidiParseError, parseMidi } from "../../app/lib/midi";

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function fixtureMidiBase64(): string {
  const midi = new Midi();
  const track = midi.addTrack();
  track.addNote({ midi: 60, time: 0, duration: 0.5, velocity: 0.8 });
  track.addNote({ midi: 64, time: 0.5, duration: 0.5, velocity: 0.7 });
  track.addNote({ midi: 67, time: 1.0, duration: 1.0, velocity: 0.9 });
  return bytesToBase64(midi.toArray());
}

describe("base64ToArrayBuffer", () => {
  it("round-trips arbitrary bytes", () => {
    const original = new Uint8Array([0, 1, 2, 253, 254, 255, 77, 120]);
    const restored = new Uint8Array(base64ToArrayBuffer(bytesToBase64(original)));
    expect(Array.from(restored)).toEqual(Array.from(original));
  });
});

describe("parseMidi", () => {
  it("returns the notes in time order with a total duration", () => {
    const parsed = parseMidi(fixtureMidiBase64());

    expect(parsed.notes.map((note) => note.name)).toEqual(["C4", "E4", "G4"]);
    expect(parsed.notes[0].time).toBeCloseTo(0);
    expect(parsed.notes[1].time).toBeCloseTo(0.5);
    expect(parsed.notes[2].duration).toBeCloseTo(1.0);
    expect(parsed.durationSec).toBeCloseTo(2.0);
  });

  it("treats an empty MIDI as nothing to play, not an error", () => {
    const empty = new Midi();
    empty.addTrack();
    const parsed = parseMidi(bytesToBase64(empty.toArray()));

    expect(parsed.notes).toEqual([]);
    expect(parsed.durationSec).toBe(0);
  });

  it("throws MidiParseError on input that is not a MIDI file", () => {
    expect(() => parseMidi(btoa("this is plainly not midi"))).toThrow(MidiParseError);
  });

  it("throws MidiParseError on input that is not valid base64", () => {
    expect(() => parseMidi("@@@not base64@@@")).toThrow(MidiParseError);
  });
});
