import { Midi } from "@tonejs/midi";

// Parsing helpers for the base64 MIDI the generation endpoints return. Pure and
// framework-free -- the Tone.js playback lifecycle lives in
// hooks/use-midi-player.ts, this file just turns bytes into a note list.

export type PlayableNote = {
  /** Scientific pitch name, e.g. "C4" -- what Tone's synths accept. */
  name: string;
  /** Seconds from the start of the piece. */
  time: number;
  /** Seconds. */
  duration: number;
  /** 0..1. */
  velocity: number;
};

export type ParsedMidi = {
  notes: PlayableNote[];
  durationSec: number;
};

export class MidiParseError extends Error {
  constructor(message = "Could not read this MIDI data") {
    super(message);
    this.name = "MidiParseError";
  }
}

export function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

export function parseMidi(b64: string): ParsedMidi {
  let midi: Midi;
  try {
    midi = new Midi(new Uint8Array(base64ToArrayBuffer(b64)));
  } catch (error) {
    throw new MidiParseError(
      error instanceof Error ? `Could not read this MIDI data: ${error.message}` : undefined,
    );
  }

  const notes: PlayableNote[] = midi.tracks
    .flatMap((track) => track.notes)
    .map((note) => ({
      name: note.name,
      time: note.time,
      duration: note.duration,
      velocity: note.velocity,
    }))
    .sort((a, b) => a.time - b.time);

  return {
    notes,
    durationSec: notes.length > 0 ? midi.duration : 0,
  };
}
