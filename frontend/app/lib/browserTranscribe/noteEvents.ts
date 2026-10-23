import type { NoteEvent } from "../transcribeJob";
import type { BasicPitchNote } from "./protocol";

const round3 = (value: number) => Math.round(value * 1000) / 1000;

// Basic Pitch's timed notes -> the NoteEvent shape the server persists.
// Mirrors backend/app/inference.py: velocity = amplitude * 127 clamped to
// 1..127 (_coerce_note_events), times rounded to ms with end >= start
// (_trim_note_events).
export function toNoteEvents(notes: BasicPitchNote[]): NoteEvent[] {
  return notes
    .map((note) => {
      const start = round3(Math.max(0, note.startTimeSeconds));
      return {
        start,
        end: Math.max(start, round3(note.startTimeSeconds + note.durationSeconds)),
        pitch: Math.round(note.pitchMidi),
        velocity: Math.max(1, Math.min(127, Math.round(note.amplitude * 127))),
      };
    })
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}
