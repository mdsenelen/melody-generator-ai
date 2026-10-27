from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# Cap on note events a browser registration may carry. A dense 10-minute
# polyphonic clip from Basic Pitch lands in the low thousands; 20k leaves wide
# margin while bounding validation + MIDI-build work on the 0.1 vCPU tier. A
# 422 here is not user-facing: the frontend falls back to server transcription.
MAX_REGISTERED_NOTE_EVENTS = 20_000

# Notes may end slightly past the probed duration (decoder padding, the last
# frame's hop); anything further out is malformed.
NOTE_END_TOLERANCE_SEC = 1.0


class GenerateRequest(BaseModel):
    filename: Optional[str] = None
    id: Optional[str] = None
    chord: Optional[str] = None
    creativity: float = 0.7
    duration: Optional[float] = None
    bpm: float = 140.0
    instrument: int = 0
    seed: Optional[int] = None


class ProcessRequest(BaseModel):
    id: str
    intensity: float = 0.5
    creativity: float = 0.7


class GenerateProgressionRequest(BaseModel):
    progression: list[str]
    bpm: float = 120.0
    instrument: int = 0


class AnalyzeRequest(BaseModel):
    """POST /api/analyze -- mood / key / tempo / chords / pitch histogram for a
    clip of a completed transcribe job, computed from its stored note events
    (no audio, no Basic Pitch). clip_end_sec None means "to the end"."""

    job_id: str
    clip_start_sec: float = 0.0
    clip_end_sec: Optional[float] = None


class NoteEventIn(BaseModel):
    start: float = Field(ge=0.0)
    end: float
    pitch: int = Field(ge=0, le=127)
    velocity: int = Field(ge=1, le=127)

    @model_validator(mode="after")
    def _end_not_before_start(self) -> "NoteEventIn":
        # end == start is allowed: the server's own _trim_note_events
        # produces zero-length notes via max(start, end).
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class RegisterTranscriptionRequest(BaseModel):
    """POST /api/transcriptions -- note events the browser already
    transcribed (Basic Pitch via TF.js), registered as a completed job.
    source_duration_sec's upper bound (MAX_UPLOAD_DURATION_SEC) is checked in
    the route, which can import inference lazily; this module can't."""

    # /api/upload's ids are uuid4().hex. Anything else is rejected so it can't
    # glob another upload (_resolve_upload_path matches upload_{id}.*).
    upload_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    source_filename: str = Field(min_length=1, max_length=255)
    source_duration_sec: float = Field(gt=0.0)
    tfjs_backend: Optional[Literal["webgl", "wasm", "cpu"]] = None
    note_events: list[NoteEventIn] = Field(max_length=MAX_REGISTERED_NOTE_EVENTS)

    @model_validator(mode="after")
    def _notes_within_duration(self) -> "RegisterTranscriptionRequest":
        limit = self.source_duration_sec + NOTE_END_TOLERANCE_SEC
        if any(n.end > limit for n in self.note_events):
            raise ValueError("note_events extend past source_duration_sec")
        return self
