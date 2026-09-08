from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


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
