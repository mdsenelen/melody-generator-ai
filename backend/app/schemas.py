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
