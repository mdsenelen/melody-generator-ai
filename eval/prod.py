"""Adapter to the production transcription code. Everything here imports from
backend/app/inference.py; nothing is copied. Importing this module never changes
production behaviour: the two context managers patch module attributes only for
the duration of a `with` block."""
from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import inspect
import io
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from . import config

if str(config.BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(config.BACKEND_DIR))
os.environ.setdefault("PYTHONPATH", str(config.BACKEND_DIR))

from app import inference as inf  # noqa: E402  (path set up above)


class BasicPitchUnavailable(RuntimeError):
    pass


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """basic_pitch print()s a banner per call; keep result output clean."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def require_real_basic_pitch() -> dict[str, str]:
    """Fail loudly instead of benchmarking a fallback.

    In production code a failed `import basic_pitch` leaves `basic_pitch_predict = None`
    and `_run_basic_pitch_predict` then returns zero notes without raising. Scoring that
    would produce a confident-looking F1 of 0, so refuse to run."""
    if inf.basic_pitch_predict is None:
        raise BasicPitchUnavailable(
            "backend.app.inference could not import basic_pitch (predict is None). "
            "Use the eval venv: pip install -r eval/requirements-eval.txt (it pins setuptools<81)."
        )
    model = inf._get_basic_pitch_model()
    if model is None:
        raise BasicPitchUnavailable("Basic Pitch model failed to load")
    runtime = type(model.model)
    return {
        "backend": f"{runtime.__module__.split('.')[0]}.{runtime.__name__}",
        "model_path": str(Path(inf.ICASSP_2022_MODEL_PATH).name),
    }


def sample_rate() -> int:
    return int(inf.NOTEBOOK_VARIANT_AUDIO_DEFAULTS["sample_rate"])


def chunk_settings() -> dict[str, float]:
    sig = inspect.signature(inf._merge_chunk_notes).parameters
    return {
        "chunk_sec": float(inf.TRANSCRIBE_CHUNK_SEC),
        "overlap_sec": float(inf.TRANSCRIBE_OVERLAP_SEC),
        "hop_sec": max(1.0, float(inf.TRANSCRIBE_CHUNK_SEC - inf.TRANSCRIBE_OVERLAP_SEC)),
        "weld_edge_eps": float(sig["edge_eps"].default),
        "weld_edge_in": float(sig["edge_in"].default),
        "weld_gap": float(sig["gap"].default),
        "weld_dedup": float(sig["dedup"].default),
    }


def transcribe_chunked(audio_bytes: bytes) -> tuple[list[dict], int]:
    """The production chunked path (30 s windows, 4 s overlap, boundary welding),
    exactly as `_transcribe_and_mood_chunked` drives it, minus the MIDI-bytes build
    and the pyin fallback for an empty result (which would not be Basic Pitch)."""
    duration = inf._probe_source_duration_sec(audio_bytes)
    if duration is None:
        raise ValueError("could not probe audio duration")
    with _quiet():
        notes, n_chunks = inf._transcribe_full_chunked(audio_bytes, sample_rate(), duration)
    return notes, n_chunks


def transcribe_whole(wav_path: str) -> list[dict]:
    """Single-pass Basic Pitch over the entire file (what chunking approximates)."""
    with _quiet():
        _, notes = inf._run_basic_pitch_predict(wav_path)
    return notes


@contextlib.contextmanager
def weld_edge_eps(value: float) -> Iterator[None]:
    original = inf._merge_chunk_notes
    inf._merge_chunk_notes = functools.partial(original, edge_eps=value)
    try:
        yield
    finally:
        inf._merge_chunk_notes = original


@contextlib.contextmanager
def memoized_predictions() -> Iterator[dict[str, int]]:
    """Cache Basic Pitch results by audio content so a parameter sweep over the
    merge step does not re-run the network. Yields hit/miss counters."""
    original = inf._run_basic_pitch_predict
    cache: dict[str, tuple[Any, list[dict]]] = {}
    stats = {"hits": 0, "misses": 0}

    def wrapper(audio_path: str):
        key = hashlib.sha1(Path(audio_path).read_bytes()).hexdigest()
        if key in cache:
            stats["hits"] += 1
        else:
            stats["misses"] += 1
            cache[key] = original(audio_path)
        midi, notes = cache[key]
        return midi, copy.deepcopy(notes)

    inf._run_basic_pitch_predict = wrapper
    try:
        yield stats
    finally:
        inf._run_basic_pitch_predict = original
