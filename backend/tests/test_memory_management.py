"""Regression tests for the OOM investigation's memory-management fixes
(docs/PROGRESS.md).

HEAVY_WORK_LOCK serializes the transcribe worker against every generation
route -- they previously had no shared lock at all, so a /generate-variants
request (which separately loads torch, ~200-250MB) could run concurrently
with an in-flight transcribe job on the same 512MiB instance. And
_release_memory_to_os() hands freed heap memory back to the OS after each
heavy call, so one job's peak doesn't become the permanent floor for the
next (observed directly: idle RSS ratcheted from ~94MB to 500MB+ over a
handful of real transcriptions).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference  # noqa: E402


def _fake_transcription() -> dict:
    return {
        "midi_bytes": b"MThd" + b"\x00" * 10,
        "note_events": [],
        "n_notes": 0,
        "duration_sec": 0.0,
        "source_duration_sec": 0.0,
        "truncated": False,
        "tempo_bpm": 120.0,
        "avg_pitch": 60.0,
        "mood_idx": 2,
        "mood_label": "neutral",
        "key": "",
        "pitch_histogram": [0] * 12,
        "detected_chords": [],
    }


def test_run_basic_pitch_holds_heavy_work_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(inference, "_midi_bytes_to_wav_b64", lambda *a, **k: None)

    lock_states: list[bool] = []

    def fake_transcribe_and_mood(audio_bytes, decoded=None):
        lock_states.append(inference.HEAVY_WORK_LOCK.locked())
        return _fake_transcription()

    monkeypatch.setattr(inference, "_transcribe_and_mood", fake_transcribe_and_mood)

    assert not inference.HEAVY_WORK_LOCK.locked()
    inference.run_basic_pitch(b"irrelevant", "probe.wav")
    assert lock_states == [True], "run_basic_pitch must hold HEAVY_WORK_LOCK while it works"
    assert not inference.HEAVY_WORK_LOCK.locked(), "must release the lock afterward"


def test_run_generation_holds_heavy_work_lock():
    lock_states: list[bool] = []

    def fake_generation_func():
        lock_states.append(inference.HEAVY_WORK_LOCK.locked())
        return "ok"

    assert not inference.HEAVY_WORK_LOCK.locked()
    result = asyncio.run(inference._run_generation(fake_generation_func))
    assert result == "ok"
    assert lock_states == [True], "_run_generation must hold HEAVY_WORK_LOCK while it works"
    assert not inference.HEAVY_WORK_LOCK.locked()


def test_run_generation_returns_429_immediately_when_lock_is_held():
    """On the 512MB tier a second heavy pipeline OOMs the box, so a
    generation request that can't get the lock must fail fast, not block."""
    from fastapi import HTTPException

    inference.HEAVY_WORK_LOCK.acquire()
    try:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(inference._run_generation(lambda: "should not run"))
        assert exc.value.status_code == 429
        assert exc.value.headers.get("Retry-After")
    finally:
        inference.HEAVY_WORK_LOCK.release()


def test_run_basic_pitch_reraises_when_lock_stays_held(monkeypatch):
    """The transcribe worker waits a bounded while for an in-flight
    generation, then raises so the job is re-queued rather than blocking
    the single worker thread forever."""
    monkeypatch.setattr(inference, "WORKER_HEAVY_WORK_WAIT_SEC", 0.1)
    inference.HEAVY_WORK_LOCK.acquire()
    try:
        with pytest.raises(RuntimeError):
            inference.run_basic_pitch(b"irrelevant", "probe.wav")
    finally:
        inference.HEAVY_WORK_LOCK.release()


def test_transcribe_and_generation_never_run_concurrently(monkeypatch, tmp_path):
    """The actual bug: before HEAVY_WORK_LOCK, the transcribe worker and a
    generation route shared no lock and could run their heavy work at the
    same time, doubling peak memory on the 512MiB instance."""
    monkeypatch.setattr(inference, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(inference, "_midi_bytes_to_wav_b64", lambda *a, **k: None)

    concurrent_count = 0
    count_lock = threading.Lock()
    overlap_detected = threading.Event()

    def enter_critical_section() -> None:
        nonlocal concurrent_count
        with count_lock:
            concurrent_count += 1
            if concurrent_count > 1:
                overlap_detected.set()
        time.sleep(0.15)

    def exit_critical_section() -> None:
        nonlocal concurrent_count
        with count_lock:
            concurrent_count -= 1

    def fake_transcribe_and_mood(audio_bytes, decoded=None):
        enter_critical_section()
        try:
            return _fake_transcription()
        finally:
            exit_critical_section()

    def fake_generation_func():
        enter_critical_section()
        try:
            return "ok"
        finally:
            exit_critical_section()

    monkeypatch.setattr(inference, "_transcribe_and_mood", fake_transcribe_and_mood)

    def run_transcribe() -> None:
        inference.run_basic_pitch(b"irrelevant", "probe.wav")

    def run_generation() -> None:
        asyncio.run(inference._run_generation(fake_generation_func))

    t1 = threading.Thread(target=run_transcribe)
    t2 = threading.Thread(target=run_generation)
    t1.start()
    time.sleep(0.02)  # give t1 a head start so it claims the lock first
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not t1.is_alive() and not t2.is_alive(), "a thread deadlocked"
    assert not overlap_detected.is_set(), (
        "transcribe and generation ran their heavy work at the same time -- "
        "HEAVY_WORK_LOCK isn't actually serializing them"
    )


def test_warm_up_basic_pitch_sync_holds_heavy_work_lock(monkeypatch):
    """The boot warm-up does real CPU-bound work (a librosa resample plus a
    real Basic Pitch predict, docs/PROGRESS.md's 2026-09-15/16 entry) but
    historically never took HEAVY_WORK_LOCK -- so a /generate-variants
    request landing mid-warm-up ran concurrently with it on the free tier's
    single throttled CPU core, starving the generation of CPU until it blew
    through GENERATION_TIMEOUT_SECONDS (the 2026-09-16 "run 3" 504
    regression: client-facing timeout at the 60s mark while memory kept
    climbing in the background). Warm-up must hold the same lock so a
    request arriving during it gets _run_generation's existing fast 429
    instead.
    """
    lock_states: list[bool] = []

    def fake_read_audio_bytes(raw_bytes, target_sr):
        lock_states.append(inference.HEAVY_WORK_LOCK.locked())
        return np.zeros(target_sr, dtype=np.float32), 1.0, False

    def fake_run_basic_pitch_predict(path):
        lock_states.append(inference.HEAVY_WORK_LOCK.locked())
        return None, []

    monkeypatch.setattr(inference, "_read_audio_bytes", fake_read_audio_bytes)
    monkeypatch.setattr(inference, "_run_basic_pitch_predict", fake_run_basic_pitch_predict)

    assert not inference.HEAVY_WORK_LOCK.locked()
    inference._warm_up_basic_pitch_sync()
    assert lock_states == [True, True], "warm-up must hold HEAVY_WORK_LOCK while it works"
    assert not inference.HEAVY_WORK_LOCK.locked(), "must release the lock afterward"


def test_warm_up_and_generation_never_run_concurrently(monkeypatch):
    """The 2026-09-16 "run 3" regression, reproduced: without a shared lock,
    the boot warm-up and a real generation request could run their heavy
    work at the same time, each starving the other of the free tier's one
    CPU core until the generation blew through GENERATION_TIMEOUT_SECONDS."""
    concurrent_count = 0
    count_lock = threading.Lock()
    overlap_detected = threading.Event()

    def enter_critical_section() -> None:
        nonlocal concurrent_count
        with count_lock:
            concurrent_count += 1
            if concurrent_count > 1:
                overlap_detected.set()
        time.sleep(0.15)

    def exit_critical_section() -> None:
        nonlocal concurrent_count
        with count_lock:
            concurrent_count -= 1

    def fake_read_audio_bytes(raw_bytes, target_sr):
        enter_critical_section()
        try:
            return np.zeros(target_sr, dtype=np.float32), 1.0, False
        finally:
            exit_critical_section()

    def fake_run_basic_pitch_predict(path):
        return None, []

    def fake_generation_func():
        enter_critical_section()
        try:
            return "ok"
        finally:
            exit_critical_section()

    monkeypatch.setattr(inference, "_read_audio_bytes", fake_read_audio_bytes)
    monkeypatch.setattr(inference, "_run_basic_pitch_predict", fake_run_basic_pitch_predict)

    def run_warm_up() -> None:
        inference._warm_up_basic_pitch_sync()

    def run_generation() -> None:
        asyncio.run(inference._run_generation(fake_generation_func))

    t1 = threading.Thread(target=run_warm_up)
    t2 = threading.Thread(target=run_generation)
    t1.start()
    time.sleep(0.02)  # give t1 a head start so it claims the lock first
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not t1.is_alive() and not t2.is_alive(), "a thread deadlocked"
    assert not overlap_detected.is_set(), (
        "warm-up and generation ran their heavy work at the same time -- "
        "warm-up isn't holding HEAVY_WORK_LOCK"
    )


def test_release_memory_to_os_is_safe_even_if_ctypes_fails(monkeypatch):
    import ctypes

    def boom(*args, **kwargs):
        raise OSError("no libc here")

    monkeypatch.setattr(ctypes, "CDLL", boom)
    inference._release_memory_to_os()  # must not raise


def test_max_analysis_duration_default_is_60_seconds():
    # Cut from 240s during the OOM investigation (docs/PROGRESS.md): most of
    # a transcription's memory turned out to be a fixed librosa/numba/tflite
    # load cost, not audio-length-proportional, but every MB of margin
    # matters at the 512MiB ceiling. Guards against silently drifting back up.
    if "MAX_ANALYSIS_DURATION_SEC" in os.environ:
        pytest.skip("MAX_ANALYSIS_DURATION_SEC overridden in the environment")
    assert inference.MAX_ANALYSIS_DURATION_SEC == 60.0
