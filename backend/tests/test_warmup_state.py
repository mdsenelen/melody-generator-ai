"""Tests for the time-bounded warming-up advisory state that replaced
warm-up holding HEAVY_WORK_LOCK (docs/PROGRESS.md, 2026-10-10 design
change).

Why a marker and not the lock: asyncio.wait_for can only stop the *caller*
from waiting -- it cannot forcibly stop the underlying thread, and neither
can anything else in CPython. A warm-up that ever hung while holding
HEAVY_WORK_LOCK would hold it forever, permanently 429ing every later
request. The marker is advisory and self-expiring instead: heavy endpoints
check it (fresh -> fast, distinct rejection; stale, i.e. older than
WARMUP_MAX_SECONDS -> ignored), so a hung/never-cleared warm-up can only
ever block traffic for WARMUP_MAX_SECONDS, never indefinitely.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference  # noqa: E402


@pytest.fixture(autouse=True)
def clean_warming_state():
    """The warming marker is module-level global state -- isolate tests
    from each other and from whatever ran before this module."""
    inference._mark_warming_finished()
    yield
    inference._mark_warming_finished()


def test_request_rejected_while_warm_up_is_in_flight(caplog):
    """A generation request landing while the warm-up marker is fresh must
    fail fast with the distinct "starting up" response, before the real
    work ever runs -- and the rejection must be logged."""
    inference._mark_warming_started()
    called = False

    def should_not_run():
        nonlocal called
        called = True
        return "ok"

    with caplog.at_level(logging.INFO, logger="app.inference"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(inference._run_generation(should_not_run))

    assert not called, "the real work must not run while warm-up is fresh"
    assert exc.value.status_code == 429
    assert exc.value.headers.get("X-Error-Code") == inference.WARMING_UP_ERROR_CODE
    assert exc.value.headers.get("Retry-After")
    assert "starting up" in exc.value.detail.lower()
    assert any("warm-up in progress" in record.message for record in caplog.records)


def test_request_proceeds_once_warming_state_is_older_than_the_max(monkeypatch, caplog):
    """A warm-up marker that's still set but past WARMUP_MAX_SECONDS (a
    hang, a crash before the finally -- anything that left it uncleared)
    must be ignored, not block requests forever -- and that must be logged
    too, distinctly from the rejection case above."""
    monkeypatch.setattr(inference, "WARMUP_MAX_SECONDS", 1.0)
    inference._mark_warming_started()

    real_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 10.0)

    with caplog.at_level(logging.INFO, logger="app.inference"):
        result = asyncio.run(inference._run_generation(lambda: "ok"))

    assert result == "ok"
    assert any("Ignoring stale warm-up marker" in record.message for record in caplog.records)


def test_failing_warm_up_clears_the_warming_state(monkeypatch):
    """warm_up_basic_pitch must clear its marker in a finally -- a caught
    exception must not leave the marker set, which would otherwise block
    every request for the rest of WARMUP_MAX_SECONDS for no reason."""
    monkeypatch.setattr(inference, "basic_pitch_predict", object())

    def boom():
        raise RuntimeError("simulated warm-up failure")

    monkeypatch.setattr(inference, "_warm_up_basic_pitch_sync", boom)

    assert inference._warming_elapsed_seconds() is None
    asyncio.run(inference.warm_up_basic_pitch())
    assert inference._warming_elapsed_seconds() is None, "marker must be cleared after a failure"


def test_health_runtime_reports_warming_up():
    assert inference.get_runtime_status()["warming_up"] is False

    inference._mark_warming_started()
    assert inference.get_runtime_status()["warming_up"] is True

    inference._mark_warming_finished()
    assert inference.get_runtime_status()["warming_up"] is False


def test_health_runtime_warming_up_is_false_once_stale(monkeypatch):
    monkeypatch.setattr(inference, "WARMUP_MAX_SECONDS", 1.0)
    inference._mark_warming_started()

    real_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 10.0)

    assert inference.get_runtime_status()["warming_up"] is False


def test_run_basic_pitch_rejected_while_warm_up_is_in_flight():
    """The transcribe worker must also respect the warming marker --
    without this, warm-up (no longer holding HEAVY_WORK_LOCK) and a real
    transcription would run their heavy work concurrently again, exactly
    the CPU-contention bug this design change fixes."""
    inference._mark_warming_started()
    called = False

    def should_not_run(audio_bytes, on_progress=None):
        nonlocal called
        called = True
        return {}

    with pytest.raises(RuntimeError, match="warm-up"):
        inference.run_basic_pitch(b"irrelevant", "probe.wav")

    assert not inference.HEAVY_WORK_LOCK.locked(), "must never have acquired the lock"


def test_run_basic_pitch_proceeds_once_warming_state_is_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(inference, "WARMUP_MAX_SECONDS", 1.0)
    inference._mark_warming_started()

    real_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 10.0)

    monkeypatch.setattr(
        inference,
        "_transcribe_and_mood_chunked",
        lambda audio_bytes, on_progress=None: {
            "midi_bytes": b"MThd" + b"\x00" * 10,
            "note_events": [],
            "n_notes": 0,
            "duration_sec": 0.0,
            "source_duration_sec": 0.0,
            "truncated": False,
        },
    )

    result = inference.run_basic_pitch(b"irrelevant", "probe.wav")
    assert result["n_notes"] == 0


def test_429_logs_which_label_holds_the_lock_and_for_how_long(caplog):
    """The busy-429 (lock contention, not warming-up) must log who holds
    the lock and how long they've held it -- a bare threading.Lock doesn't
    expose this on its own, so _describe_lock_holder's tracking state is
    what makes it loggable."""
    inference.HEAVY_WORK_LOCK.acquire()
    inference._mark_lock_acquired("test_holder")
    try:
        time.sleep(0.05)
        with caplog.at_level(logging.INFO, logger="app.inference"):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(inference._run_generation(lambda: "should not run"))
        assert exc.value.status_code == 429
        assert exc.value.headers.get("X-Error-Code") == inference.BUSY_ERROR_CODE
        busy_logs = [r.message for r in caplog.records if "HEAVY_WORK_LOCK busy" in r.message]
        assert busy_logs, "must log the 429 with the current holder"
        assert "test_holder" in busy_logs[0]
        assert "held" in busy_logs[0]
    finally:
        inference.HEAVY_WORK_LOCK.release()


def test_lock_acquire_and_release_are_logged_with_label_and_duration(caplog):
    """Timing logs for a normal (uncontended) acquire/release cycle."""
    with caplog.at_level(logging.INFO, logger="app.inference"):
        result = asyncio.run(inference._run_generation(lambda: "ok"))
    assert result == "ok"
    messages = [r.message for r in caplog.records]
    assert any("HEAVY_WORK_LOCK acquired by" in m for m in messages)
    assert any("HEAVY_WORK_LOCK released by" in m and "held" in m for m in messages)
