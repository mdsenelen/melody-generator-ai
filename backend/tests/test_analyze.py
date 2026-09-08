"""Step 2 of docs/PLAN-full-transcription-split.md: MIDI-based /api/analyze.

Analysis (mood / key / tempo / chords / pitch histogram) is arithmetic on the
transcribe job's stored note events, sliced to a clip window -- no librosa, no
Basic Pitch, no audio decode. These tests pin that: the numbers are right, a
sub-window slice changes them, and the path never touches the audio helpers.
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference  # noqa: E402
from app.jobs import routes as job_routes  # noqa: E402
from app.jobs import service  # noqa: E402
from app.jobs.store import create_sqlite_job_store  # noqa: E402


def _notes(spec: list[tuple[float, float, int]]) -> list[dict[str, float | int]]:
    return [{"start": s, "end": e, "pitch": p, "velocity": 90} for s, e, p in spec]


# --- _chords_from_note_events -------------------------------------------------


def test_chords_from_note_events_names_a_triad_from_simultaneous_notes():
    # C major triad held together for 2s -> "C"
    chords = inference._chords_from_note_events(_notes([(0.0, 2.0, 60), (0.0, 2.0, 64), (0.0, 2.0, 67)]))
    assert chords == ["C"]


def test_chords_from_note_events_is_empty_for_no_notes():
    assert inference._chords_from_note_events([]) == []


def test_chords_from_note_events_tracks_a_progression_over_windows():
    # C triad for 2s, then G triad for 2s
    spec = [
        (0.0, 2.0, 60), (0.0, 2.0, 64), (0.0, 2.0, 67),
        (2.0, 4.0, 67), (2.0, 4.0, 71), (2.0, 4.0, 74),
    ]
    chords = inference._chords_from_note_events(_notes(spec), window_sec=2.0)
    assert chords == ["C", "G"]


def test_chords_from_note_events_collapses_repeats():
    spec = [(0.0, 6.0, 60), (0.0, 6.0, 64), (0.0, 6.0, 67)]
    assert inference._chords_from_note_events(_notes(spec), window_sec=2.0) == ["C"]


# --- analyze_clip ------------------------------------------------------------


def test_analyze_clip_returns_every_analysis_field():
    spec = [(i * 0.5, i * 0.5 + 0.4, 60 + (i % 5)) for i in range(24)]  # ~120 BPM eighths
    out = inference.analyze_clip(_notes(spec))
    for key in (
        "tempo_bpm", "average_pitch", "mood_idx", "mood_label",
        "key", "pitch_histogram", "detected_chords", "clip_start_sec", "clip_end_sec",
    ):
        assert key in out, key
    assert out["mood_label"] in ("happy", "sad", "neutral")
    assert len(out["pitch_histogram"]) == 12
    assert 40.0 <= out["tempo_bpm"] <= 220.0


def test_analyze_clip_slices_to_the_requested_window():
    # low register 0-10s, high register 20-30s
    low = [(i, i + 0.5, 50) for i in range(10)]
    high = [(20 + i, 20 + i + 0.5, 84) for i in range(10)]
    events = _notes(low + high)

    first = inference.analyze_clip(events, 0.0, 10.0)
    second = inference.analyze_clip(events, 20.0, 30.0)

    assert first["average_pitch"] < second["average_pitch"]
    assert first["pitch_histogram"] != second["pitch_histogram"]


def test_analyze_clip_empty_window_is_neutral_not_a_crash():
    events = _notes([(0.0, 1.0, 60)])
    out = inference.analyze_clip(events, 100.0, 110.0)
    assert out["mood_label"] == "neutral"
    assert out["detected_chords"] == []
    assert out["key"] in ("Unknown", "")


def test_analyze_clip_never_touches_audio_helpers(monkeypatch):
    def _boom(*a, **k):  # pragma: no cover - only fires on regression
        raise AssertionError("analyze_clip must not decode audio or run Basic Pitch")

    monkeypatch.setattr(inference, "_detect_chords_from_audio", _boom)
    monkeypatch.setattr(inference, "_decode_audio_window", _boom)
    monkeypatch.setattr(inference, "_read_audio_bytes", _boom)
    monkeypatch.setattr(inference, "_run_basic_pitch_predict", _boom)

    spec = [(i * 0.4, i * 0.4 + 0.3, 62 + (i % 4)) for i in range(20)]
    out = inference.analyze_clip(_notes(spec))
    assert out["detected_chords"]  # produced from the notes


# --- POST /api/analyze route ----------------------------------------------


@pytest.fixture
def store(tmp_path):
    return create_sqlite_job_store(str(tmp_path / "jobs.db"))


@pytest.fixture(autouse=True)
def _reset_service_singletons(monkeypatch):
    monkeypatch.setattr(service, "_JOB_STORE", None)
    monkeypatch.setattr(service, "_JOB_QUEUE", None)
    monkeypatch.setattr(service, "_OBJECT_STORAGE", None)


def _completed_job_with_notes(store, note_events):
    job, _ = store.create_job(source_filename="clip.wav", storage_key="jobs/1/input.wav")
    store.mark_creation_ready(job.id)
    claimed = store.claim_job(job.id, lease_seconds=120.0)
    result = {"n_notes": len(note_events), "note_events": note_events, "midi_b64": "AAA="}
    assert store.mark_completed(job.id, result, lease_token=claimed.lease_token)
    return store.get_job(job.id)


def test_analyze_route_404_for_unknown_job(store, monkeypatch):
    from app.schemas import AnalyzeRequest

    monkeypatch.setattr(job_routes, "get_job_store", lambda: store)
    with pytest.raises(Exception) as exc:
        asyncio.run(job_routes.analyze_clip_route(AnalyzeRequest(job_id="nope")))
    assert getattr(exc.value, "status_code", None) == 404


def test_analyze_route_409_when_the_job_stored_no_note_events(store, monkeypatch):
    from app.schemas import AnalyzeRequest

    monkeypatch.setattr(job_routes, "get_job_store", lambda: store)
    job, _ = store.create_job(source_filename="c.wav", storage_key="jobs/1/in.wav")
    store.mark_creation_ready(job.id)
    claimed = store.claim_job(job.id, lease_seconds=120.0)
    store.mark_completed(job.id, {"n_notes": 0, "midi_b64": "AAA="}, lease_token=claimed.lease_token)

    with pytest.raises(Exception) as exc:
        asyncio.run(job_routes.analyze_clip_route(AnalyzeRequest(job_id=job.id)))
    assert getattr(exc.value, "status_code", None) == 409


def test_analyze_route_returns_analysis_for_a_completed_job(store, monkeypatch):
    from app.schemas import AnalyzeRequest

    monkeypatch.setattr(job_routes, "get_job_store", lambda: store)
    notes = _notes([(i * 0.5, i * 0.5 + 0.4, 60 + (i % 4)) for i in range(16)])
    job = _completed_job_with_notes(store, notes)

    out = asyncio.run(job_routes.analyze_clip_route(AnalyzeRequest(job_id=job.id)))
    assert out["mood_label"] in ("happy", "sad", "neutral")
    assert len(out["pitch_histogram"]) == 12


def test_analyze_route_rejects_an_inverted_window(store, monkeypatch):
    from app.schemas import AnalyzeRequest

    monkeypatch.setattr(job_routes, "get_job_store", lambda: store)
    job = _completed_job_with_notes(store, _notes([(0.0, 1.0, 60)]))
    with pytest.raises(Exception) as exc:
        asyncio.run(
            job_routes.analyze_clip_route(
                AnalyzeRequest(job_id=job.id, clip_start_sec=10.0, clip_end_sec=5.0)
            )
        )
    assert getattr(exc.value, "status_code", None) == 400
