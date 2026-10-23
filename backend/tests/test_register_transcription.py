"""POST /api/transcriptions -- register note events the browser already
transcribed (Basic Pitch via TF.js) as a completed job, so /api/analyze,
GET /api/transcribe/{id} and generate-variants' jobId shortcut work exactly as
they do for a server-transcribed job. No audio, no Basic Pitch, no
HEAVY_WORK_LOCK on this path.

Goes through TestClient (the real ASGI stack) so Pydantic validation (422s)
and the rate-limit dependency are exercised, not bypassed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference, rate_limit  # noqa: E402
from app.jobs import routes as job_routes  # noqa: E402
from app.jobs import service  # noqa: E402
from app.jobs.queue import InProcessJobQueue  # noqa: E402
from app.jobs.storage import LocalFilesystemStorage  # noqa: E402
from app.jobs.store import create_sqlite_job_store  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "_JOB_STORE", create_sqlite_job_store(str(tmp_path / "jobs.db")))
    monkeypatch.setattr(service, "_JOB_QUEUE", InProcessJobQueue())
    monkeypatch.setattr(service, "_OBJECT_STORAGE", LocalFilesystemStorage(tmp_path / "storage"))
    rate_limit._HITS.clear()
    yield TestClient(app)
    rate_limit._HITS.clear()


def _payload(**overrides):
    payload = {
        "upload_id": None,
        "source_filename": "vokal izole.mp3",
        "source_duration_sec": 4.0,
        "tfjs_backend": "webgl",
        "note_events": [
            {"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 90},
            {"start": 0.5, "end": 1.0, "pitch": 64, "velocity": 80},
            {"start": 1.0, "end": 1.5, "pitch": 67, "velocity": 70},
            {"start": 2.0, "end": 3.0, "pitch": 72, "velocity": 100},
        ],
    }
    payload.update(overrides)
    return payload


def test_registers_a_completed_job_with_the_transcription_result_shape(client):
    response = client.post("/api/transcriptions", json=_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"

    status = client.get(f"/api/transcribe/{body['job_id']}").json()
    assert status["status"] == "completed"
    result = status["result"]
    for key in (
        "n_notes", "duration_sec", "source_duration_sec", "truncated",
        "midi_b64", "wav_b64", "midi_filename", "wav_filename", "note_events",
    ):
        assert key in result, key
    assert result["n_notes"] == 4
    assert result["source_duration_sec"] == 4.0
    assert result["truncated"] is False
    assert result["midi_b64"]  # real MIDI built server-side from the notes
    assert result["transcribed_by"] == "browser"
    assert result["tfjs_backend"] == "webgl"


def test_a_registered_job_can_be_analyzed(client):
    job_id = client.post("/api/transcriptions", json=_payload()).json()["job_id"]
    response = client.post("/api/analyze", json={"job_id": job_id})
    assert response.status_code == 200, response.text
    assert "key" in response.json()


def test_never_touches_basic_pitch_or_the_heavy_work_lock(client, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("browser registration must not run server transcription")

    monkeypatch.setattr(inference, "_run_basic_pitch_predict", _boom)
    monkeypatch.setattr(inference, "_acquire_heavy_work_lock_blocking", _boom)
    assert client.post("/api/transcriptions", json=_payload()).status_code == 200


def test_zero_length_notes_are_accepted(client):
    # _trim_note_events itself produces end == start via max(start, end).
    notes = [{"start": 1.0, "end": 1.0, "pitch": 60, "velocity": 90}]
    assert client.post("/api/transcriptions", json=_payload(note_events=notes)).status_code == 200


@pytest.mark.parametrize(
    "note",
    [
        {"start": 0.0, "end": 0.5, "pitch": 200, "velocity": 90},
        {"start": 0.0, "end": 0.5, "pitch": -1, "velocity": 90},
        {"start": 1.0, "end": 0.5, "pitch": 60, "velocity": 90},
        {"start": -0.1, "end": 0.5, "pitch": 60, "velocity": 90},
        {"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 0},
        {"start": 0.0, "end": 99.0, "pitch": 60, "velocity": 90},  # past duration + 1s
    ],
)
def test_rejects_invalid_notes_with_422(client, note):
    assert client.post("/api/transcriptions", json=_payload(note_events=[note])).status_code == 422


def test_rejects_a_duration_over_the_upload_maximum(client):
    payload = _payload(source_duration_sec=inference.MAX_UPLOAD_DURATION_SEC + 1)
    assert client.post("/api/transcriptions", json=payload).status_code == 422


def test_rejects_too_many_notes(client, monkeypatch):
    from app import schemas

    too_many = schemas.MAX_REGISTERED_NOTE_EVENTS + 1
    notes = [{"start": 0.0, "end": 0.1, "pitch": 60, "velocity": 90}] * too_many
    assert client.post("/api/transcriptions", json=_payload(note_events=notes)).status_code == 422


def test_rejects_an_unknown_tfjs_backend(client):
    assert client.post("/api/transcriptions", json=_payload(tfjs_backend="cuda")).status_code == 422


def test_has_its_own_rate_limit_bucket_separate_from_transcribe(client):
    # A browser attempt that 422s falls back to POST /api/transcribe; if the
    # two shared a bucket, that fallback would count twice and could 429.
    statuses = [
        client.post("/api/transcriptions", json=_payload()).status_code
        for _ in range(job_routes.REGISTER_TRANSCRIPTION_RATE_LIMIT + 1)
    ]
    assert statuses[:-1] == [200] * job_routes.REGISTER_TRANSCRIPTION_RATE_LIMIT
    assert statuses[-1] == 429

    fallback = client.post(
        "/api/transcribe", files={"file": ("clip.wav", b"audio-bytes", "audio/wav")}
    )
    assert fallback.status_code == 202
