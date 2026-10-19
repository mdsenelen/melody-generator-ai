from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference, main  # noqa: E402

# The catch-all Exception handler (like every 500/Exception handler in this
# Starlette version -- see ServerErrorMiddleware) still sends its response,
# but always re-raises afterward so servers can log it; TestClient's default
# raise_server_exceptions=True surfaces that reraise instead of returning the
# response. Disabled here so these tests see the same response a real client
# would.
client = TestClient(main.app, raise_server_exceptions=False)


# --- request body size cap --------------------------------------------------


def test_upload_over_cap_is_rejected_before_being_written_to_disk():
    # MaxBodySizeMiddleware is constructed once at app build time with
    # MAX_UPLOAD_BYTES's value baked in as a constructor kwarg, so
    # monkeypatching the module constant after import wouldn't affect the
    # already-built middleware -- exercise the real configured default
    # instead (matches this codebase's existing convention of passing
    # env-configured values explicitly rather than relying on a monkeypatched
    # default taking effect after the fact, e.g. test_inference_variants.py's
    # max_duration_sec).
    before = set(main.UPLOAD_DIR.glob("*"))

    response = client.post(
        "/api/upload",
        files={"file": ("clip.wav", b"x" * (main.MAX_UPLOAD_BYTES + 1), "audio/wav")},
    )

    assert response.status_code == 413
    after = set(main.UPLOAD_DIR.glob("*"))
    assert after == before  # nothing new was written -- rejected before the read


def test_upload_within_cap_still_succeeds():
    response = client.post(
        "/api/upload",
        files={"file": ("clip.wav", b"RIFF....WAVEfmt ", "audio/wav")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] and body["filename"]
    (main.UPLOAD_DIR / body["filename"]).unlink(missing_ok=True)


def test_upload_rejects_audio_over_the_duration_cap(monkeypatch):
    # Best-effort header probe -- when it can read the duration and it's over
    # MAX_UPLOAD_DURATION_SEC, reject before a job (and before chunked
    # transcription) ever sees it.
    from app import inference

    monkeypatch.setattr(
        inference, "_probe_source_duration_sec",
        lambda raw: inference.MAX_UPLOAD_DURATION_SEC + 120.0,
    )
    before = len(list(main.UPLOAD_DIR.glob("*")))
    response = client.post(
        "/api/upload",
        files={"file": ("long.wav", b"RIFF....WAVEfmt " + b"\x00" * 64, "audio/wav")},
    )
    assert response.status_code == 400
    assert "minutes" in response.json()["detail"].lower()
    assert len(list(main.UPLOAD_DIR.glob("*"))) == before  # not written to disk


def test_upload_allows_audio_whose_duration_cannot_be_probed(monkeypatch):
    # mp3 / m4a: libsndfile can't read the header, probe returns None -- the
    # upload must still go through (the chunked transcribe path is the hard cap).
    from app import inference

    monkeypatch.setattr(inference, "_probe_source_duration_sec", lambda raw: None)
    response = client.post(
        "/api/upload",
        files={"file": ("song.mp3", b"ID3\x04\x00\x00\x00\x00", "audio/mpeg")},
    )
    assert response.status_code == 201
    (main.UPLOAD_DIR / response.json()["filename"]).unlink(missing_ok=True)


def _wav_bytes(duration_sec: float, sample_rate: int = 8000) -> bytes:
    import io

    import numpy as np
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(duration_sec * sample_rate), dtype=np.float32),
             sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_empty_upload_is_rejected_and_leaves_nothing_behind():
    before = set(main.UPLOAD_DIR.glob("*"))

    response = client.post("/api/upload", files={"file": ("clip.wav", b"", "audio/wav")})

    assert response.status_code == 400
    assert response.json() == {"detail": "Empty upload"}
    assert set(main.UPLOAD_DIR.glob("*")) == before


def test_real_over_duration_wav_is_rejected_and_its_file_deleted(monkeypatch):
    # No probe monkeypatch: a real WAV header, probed from the file on disk.
    monkeypatch.setattr(inference, "MAX_UPLOAD_DURATION_SEC", 1.0)
    before = set(main.UPLOAD_DIR.glob("*"))

    response = client.post(
        "/api/upload", files={"file": ("long.wav", _wav_bytes(2.0), "audio/wav")},
    )

    assert response.status_code == 400
    assert "minutes" in response.json()["detail"].lower()
    assert set(main.UPLOAD_DIR.glob("*")) == before


def test_real_wav_within_duration_cap_is_stored_intact():
    payload = _wav_bytes(0.5)

    response = client.post("/api/upload", files={"file": ("ok.wav", payload, "audio/wav")})

    assert response.status_code == 201
    stored = main.UPLOAD_DIR / response.json()["filename"]
    try:
        assert stored.read_bytes() == payload
    finally:
        stored.unlink(missing_ok=True)


def test_upload_streams_to_disk_without_materialising_the_body(monkeypatch):
    # The whole point: never `await file.read()` (up to MAX_UPLOAD_BYTES in
    # RAM on a 512MB container). Fail loudly if anything reads the body in
    # one go, and check the probe gets a path, not bytes.
    import fastapi.datastructures
    import starlette.datastructures

    async def no_full_read(self, size=-1):
        raise AssertionError("upload_audio must not read the whole UploadFile into memory")

    monkeypatch.setattr(starlette.datastructures.UploadFile, "read", no_full_read)
    monkeypatch.setattr(fastapi.datastructures.UploadFile, "read", no_full_read)
    probed_sources = []
    monkeypatch.setattr(
        inference, "_probe_source_duration_sec",
        lambda source: probed_sources.append(source) or None,
    )

    response = client.post(
        "/api/upload", files={"file": ("song.mp3", b"ID3\x04" + b"\x00" * 4096, "audio/mpeg")},
    )

    assert response.status_code == 201
    stored = main.UPLOAD_DIR / response.json()["filename"]
    try:
        assert len(probed_sources) == 1
        assert not isinstance(probed_sources[0], (bytes, bytearray))
        assert Path(probed_sources[0]) == stored
    finally:
        stored.unlink(missing_ok=True)


def test_probe_reads_duration_from_a_path_and_from_bytes_alike(tmp_path):
    payload = _wav_bytes(1.5)
    path = tmp_path / "clip.wav"
    path.write_bytes(payload)

    assert inference._probe_source_duration_sec(payload) == pytest.approx(1.5)
    assert inference._probe_source_duration_sec(path) == pytest.approx(1.5)
    assert inference._probe_source_duration_sec(b"not audio") is None


# --- global unhandled-exception handler --------------------------------------


def test_unhandled_exception_is_logged_and_sanitized(monkeypatch, caplog):
    def boom():
        raise RuntimeError("boom: leaking internals should not happen")

    monkeypatch.setattr(inference, "get_runtime_status", boom)

    with caplog.at_level(logging.ERROR):
        response = client.get("/health")

    assert response.status_code == 500
    assert response.json() == {"detail": "An unexpected error occurred. Please try again."}
    assert "boom" not in response.text
    # The real error (including "boom") is logged via logger.exception's
    # exc_info, not the format string itself -- caplog.text includes the
    # rendered traceback, unlike record.getMessage().
    assert "boom" in caplog.text
    assert any(record.exc_info is not None for record in caplog.records)


def test_unhandled_exception_500_carries_cors_headers(monkeypatch):
    # The browser uploads straight to this backend cross-origin. A 500 without
    # Access-Control-Allow-Origin is hidden behind an opaque "Failed to fetch",
    # which the frontend can only report as "Couldn't reach the server".
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(inference, "get_runtime_status", boom)
    origin = main.CORS_ALLOWED_ORIGINS[0]

    response = client.get("/health", headers={"Origin": origin})

    assert response.status_code == 500
    assert response.headers.get("access-control-allow-origin") == origin


def test_existing_http_exception_handling_is_unchanged():
    response = client.post(
        "/api/upload",
        files={"file": ("clip.xyz", b"not-audio", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Supported formats: wav, mp3, flac, ogg, m4a, webm"}
