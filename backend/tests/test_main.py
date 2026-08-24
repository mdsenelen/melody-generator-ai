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


def test_existing_http_exception_handling_is_unchanged():
    response = client.post(
        "/api/upload",
        files={"file": ("clip.xyz", b"not-audio", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Supported formats: wav, mp3, flac, ogg, m4a, webm"}
