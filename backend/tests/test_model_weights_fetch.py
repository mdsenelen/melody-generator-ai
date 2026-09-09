"""_ensure_joint_weights: fetch the joint checkpoint on first use.

The trained weights aren't in git (docs/PROGRESS.md "Model weights delivery").
On the deployed backend they're pulled from MODEL_WEIGHTS_URL the first time a
generation request reaches _load_cvae_iddm. This must stay stdlib-only (no
torch) and degrade to the existing 503 when the download can't happen.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_bundle(monkeypatch):
    monkeypatch.setattr(inference, "_CVAE_IDDM_BUNDLE", None)


def test_noop_when_the_file_already_exists(tmp_path, monkeypatch):
    dest = tmp_path / "joint_e2e_weights.pth"
    dest.write_bytes(b"already here" * 100)
    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", dest)
    monkeypatch.setattr(inference, "MODEL_WEIGHTS_URL", "https://example.test/w.pth")

    def _boom(*a, **k):  # pragma: no cover - only fires on regression
        raise AssertionError("must not download when the file is already on disk")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    inference._ensure_joint_weights()
    assert dest.read_bytes().startswith(b"already here")


def test_noop_when_no_url_is_configured(tmp_path, monkeypatch):
    dest = tmp_path / "joint_e2e_weights.pth"
    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", dest)
    monkeypatch.setattr(inference, "MODEL_WEIGHTS_URL", "")

    inference._ensure_joint_weights()
    assert not dest.exists()


def test_downloads_and_places_the_file_atomically(tmp_path, monkeypatch):
    dest = tmp_path / "sub" / "joint_e2e_weights.pth"
    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", dest)
    monkeypatch.setattr(inference, "WEIGHTS_DIR", dest.parent)
    monkeypatch.setattr(inference, "MODEL_WEIGHTS_URL", "https://example.test/w.pth")

    payload = b"PK\x03\x04" + b"fake torch checkpoint bytes" * 500

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(payload))

    inference._ensure_joint_weights()

    assert dest.exists()
    assert dest.read_bytes() == payload
    assert not dest.with_suffix(".pth.partial").exists()


def test_download_failure_leaves_no_file_and_does_not_raise(tmp_path, monkeypatch):
    dest = tmp_path / "joint_e2e_weights.pth"
    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", dest)
    monkeypatch.setattr(inference, "WEIGHTS_DIR", dest.parent)
    monkeypatch.setattr(inference, "MODEL_WEIGHTS_URL", "https://example.test/w.pth")

    def _fail(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", _fail)

    inference._ensure_joint_weights()  # must not raise
    assert not dest.exists()
    assert not dest.with_suffix(".pth.partial").exists()


def test_truncated_download_is_rejected(tmp_path, monkeypatch):
    dest = tmp_path / "joint_e2e_weights.pth"
    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", dest)
    monkeypatch.setattr(inference, "WEIGHTS_DIR", dest.parent)
    monkeypatch.setattr(inference, "MODEL_WEIGHTS_URL", "https://example.test/w.pth")

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(b"nope"))

    inference._ensure_joint_weights()
    assert not dest.exists()  # too small -> discarded


def test_load_cvae_iddm_503s_and_caches_when_no_url_and_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", tmp_path / "joint.pth")
    monkeypatch.setattr(inference, "CVAE_WEIGHTS_PATH", tmp_path / "cvae.pth")
    monkeypatch.setattr(inference, "IDDM_WEIGHTS_PATH", tmp_path / "iddm.pth")
    monkeypatch.setattr(inference, "MODEL_WEIGHTS_URL", "")

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        inference._load_cvae_iddm()
    assert exc.value.status_code == 503
    # cached -- a permanent "no weights here" state
    assert inference._CVAE_IDDM_BUNDLE is not None
    assert inference._CVAE_IDDM_BUNDLE.get("load_error")


def test_load_cvae_iddm_503s_without_caching_when_the_download_failed(tmp_path, monkeypatch):
    # URL configured but the fetch failed -> 503 that a later request retries.
    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", tmp_path / "joint.pth")
    monkeypatch.setattr(inference, "CVAE_WEIGHTS_PATH", tmp_path / "cvae.pth")
    monkeypatch.setattr(inference, "IDDM_WEIGHTS_PATH", tmp_path / "iddm.pth")
    monkeypatch.setattr(inference, "WEIGHTS_DIR", tmp_path)
    monkeypatch.setattr(inference, "MODEL_WEIGHTS_URL", "https://example.test/w.pth")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        inference._load_cvae_iddm()
    assert exc.value.status_code == 503
    # NOT cached -- so the next generation attempt re-tries the fetch
    assert inference._CVAE_IDDM_BUNDLE is None
