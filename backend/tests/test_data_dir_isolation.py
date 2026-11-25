"""The app lifespan runs cleanup_stale_files(), which deletes week-old files
under UPLOAD_DIR/OUTPUT_DIR. Under pytest those must point at a temp dir, or
every full run silently wipes the developer's real backend/data/recordings."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app import inference, main

REAL_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STALE_MTIME = time.time() - 30 * 24 * 3600


def _write_stale(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stale")
    os.utime(path, (STALE_MTIME, STALE_MTIME))
    return path


def test_data_dirs_point_at_tmp_not_the_real_data_dir() -> None:
    for module in (inference, main):
        for name in ("UPLOAD_DIR", "OUTPUT_DIR", "LOG_DIR"):
            resolved = getattr(module, name).resolve()
            assert REAL_DATA_DIR.resolve() not in (resolved, *resolved.parents), (
                f"{module.__name__}.{name} = {resolved} is inside the real data dir"
            )


def test_app_lifespan_cleanup_leaves_real_recordings_untouched() -> None:
    # A stale sentinel in the real folder is exactly what the lifespan
    # cleanup used to delete; it must survive app startup under pytest.
    sentinel = _write_stale(REAL_DATA_DIR / "recordings" / f"pytest_sentinel_{uuid.uuid4().hex}.wav")
    stale_tmp_upload = _write_stale(inference.UPLOAD_DIR / "upload_old.wav")
    stale_tmp_output = _write_stale(inference.OUTPUT_DIR / "old.mid")
    try:
        with TestClient(main.app):
            pass
        assert sentinel.exists(), "app startup deleted a file in the real data/recordings"
        # Cleanup itself still works -- it just runs against the temp dirs.
        assert not stale_tmp_upload.exists()
        assert not stale_tmp_output.exists()
    finally:
        sentinel.unlink(missing_ok=True)
