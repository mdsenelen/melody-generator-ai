from __future__ import annotations

import sys
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))


@pytest.fixture(autouse=True)
def _isolate_data_dirs(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every module-level data dir at a per-test temp dir.

    main.py's lifespan runs inference.cleanup_stale_files() on startup, and
    uploads/generated files are written via these constants -- left alone, any
    TestClient(main.app) test deletes real files in backend/data/recordings.
    Tests that need a specific dir still monkeypatch over this one.
    """
    from app import inference, main

    data_dir = tmp_path_factory.mktemp("data")
    for name in ("UPLOAD_DIR", "OUTPUT_DIR", "LOG_DIR"):
        directory = data_dir / name.removesuffix("_DIR").lower()
        directory.mkdir()
        for module in (inference, main):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, directory)
