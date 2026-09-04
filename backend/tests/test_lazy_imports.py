"""Guards the OOM fix (see docs/PROGRESS.md).

Importing the app, hitting ``/health``'s status call, and running the whole
transcription path must NOT import ``torch`` -- it costs ~150-250 MB RSS and
on the 512 MiB deploy that is the difference between fitting and being
OOM-killed. Only the generation paths (CVAE / IDDM-PPO) may pull it in.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _run_isolated(snippet: str) -> None:
    """Run snippet in a fresh interpreter (so sys.modules starts clean)."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )


def test_importing_app_main_does_not_import_torch() -> None:
    _run_isolated(
        """
        import sys
        import app.main  # noqa: F401

        leaked = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
        assert not leaked, f"torch imported just by importing app.main: {leaked}"
        """
    )


def test_runtime_status_does_not_import_torch() -> None:
    _run_isolated(
        """
        import sys
        from app import inference

        status = inference.get_runtime_status()
        assert status["device"] == "cpu"
        assert "torch" not in sys.modules, "get_runtime_status() imported torch"
        """
    )


def test_transcription_path_does_not_import_torch() -> None:
    _run_isolated(
        """
        import io, sys
        import numpy as np
        import soundfile as sf
        from app import inference

        buf = io.BytesIO()
        sr = 22050
        t = np.linspace(0.0, 1.0, sr, endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype("float32")
        sf.write(buf, tone, sr, format="WAV")

        result = inference.run_basic_pitch(buf.getvalue(), "probe.wav")
        assert "mood_label" in result

        leaked = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
        assert not leaked, f"the transcription path imported torch: {leaked}"
        """
    )


def test_generation_import_path_still_exposes_torch_backed_names() -> None:
    # The split moved torch-free helpers to app.model.tokens; colab_parity
    # must still re-export them, and importing it (the generation path) does
    # bring in torch -- that's expected and fine.
    _run_isolated(
        """
        import sys
        from app.model.colab_parity import (  # noqa: F401
            MelodyCVAE,
            build_mood_onehot,
            heuristic_mood_from_metrics,
            tokens_to_midi,
        )

        assert "torch" in sys.modules
        """
    )
