"""TEMPORARY diagnostic (Phase B import audit) -- delete after reading CI output.

Reports which heavy modules are resident in sys.modules at three points:
  1. right after `import app.main`
  2. after `warm_up_basic_pitch()`
  3. after a real `run_basic_pitch()` transcription
so we know what's actually loaded at boot vs deferred, and whether the
music21 key-analysis path or the numpy histogram fallback is taken.
Fails on purpose so pytest prints the captured report.
"""

from __future__ import annotations

import asyncio
import io
import sys

import numpy as np
import soundfile as sf

HEAVY = [
    "torch",
    "music21",
    "matplotlib",
    "tensorflow",
    "tflite_runtime",
    "tensorflow_lite",
    "onnxruntime",
    "coremltools",
    "sklearn",
    "scipy",
    "numba",
    "llvmlite",
    "resampy",
    "soxr",
    "pooch",
    "mir_eval",
    "pretty_midi",
    "librosa",
    "audioread",
    "lazy_loader",
    "joblib",
    "sympy",
]


def _present() -> list[str]:
    out = []
    for name in HEAVY:
        if name in sys.modules or any(m.startswith(name + ".") for m in sys.modules):
            out.append(name)
    return sorted(out)


def test_report_import_surface() -> None:
    lines: list[str] = []

    import app.main  # noqa: F401

    lines.append(f"after import app.main:\n    {_present()}")

    from app import inference

    asyncio.run(inference.warm_up_basic_pitch())
    lines.append(f"after warm_up_basic_pitch():\n    {_present()}")

    # 2s A4 tone -> real transcription
    sr = 22050
    t = np.linspace(0.0, 2.0, int(sr * 2), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV")

    key_calls: list[str] = []
    _real_from_hist = inference._key_from_histogram

    def _spy_from_hist(hist):
        key_calls.append("histogram-fallback")
        return _real_from_hist(hist)

    inference._key_from_histogram = _spy_from_hist
    try:
        result = inference.run_basic_pitch(buf.getvalue(), "diag.wav")
    finally:
        inference._key_from_histogram = _real_from_hist

    lines.append(f"after run_basic_pitch():\n    {_present()}")
    lines.append(f"key='{result.get('key')}'  key_path={key_calls or ['music21.analyze']}")
    lines.append(f"TRANSCRIBE_CHUNKED={inference.TRANSCRIBE_CHUNKED}  CHUNK_SEC={inference.TRANSCRIBE_CHUNK_SEC}")

    raise AssertionError("DIAGNOSTIC REPORT:\n" + "\n".join(lines))
