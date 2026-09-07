"""TEMPORARY diagnostic (Phase B import audit) -- delete after reading CI output.

Runs in a FRESH interpreter (clean sys.modules) so we see what the real boot
loads, not what pytest collection already dragged in. Reports heavy modules
resident at three points and which key-detection path runs.
Fails on purpose so pytest prints the captured report.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

SNIPPET = r"""
import io, sys
import numpy as np
import soundfile as sf

HEAVY = ["torch","music21","matplotlib","tensorflow","tflite_runtime",
    "onnxruntime","coremltools","sklearn","scipy","numba","llvmlite",
    "resampy","soxr","pooch","mir_eval","pretty_midi","librosa","audioread",
    "lazy_loader","joblib","sympy","numpy"]

def present():
    return sorted(n for n in HEAVY
                  if n in sys.modules or any(m.startswith(n+".") for m in sys.modules))

report = []

import app.main  # noqa
report.append(("after import app.main", present()))

import asyncio
from app import inference
asyncio.run(inference.warm_up_basic_pitch())
report.append(("after warm_up_basic_pitch", present()))

sr = 22050
t = np.linspace(0.0, 2.0, sr*2, endpoint=False)
tone = (0.2*np.sin(2*np.pi*440.0*t)).astype("float32")
buf = io.BytesIO(); sf.write(buf, tone, sr, format="WAV")

calls = []
real = inference._key_from_histogram
inference._key_from_histogram = lambda h: (calls.append("histogram-fallback"), real(h))[1]
res = inference.run_basic_pitch(buf.getvalue(), "diag.wav")
inference._key_from_histogram = real
report.append(("after run_basic_pitch", present()))

# which resampler does a real 44.1k->22.05k load use?
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    t2 = np.linspace(0.0, 1.0, 44100, endpoint=False)
    b2 = io.BytesIO(); sf.write(b2, (0.1*np.sin(2*np.pi*220*t2)).astype("float32"), 44100, format="WAV")
    import librosa
    librosa.load(b2, sr=22050, mono=True)
resampler_new = sorted(n for n in ["resampy","soxr","samplerate"]
                       if n in sys.modules or any(m.startswith(n+".") for m in sys.modules))

print("=== IMPORT SURFACE (fresh interpreter) ===")
for label, mods in report:
    print(f"{label}:")
    print(f"    {mods}")
print(f"key='{res.get('key')}'  key_path={calls or ['music21.analyze']}")
print(f"resampler modules loaded after a real 44.1k->22.05k librosa.load: {resampler_new}")
print(f"TRANSCRIBE_CHUNKED={inference.TRANSCRIBE_CHUNKED}  CHUNK_SEC={inference.TRANSCRIBE_CHUNK_SEC}")
"""


def test_report_import_surface() -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(SNIPPET)],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    raise AssertionError(
        "DIAGNOSTIC REPORT:\n"
        f"--- returncode {result.returncode} ---\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
    )
