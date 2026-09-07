"""TEMPORARY diagnostic (Phase B import audit) -- delete after reading CI output.

Fresh interpreter (clean sys.modules). Prints incrementally so a late crash
can't swallow the report. Fails on purpose to surface captured stdout.
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
    "lazy_loader","joblib","sympy","numpy","samplerate"]

def show(label):
    mods = sorted(n for n in HEAVY
                  if n in sys.modules or any(m.startswith(n+".") for m in sys.modules))
    print(f"{label}: {mods}", flush=True)

show("[0] before any app import")
import app.main  # noqa
show("[1] after import app.main")

import asyncio
from app import inference
asyncio.run(inference.warm_up_basic_pitch())
show("[2] after warm_up_basic_pitch")

sr = 22050
t = np.linspace(0.0, 2.0, sr*2, endpoint=False)
tone = (0.2*np.sin(2*np.pi*440.0*t)).astype("float32")
raw = io.BytesIO(); sf.write(raw, tone, sr, format="WAV"); raw = raw.getvalue()

calls = []
_real = inference._key_from_histogram
inference._key_from_histogram = lambda h: (calls.append("hist"), _real(h))[1]
res = inference.run_basic_pitch(raw, "diag.wav")
inference._key_from_histogram = _real
show("[3] after run_basic_pitch")
print(f"key={res.get('key')!r}  key_path={calls or ['music21.analyze']}", flush=True)

# real 44.1k -> 22.05k resample: which backend?
before = set(sys.modules)
b2 = io.BytesIO()
sf.write(b2, (0.1*np.sin(2*np.pi*220*np.linspace(0,1,44100,endpoint=False))).astype("float32"), 44100, format="WAV")
import librosa
librosa.load(io.BytesIO(b2.getvalue()), sr=22050, mono=True)
newmods = sorted(m for m in set(sys.modules)-before if m.split('.')[0] in ("resampy","soxr","samplerate"))
print(f"resample backend newly imported by librosa.load(sr=): {newmods or 'none (already loaded or C-only)'}", flush=True)

# is resampy actually reachable / used, or just installed?
import importlib.util
print(f"resampy installed: {importlib.util.find_spec('resampy') is not None}", flush=True)
print(f"soxr installed: {importlib.util.find_spec('soxr') is not None}", flush=True)
print(f"TRANSCRIBE_CHUNKED={inference.TRANSCRIBE_CHUNKED}  CHUNK_SEC={inference.TRANSCRIBE_CHUNK_SEC}", flush=True)
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
        f"--- rc {result.returncode} ---\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr tail ---\n{result.stderr[-1500:]}"
    )
