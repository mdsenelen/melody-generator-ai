"""TEMPORARY diagnostic (Phase B import audit) -- delete after reading CI output."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

SNIPPET = r"""
import sys

def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0  # kB -> MiB
    return -1.0

def step(label, fn):
    b = rss_mb(); fn(); a = rss_mb()
    print(f"{label:42s} {a:7.1f} MiB   (+{a-b:5.1f})", flush=True)

print(f"{'interpreter start':42s} {rss_mb():7.1f} MiB", flush=True)

def _imp(n):
    import importlib; importlib.import_module(n)

step("import numpy", lambda: _imp("numpy"))
step("import music21 (ISOLATED cost)", lambda: _imp("music21"))
print(f"  music21 submodules: {sum(1 for m in sys.modules if m.split('.')[0]=='music21')}", flush=True)
step("import librosa", lambda: _imp("librosa"))
step("import scipy.signal", lambda: _imp("scipy.signal"))
step("import numba", lambda: _imp("numba"))
step("import basic_pitch.inference", lambda: _imp("basic_pitch.inference"))
step("import app.main", lambda: _imp("app.main"))

from app import inference
import asyncio, io
import numpy as np, soundfile as sf
step("warm_up_basic_pitch()", lambda: asyncio.run(inference.warm_up_basic_pitch()))

sr = 22050
raw = io.BytesIO()
sf.write(raw, (0.2*np.sin(2*np.pi*440*np.linspace(0,8,sr*8,endpoint=False))).astype("float32"), sr, format="WAV")
raw = raw.getvalue()
step("run_basic_pitch() 8s clip #1", lambda: inference.run_basic_pitch(raw, "d.wav"))
step("run_basic_pitch() 8s clip #2", lambda: inference.run_basic_pitch(raw, "d.wav"))
step("run_basic_pitch() 8s clip #3", lambda: inference.run_basic_pitch(raw, "d.wav"))
print(f"{'final':42s} {rss_mb():7.1f} MiB", flush=True)
"""


def test_report_import_surface() -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(SNIPPET)],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    raise AssertionError(
        f"DIAGNOSTIC REPORT (rc {result.returncode}):\n{result.stdout}\n"
        f"--- stderr tail ---\n{result.stderr[-1200:]}"
    )
