"""TEMPORARY diagnostic (Phase B) -- delete after reading CI output.

Now that music21 is gone: (a) confirm the transcription path's RSS,
(b) check the per-transcription ratchet across repeated calls,
(c) A/B the chunked peak at TRANSCRIBE_CHUNK_SEC 30 vs 15 on a 90s clip.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

SNIPPET = r"""
import io, os, sys
import numpy as np, soundfile as sf

def rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0

def line(label):
    print(f"{label:44s} {rss():7.1f} MiB", flush=True)

line("interpreter start")
from app import inference
import asyncio
asyncio.run(inference.warm_up_basic_pitch())
line("after warm_up")

forbidden = [m for m in sys.modules if m.split('.')[0] in ("music21","matplotlib","sklearn","tensorflow")]
print(f"forbidden modules resident: {forbidden or 'none'}", flush=True)

sr = 22050

def clip(seconds):
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    sig = sum(0.15*np.sin(2*np.pi*f*t) for f in (220.0, 277.18, 329.63))
    b = io.BytesIO(); sf.write(b, sig.astype("float32"), sr, format="WAV")
    return b.getvalue()

short = clip(8)
for i in range(4):
    inference.run_basic_pitch(short, "d.wav")
    line(f"  after single-pass transcription #{i+1}")

# chunked A/B on a 90s clip
inference.TRANSCRIBE_CHUNKED = True
long_raw = clip(90)

for cs in (30.0, 15.0):
    inference.TRANSCRIBE_CHUNK_SEC = cs
    peak = [0.0]
    real = inference._run_basic_pitch_predict
    def traced(path, _real=real, _peak=peak):
        r = _real(path)
        _peak[0] = max(_peak[0], rss())
        return r
    inference._run_basic_pitch_predict = traced
    before = rss()
    res = inference.run_basic_pitch(long_raw, "long.wav")
    inference._run_basic_pitch_predict = real
    print(f"CHUNK_SEC={cs:4.0f}: n_chunks={res.get('n_chunks')} before={before:.1f} peak_during={peak[0]:.1f} after={rss():.1f} MiB", flush=True)
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
        f"--- stderr tail ---\n{result.stderr[-1500:]}"
    )
