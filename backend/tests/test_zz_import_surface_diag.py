"""TEMPORARY diagnostic (Phase B import audit) -- delete after reading CI output."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

SNIPPET = r"""
import resource, sys

def rss_mb():
    # linux ru_maxrss is KiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

base = rss_mb()
print(f"baseline rss: {base:.1f} MB", flush=True)

import music21  # noqa
after_m21 = rss_mb()
print(f"after 'import music21': {after_m21:.1f} MB  (delta +{after_m21-base:.1f})", flush=True)
print(f"  music21 submodules loaded: {sum(1 for m in sys.modules if m=='music21' or m.startswith('music21.'))}", flush=True)

import importlib
for name in ("numpy", "scipy", "numba", "librosa"):
    b = rss_mb(); importlib.import_module(name); a = rss_mb()
    print(f"after 'import {name}': {a:.1f} MB  (delta +{a-b:.1f})", flush=True)

b = rss_mb()
import app.main  # noqa
a = rss_mb()
print(f"after 'import app.main' (music21+libs already in): {a:.1f} MB  (delta +{a-b:.1f})", flush=True)

from app import inference
import asyncio
b = rss_mb(); asyncio.run(inference.warm_up_basic_pitch()); a = rss_mb()
print(f"after warm_up_basic_pitch: {a:.1f} MB  (delta +{a-b:.1f})", flush=True)
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
