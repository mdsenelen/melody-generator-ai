"""Facts about the machine and library versions, stored in every result file."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psutil

from . import config, data

LIBRARIES = ["basic-pitch", "numpy", "scipy", "librosa", "soundfile", "pretty_midi", "mir_eval",
             "coremltools", "torch", "tensorflow", "onnxruntime", "tflite-runtime", "matplotlib"]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(config.REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()


def git_info() -> dict:
    dirty = _git("status", "--porcelain", "--", ".", ":!eval/results", ":!eval/plots", ":!eval/EVAL.md")
    return {"sha": _git("rev-parse", "--short", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(dirty)}


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        if out:
            return out
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def capture(backend: dict[str, str]) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_info(),
        "machine": {
            "cpu": _cpu_name(),
            "arch": platform.machine(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        },
        "os": f"{platform.system()} {platform.release()} ({platform.platform()})",
        "python": platform.python_version(),
        "libraries": {name: v for name in LIBRARIES if (v := _version(name)) is not None},
        "basic_pitch_runtime": backend,
        "fluidsynth": data.fluidsynth_version(),
        "soundfont": {"file": config.SOUNDFONT_PATH.name, "sha256": data.sha256_file(config.SOUNDFONT_PATH)},
        "manifest_sha256": hashlib.sha256(config.MANIFEST_PATH.read_bytes()).hexdigest(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
