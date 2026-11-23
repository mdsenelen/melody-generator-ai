"""Real-time factor and peak memory, each configuration in a fresh subprocess."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Sequence

from . import config, data

SINGLE_THREAD_ENV = {k: "1" for k in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")}


def measure(clip_id: str, mode: str, threads: str, repeats: int) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SINGLE_THREAD_ENV}
    if threads == "1":
        env.update(SINGLE_THREAD_ENV)
    _, wav = data.excerpt_paths(clip_id)
    proc = subprocess.run(
        [sys.executable, "-m", "eval.cost_worker", "--wav", str(wav), "--mode", mode, "--repeats", str(repeats)],
        cwd=config.REPO_ROOT, env=env, capture_output=True, text=True, timeout=1800,
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("COST_JSON:")), None)
    if proc.returncode != 0 or line is None:
        raise RuntimeError(f"cost worker failed ({clip_id}/{mode}/{threads}):\n{proc.stderr[-2000:]}")
    return {"clip": clip_id, "mode": mode, "threads": threads, "repeats": repeats, **json.loads(line[len("COST_JSON:"):])}


def run_cost(plan: Sequence[tuple[str, str, str]], repeats: int, log=lambda m: print(m, file=sys.stderr, flush=True)) -> list[dict]:
    results = []
    for clip_id, mode, threads in plan:
        results.append(measure(clip_id, mode, threads, repeats))
        log(f"[cost] {clip_id} {mode} threads={threads} done")
    return results
