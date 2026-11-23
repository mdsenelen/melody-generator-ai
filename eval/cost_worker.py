"""One isolated measurement process: python -m eval.cost_worker --wav X --mode chunked|whole --repeats N.

Runs in its own interpreter so ru_maxrss is that of this process alone. The parent
(eval.cost) sets the thread-limit environment before launch."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import psutil

PROC = psutil.Process()


def rss_mb() -> float:
    return PROC.memory_info().rss / 2**20


def peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 2**20 if sys.platform == "darwin" else raw / 2**10  # bytes on macOS, KiB on Linux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--mode", choices=["chunked", "whole"], required=True)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    out: dict = {"rss_start_mb": rss_mb()}
    from . import prod  # imports backend/app/inference.py
    out["rss_after_import_mb"] = rss_mb()

    t0 = time.perf_counter()
    prod.require_real_basic_pitch()
    out["model_load_s"] = time.perf_counter() - t0
    out["rss_after_model_load_mb"] = rss_mb()

    import soundfile as sf

    info = sf.info(args.wav)
    out["audio_sec"] = info.duration
    payload = open(args.wav, "rb").read()

    runs = []
    for _ in range(1 + args.repeats):  # first run is cold (runtime compile, page-in), the rest warm
        wall0, cpu0 = time.perf_counter(), time.process_time()
        if args.mode == "chunked":
            prod.transcribe_chunked(payload)
        else:
            prod.transcribe_whole(args.wav)
        runs.append({"wall_s": time.perf_counter() - wall0, "cpu_s": time.process_time() - cpu0})
    out["runs"] = runs
    out["rss_end_mb"] = rss_mb()
    out["peak_rss_mb"] = peak_rss_mb()
    out["max_threads_seen"] = PROC.num_threads()
    print("COST_JSON:" + json.dumps(out))


if __name__ == "__main__":
    main()
