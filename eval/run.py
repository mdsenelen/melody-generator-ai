"""Entry point:  python -m eval.run --quick   |   python -m eval.run --full

--quick  3 clips, every condition, 1 long clip, 3 cost workers.   Smoke test, under 2 minutes.
--full   20 clips, every condition, 3 long clips, 5 cost workers. Writes the numbers EVAL.md is built from.
Every run writes eval/results/<UTC timestamp>_<git sha>[_quick].json."""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import config, cost, data, env, experiments, noise, prod


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def build_plan(quick: bool, manifest: dict) -> dict:
    clips = [c["id"] for c in manifest["clips"]]
    longs = [c["id"] for c in manifest["long_clips"]]
    if quick:
        clips, longs = clips[: config.QUICK_N_CLIPS], longs[:1]
    cost_plan = [(clips[0], "chunked", "1"), (clips[0], "chunked", "default"), (longs[0], "chunked", "default")]
    if not quick:
        cost_plan += [(longs[0], "chunked", "1"), (longs[0], "whole", "default")]
    return {
        "clips": clips,
        "long_clips": longs,
        "conditions": noise.condition_list(config.SNRS_DB, config.RT60S_SEC, config.CLIP_PEAK_RATIO),
        "cost_plan": [list(p) for p in cost_plan],
        "cost_repeats": 1 if quick else 3,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.run")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--no-report", action="store_true", help="skip regenerating EVAL.md and plots after a --full run")
    args = ap.parse_args(argv)
    quick = args.quick
    started = time.perf_counter()

    backend = prod.require_real_basic_pitch()  # refuses to run on a silent fallback
    manifest = data.prepare()
    plan = build_plan(quick, manifest)
    truth_env = env.capture(backend)
    log(f"[run] mode={'quick' if quick else 'full'} backend={backend['backend']} clips={len(plan['clips'])}")

    warmup = experiments.warm_up(plan["clips"][0])
    robustness = experiments.run_robustness(plan["clips"], plan["conditions"])
    streaming = experiments.run_streaming(plan["long_clips"])
    cost_results = cost.run_cost([tuple(p) for p in plan["cost_plan"]], plan["cost_repeats"])

    result = {
        "schema_version": 1,
        "mode": "quick" if quick else "full",
        "config": {
            "seed": config.SEED, "render_sr": config.RENDER_SR, "clip_sec": config.CLIP_SEC,
            "long_clip_sec": config.LONG_CLIP_SEC, "onset_tolerance_sec": config.ONSET_TOLERANCE_SEC,
            "offset_ratio": config.OFFSET_RATIO, "snr_definition": "clip-level average power ratio, noise scaled to the realised SNR",
            **plan,
        },
        "environment": truth_env,
        "warmup_s": warmup,
        "dataset": manifest["dataset"],
        "robustness": {"records": robustness},
        "streaming": streaming,
        "cost": cost_results,
        "wall_clock_s": time.perf_counter() - started,
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "_quick" if quick else ""
    path = config.RESULTS_DIR / f"{truth_env['timestamp_utc'].replace(':', '').replace('-', '')}_{truth_env['git']['sha']}{suffix}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    log(f"[run] wrote {path.relative_to(config.REPO_ROOT)} in {result['wall_clock_s']:.0f}s")

    if not quick and not args.no_report:
        from . import report

        report.generate(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
