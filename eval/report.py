"""Build eval/EVAL.md and eval/plots/*.png from a results JSON. Every number in the output
is computed here from that file; the only hand-written text is prose that contains none.

    python -m eval.report [--results eval/results/<file>.json]    (default: newest full run)"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

# Reference palette, categorical slots 1 and 2 (validated: all checks pass on this surface).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# ------------------------------------------------------------------ stats helpers

def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (n-1) across clips."""
    return st.fmean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def pm(values: Sequence[float], digits: int = 3) -> str:
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def latest_full_result() -> Path:
    files = sorted(p for p in config.RESULTS_DIR.glob("*.json") if not p.stem.endswith("_quick"))
    if not files:
        raise SystemExit("no full-run result found in eval/results (run: python -m eval.run --full)")
    return files[-1]


def by_condition(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r["condition"], []).append(r)
    return out


def metric(records: list[dict], kind: str, name: str) -> list[float]:
    return [r[kind][name] for r in records]


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def cost_row(c: dict) -> dict:
    warm = [r["wall_s"] for r in c["runs"][1:]] or [c["runs"][0]["wall_s"]]
    warm_cpu = [r["cpu_s"] / r["wall_s"] for r in c["runs"][1:]] or [c["runs"][0]["cpu_s"] / c["runs"][0]["wall_s"]]
    return {
        "label": f"{c['audio_sec']:.0f} s clip, {c['mode']}, threads {c['threads']}",
        "audio_sec": c["audio_sec"], "cold_s": c["runs"][0]["wall_s"], "warm_s": st.median(warm),
        "rtf": st.median(warm) / c["audio_sec"], "cpu_ratio": st.median(warm_cpu),
        "load_s": c["model_load_s"], "rss_load": c["rss_after_model_load_mb"], "peak": c["peak_rss_mb"],
    }


# ------------------------------------------------------------------ plots

def _finish(fig, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_f1_vs_snr(cfg: dict, groups: dict[str, list[dict]], path: Path) -> None:
    snrs = sorted(cfg["snrs"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (kind, title) in zip(axes, [("onset", "Onset F1 (50 ms)"), ("onset_offset", "Onset + offset F1")]):
        clean = st.fmean(metric(groups["clean"], kind, "f1"))
        ax.axhline(clean, color=INK_2, linestyle="--", linewidth=1)
        ax.annotate("clean", (min(snrs), clean), xytext=(0, 4), textcoords="offset points", ha="left", color=INK_2, fontsize=9)
        for noise_kind, color in (("white", BLUE), ("pink", ORANGE)):
            stats = [mean_std(metric(groups[f"{noise_kind}_snr{s}"], kind, "f1")) for s in snrs]
            m, sd = [a for a, _ in stats], [b for _, b in stats]
            ax.plot(snrs, m, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=f"{noise_kind} noise")
            ax.fill_between(snrs, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)], color=color, alpha=0.15, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("SNR (dB, higher is cleaner)")
        ax.set_xticks(snrs)
        ax.set_ylim(0, 1)
        ax.margins(x=0.12)
    axes[0].set_ylabel("F1, mean ± 1 std over clips")
    axes[0].legend(frameon=False, loc="lower right")
    _finish(fig, path)


def plot_f1_per_condition(cfg: dict, groups: dict[str, list[dict]], path: Path) -> None:
    ids = [c["id"] for c in cfg["conditions"]]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(ids) + 1.4))
    h = 0.38
    for offset, (kind, label, color) in enumerate([("onset", "onset F1", BLUE), ("onset_offset", "onset + offset F1", ORANGE)]):
        stats = [mean_std(metric(groups[i], kind, "f1")) for i in ids]
        ys = [n + (offset - 0.5) * (h + 0.04) for n in range(len(ids))]
        ax.barh(ys, [m for m, _ in stats], height=h, color=color, label=label, edgecolor=SURFACE, linewidth=1.5)
        ax.errorbar([m for m, _ in stats], ys, xerr=[s for _, s in stats], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
    ax.set_yticks(range(len(ids)), ids)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1, mean ± 1 std over clips")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    _finish(fig, path)


def plot_rtf(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.3))
    ax.barh(range(len(rows)), [r["rtf"] for r in rows], height=0.5, color=BLUE, edgecolor=SURFACE)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['rtf']:.4f}", (r["rtf"], i), xytext=(5, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("real-time factor (processing time / audio duration, lower is faster), warm median")
    ax.set_xlim(0, max(r["rtf"] for r in rows) * 1.2)
    ax.grid(axis="y", visible=False)
    _finish(fig, path)


# ------------------------------------------------------------------ document

def generate(results_path: Path) -> Path:
    res = json.loads(results_path.read_text())
    cfg, environment = res["config"], res["environment"]
    groups = by_condition(res["robustness"]["records"])
    cfg = {**cfg, "snrs": config.SNRS_DB}
    n_clips = len(cfg["clips"])
    cond_ids = [c["id"] for c in cfg["conditions"]]
    rel = results_path.relative_to(config.REPO_ROOT)

    plot_f1_vs_snr(cfg, groups, config.PLOTS_DIR / "f1_vs_snr.png")
    plot_f1_per_condition(cfg, groups, config.PLOTS_DIR / "f1_per_condition.png")
    cost_rows = [cost_row(c) for c in res["cost"]]
    plot_rtf(cost_rows, config.PLOTS_DIR / "rtf.png")

    # --- accuracy
    clean = groups["clean"]
    acc_rows = []
    for key, label in (("onset", "Onset only (50 ms)"), ("onset_offset", "Onset + offset")):
        acc_rows.append([label] + [pm(metric(clean, key, k)) for k in ("precision", "recall", "f1")])
    per_clip = [
        [r["clip"], str(r["n_ref"]), str(r["n_est"]), f"{r['onset']['f1']:.3f}", f"{r['onset_offset']['f1']:.3f}"]
        for r in clean
    ]

    # --- robustness
    clean_on = st.fmean(metric(clean, "onset", "f1"))
    clean_off = st.fmean(metric(clean, "onset_offset", "f1"))
    rob_rows = []
    for cid in cond_ids:
        g = groups[cid]
        on, off = metric(g, "onset", "f1"), metric(g, "onset_offset", "f1")
        extra = ""
        if "measured_snr_db" in g[0]:
            extra = f"{st.fmean(r['measured_snr_db'] for r in g):.2f} dB"
        elif "clipped_fraction" in g[0]:
            extra = f"{100 * st.fmean(r['clipped_fraction'] for r in g):.1f} % of samples clipped"
        rob_rows.append([cid, pm(on), f"{st.fmean(on) - clean_on:+.3f}", pm(off), f"{st.fmean(off) - clean_off:+.3f}", extra or "n/a"])
    worst = min((c for c in cond_ids if c != "clean"), key=lambda c: st.fmean(metric(groups[c], "onset", "f1")))

    # --- streaming
    sclips = res["streaming"]["clips"]
    chunk = res["streaming"]["chunk_settings"]
    stream_rows = []
    for c in sclips:
        stream_rows.append([
            c["clip"], f"{c['duration_sec']:.0f}", str(c["n_chunks"]),
            f"{c['chunked']['vs_truth']['onset']['f1']:.3f}", f"{c['whole_file']['vs_truth']['onset']['f1']:.3f}",
            f"{c['chunked']['vs_truth']['onset_offset']['f1']:.3f}", f"{c['whole_file']['vs_truth']['onset_offset']['f1']:.3f}",
            f"{c['chunked']['vs_whole_file']['onset']['f1']:.3f}",
            f"{c['chunked']['n_notes']} / {c['whole_file']['n_notes']}",
            f"{c['chunked']['seconds']:.2f} / {c['whole_file']['seconds']:.2f}",
        ])
    d_on = [c["chunked"]["vs_truth"]["onset"]["f1"] - c["whole_file"]["vs_truth"]["onset"]["f1"] for c in sclips]
    d_off = [c["chunked"]["vs_truth"]["onset_offset"]["f1"] - c["whole_file"]["vs_truth"]["onset_offset"]["f1"] for c in sclips]
    agree = [c["chunked"]["vs_whole_file"]["onset"]["f1"] for c in sclips]

    sweep_rows, sweep_signature = [], []
    for i, eps in enumerate(res["streaming"]["eps_values"]):
        entries = [c["weld_sweep"][i] for c in sclips]
        sweep_signature.append(tuple(e["n_notes"] for e in entries) + tuple(round(e["vs_truth"]["onset_offset"]["f1"], 9) for e in entries))
        sweep_rows.append([
            f"{eps:g}" + (" (production default)" if entries[0]["is_production_default"] else ""),
            str(sum(e["n_notes"] for e in entries)),
            f"{st.fmean(e['vs_truth']['onset']['f1'] for e in entries):.4f}",
            f"{st.fmean(e['vs_truth']['onset_offset']['f1'] for e in entries):.4f}",
            f"{st.fmean(e['boundary_vs_truth']['onset_offset']['f1'] for e in entries):.4f}",
            f"{sum(e['boundary_vs_truth']['n_ref'] for e in entries)}",
        ])
    sweep_flat = len(set(sweep_signature)) == 1
    n_boundaries = sum(c["n_boundaries"] for c in sclips)

    # --- cost
    cost_tbl = [
        [r["label"], f"{r['load_s']:.2f}", f"{r['cold_s']:.2f}", f"{r['warm_s']:.2f}", f"{r['rtf']:.4f}", f"{r['cpu_ratio']:.2f}",
         f"{r['rss_load']:.0f}", f"{r['peak']:.0f}", f"{r['peak'] - r['rss_load']:.0f}"]
        for r in cost_rows
    ]
    r30 = next(r for r in cost_rows if r["audio_sec"] <= config.CLIP_SEC + 1 and "threads default" in r["label"])

    # --- environment
    libs = ", ".join(f"{k} {v}" for k, v in environment["libraries"].items())
    dirty = environment["git"]["dirty"]
    m = environment["machine"]

    md = f"""# Melodia evaluation

Numbers below are generated from `{rel}` by `python -m eval.report`. Nothing in this file is typed by hand.
{"**This result was produced from a working tree with uncommitted changes; treat it as a draft.**" if dirty else ""}
{"**This is a quick smoke-test result (few clips), not the full evaluation.**" if res["mode"] == "quick" else ""}

## Summary

- Transcription accuracy on {n_clips} synthesized MAESTRO excerpts of {cfg['clip_sec']:.0f} s: onset F1 **{pm(metric(clean, 'onset', 'f1'))}**, onset+offset F1 **{pm(metric(clean, 'onset_offset', 'f1'))}** (mean ± std over clips).
- Weakest condition by onset F1 is `{worst}` at {st.fmean(metric(groups[worst], 'onset', 'f1')):.3f}, a change of {st.fmean(metric(groups[worst], 'onset', 'f1')) - clean_on:+.3f} against clean.
- Chunked transcription (30 s windows, 4 s overlap, boundary welding) versus whole-file on {len(sclips)} long excerpts: onset F1 change {st.fmean(d_on):+.4f}, onset+offset F1 change {st.fmean(d_off):+.4f}; the two outputs agree with each other at onset F1 {st.fmean(agree):.3f}.
- Boundary weld epsilon sweep: {"no value changed any note or any score on these excerpts (see the limits of that statement below)." if sweep_flat else "values changed the output; see the table."}
- Speed: real-time factor {r30['rtf']:.4f} for a {r30['audio_sec']:.0f} s clip (warm, default threads), on {m['cpu']}.

## Setup

**Data.** MAESTRO v3.0.0 MIDI-only archive (SHA-256 `{res['dataset']['archive_sha256'][:16]}...`), `{res['dataset']['split']}` split. {n_clips} excerpts of {cfg['clip_sec']:.0f} s from distinct pieces, chosen with a fixed seed ({cfg['seed']}) and a minimum note density; plus {len(cfg['long_clips'])} excerpts of {cfg['long_clip_sec']:.0f} s from other pieces for the streaming comparison. The selection is committed as `eval/manifest.json` and is reproduced byte for byte by the code (tested).

**Audio.** Each excerpt MIDI is rendered with the FluidSynth CLI ({environment['fluidsynth']}) and `{environment['soundfont']['file']}` to {cfg['render_sr']} Hz mono, FluidSynth reverb and chorus off. Sustain pedal is stripped from the excerpt MIDI, so note-off times in the ground truth match the audio. The ground truth is read back from the exact MIDI file that was rendered.

**System under test.** The production code path in `backend/app/inference.py`, imported and not copied: `_transcribe_full_chunked` for the chunked path (chunk {chunk['chunk_sec']:g} s, overlap {chunk['overlap_sec']:g} s, `_merge_chunk_notes` welding) and `_run_basic_pitch_predict` for the whole-file path. Basic Pitch runtime in this run: `{environment['basic_pitch_runtime']['backend']}` with `{environment['basic_pitch_runtime']['model_path']}`. The harness refuses to run if Basic Pitch fails to import, because production would silently return zero notes in that case.

**Metrics.** `mir_eval.transcription`, pitch tolerance 50 cents, onset tolerance {cfg['onset_tolerance_sec'] * 1000:.0f} ms. Onset+offset additionally requires the offset within max(50 ms, {cfg['offset_ratio'] * 100:.0f} % of the reference note length). Mean and sample standard deviation (n-1) are taken across clips.

**Degradations.** White and pink noise at SNR {", ".join(str(s) for s in config.SNRS_DB)} dB (clip-level average power; the noise is scaled so the realised SNR is exact, and a test asserts it is within 0.1 dB); synthetic reverb as exponentially decaying noise impulse responses with RT60 {" and ".join(str(r) for r in config.RT60S_SEC)} s (fully wet, RMS matched to the dry clip); hard clipping at {config.CLIP_PEAK_RATIO:g} of the clip's own peak. Degraded audio is passed as 32-bit float WAV so it is never re-quantised.

**Environment.**

{table(["Item", "Value"], [
    ["Machine", f"{m['cpu']}, {m['physical_cores']} physical / {m['logical_cores']} logical cores, {m['ram_gb']} GB RAM, {m['arch']}"],
    ["OS", environment["os"]],
    ["Python", environment["python"]],
    ["Libraries", libs],
    ["Git", f"{environment['git']['sha']} on {environment['git']['branch']}" + (" (dirty)" if dirty else "")],
    ["Run", f"{environment['timestamp_utc']}, wall clock {res['wall_clock_s']:.0f} s"],
]).replace("---:", "---")}

**Reproduce.**

```
python3 -m venv eval/.venv && eval/.venv/bin/pip install -r eval/requirements-eval.txt
brew install fluidsynth            # or: apt install fluidsynth
eval/.venv/bin/python -m eval.run --quick    # 3 clips, about a minute
eval/.venv/bin/python -m eval.run --full     # all clips, writes eval/results/*.json and regenerates this file
eval/.venv/bin/python -m pytest eval/tests
```

## Metrics

### Accuracy on clean audio ({n_clips} clips)

{table(["Matching", "Precision", "Recall", "F1"], acc_rows)}

<details><summary>Per clip</summary>

{table(["Clip", "Reference notes", "Estimated notes", "Onset F1", "Onset+offset F1"], per_clip)}

</details>

### Robustness (F1, mean ± std over {n_clips} clips)

{table(["Condition", "Onset F1", "vs clean", "Onset+offset F1", "vs clean", "Realised"], rob_rows)}

![F1 versus SNR](plots/f1_vs_snr.png)

![F1 per condition](plots/f1_per_condition.png)

### Streaming cost: chunked versus whole-file ({len(sclips)} excerpts of {cfg['long_clip_sec']:.0f} s)

{table(["Clip", "Seconds", "Chunks", "Onset F1 chunked", "Onset F1 whole", "On+off F1 chunked", "On+off F1 whole", "Chunked vs whole (onset F1)", "Notes chunked / whole", "Time s chunked / whole"], stream_rows)}

Chunked time includes the overlap, so it processes more audio than the whole-file pass. The "chunked vs whole" column scores the chunked output against the whole-file output as if the latter were the reference.

### Boundary weld epsilon sweep

`edge_eps` is the tolerance in `_merge_chunk_notes` for deciding that a note in the previous chunk was cut by that chunk's right edge. The network ran once per chunk; each epsilon re-ran the real merge on the same chunk outputs. The boundary columns score only notes whose onset falls in the overlap zones ({n_boundaries} zones across the excerpts).

{table(["edge_eps (s)", "Total notes", "Onset F1", "On+off F1", "Boundary on+off F1", "Reference notes in zones"], sweep_rows)}

{"All rows are identical, so on these excerpts no note landed in the window where the epsilon matters. That shows the current value is not hurting here; it does not show that a larger value is safe or useful, because the sample has few boundaries." if sweep_flat else "Rows differ, so the epsilon changes which notes are welded."}

### Compute cost

Each row is a fresh subprocess. Cold is the first transcription after model load; warm is the median of the following runs. CPU/wall is process CPU time divided by wall time on warm runs, a direct measure of how many cores were actually busy. RSS is in MB; "peak" is `ru_maxrss` of the whole process, "added" is peak minus RSS after the model finished loading.

{table(["Configuration", "Model load s", "Cold s", "Warm s", "RTF", "CPU/wall", "RSS after load", "Peak RSS", "Added"], cost_tbl)}

![Real-time factor](plots/rtf.png)

Thread-limit rows set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` and `NUMBA_NUM_THREADS` to 1. Compare the CPU/wall column between the two settings to see whether the cap changed anything on this backend.

## Limitations

- **Synthesized piano is easier than real recordings.** A sampled piano rendered dry has no microphone, room, pedal resonance, string noise or inter-note bleed. These scores are an upper bound on what real recordings would give and should not be read as real-world accuracy.
- **Small sample.** Few excerpts from a single instrument and a single dataset split; standard deviations are across clips and the confidence intervals are wide. The streaming comparison uses only a handful of long excerpts and a few chunk boundaries each.
- **Sustain pedal removed.** The ground truth uses key-release times, and the audio was rendered without pedal so they agree. Real MAESTRO audio is pedalled, which changes what an offset means.
- **Offset scores are pessimistic for rendered audio.** The soundfont's release tails keep sounding after note-off, so estimated offsets sit later than the MIDI offsets by construction.
- **Different inference backend from production.** This run used the Basic Pitch runtime listed above. Production on Linux uses TFLite. Outputs should be close but are not guaranteed identical, and the timing and memory numbers are not comparable to the Render deployment.
- **Memory numbers are macOS numbers.** `coremltools` pulls in PyTorch on macOS, so absolute RSS includes it. Compare the "added" column between configurations, not the absolute value against Linux figures.
- **SNR is clip-level.** Quiet passages see a worse local SNR than the headline value. The reverb is a synthetic noise impulse response, not a measured room.
- **Offline model, not a streaming one.** Basic Pitch needs whole windows of audio. Chunking bounds memory, but latency is the window length, so this is not evidence of real-time on-device behaviour in the causal sense.
- **Single machine, single run.** Timings are from one machine, and cold and warm effects, thermal state and background load are not controlled beyond using fresh subprocesses.

## What I would try next

- Evaluate on real MAESTRO audio and on a non-piano set (guitar, voice, mixed music) to measure the synthetic-to-real gap directly.
- Run the same harness on a Linux x86 machine with the TFLite runtime, so the numbers match what is deployed, and compare the CoreML and TFLite outputs note by note.
- Export the model to ONNX or quantize it and compare accuracy, size and latency, since on-device deployment is the real target.
- Replace windowed chunking with a causal, low-latency formulation and measure latency to first note, not just throughput.
- Add measured room impulse responses, codec artifacts and band-limited or low-bitrate inputs to the robustness set.
- Report confidence intervals by bootstrap over clips and add more boundary-heavy excerpts so the weld sweep has statistical power.
- Add a regression gate: fail a pull request when clean F1 or real-time factor moves beyond the observed spread.
"""
    out = config.EVAL_DIR / "EVAL.md"
    out.write_text(md)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m eval.report")
    ap.add_argument("--results", type=Path, default=None)
    args = ap.parse_args()
    path = (args.results or latest_full_result()).resolve()
    print(f"wrote {generate(path)}")


if __name__ == "__main__":
    main()
