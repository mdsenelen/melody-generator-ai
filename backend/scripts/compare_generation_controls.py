"""Compare current variant generation against each opt-in control on its own
and a combined profile, on the same inputs, checkpoint and seeds.

Runs the production path (inference._build_variant_context ->
_generate_variant_tokens -> _variant_tokens_to_midi_bytes), so it needs the
joint checkpoint locally (backend/app/model/weights/joint_e2e_weights.pth).

Pass --notes with a JSON list of {pitch, start, end, velocity} per --input to
use the job_id path (known_note_events), exactly as the frontend does after
/analyse; without it the audio is transcribed in-process, which silently falls
back to the pyin heuristic when no Basic Pitch runtime is installed.

    cd backend
    python scripts/compare_generation_controls.py \
        --input clip.wav --notes clip.notes.json --out /tmp/gen_controls

Writes every variant as MIDI under <out>/<input>/<profile>/, plus
report.json and report.md. The numbers describe structure (validity, range,
grid alignment, similarity) -- they are not a measure of musical quality.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402

from app import inference  # noqa: E402
from app.generation_controls import (  # noqa: E402
    GenerationControls,
    grammar_violations,
    quantize_notes,
)

PROFILES: dict[str, GenerationControls] = {
    "current": GenerationControls(),
    "latent_zero": GenerationControls(latent_mode="zero"),
    "latent_scaled_0.25": GenerationControls(latent_mode="scaled", latent_scale=0.25),
    "grammar": GenerationControls(grammar_mask=True),
    "pitch_margin_12": GenerationControls(pitch_margin=12),
    "length_match_notes": GenerationControls(length_mode="match_notes"),
    "length_match_duration": GenerationControls(length_mode="match_duration"),
    "quantize_16th": GenerationControls(quantize_subdivision=4),
    "combined": GenerationControls(
        latent_mode="scaled",
        latent_scale=0.25,
        grammar_mask=True,
        pitch_margin=12,
        length_mode="match_duration",
        quantize_subdivision=4,
    ),
}

MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)
NOTE_INDEX = {name: i for i, name in enumerate(inference.PITCH_CLASS_NAMES)}


def _scale_for(seed_pitches: list[int]) -> set[int]:
    histogram = np.bincount([p % 12 for p in seed_pitches], minlength=12).astype(float)
    key = inference._key_from_histogram((histogram / max(histogram.sum(), 1)).tolist())
    tonic_name, _, mode = key.partition(" ")
    tonic = NOTE_INDEX.get(tonic_name, 0)
    steps = MINOR_STEPS if "minor" in mode else MAJOR_STEPS
    return {(tonic + s) % 12 for s in steps}


def _pc_hist(pitches: list[int]) -> np.ndarray:
    hist = np.bincount([p % 12 for p in pitches], minlength=12).astype(float)
    return hist / hist.sum() if hist.sum() else hist


def _edit_distance(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def variant_metrics(
    midi_bytes: bytes, tokens: list[int], context: dict[str, Any], scale: set[int]
) -> dict[str, Any]:
    try:
        midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        notes = sorted((n for inst in midi.instruments for n in inst.notes), key=lambda n: n.start)
        readable = True
    except Exception:
        notes, readable = [], False
    pitches = [n.pitch for n in notes]
    seed = context["seed_pitches"] or [60]
    low, high = min(seed) - 12, max(seed) + 12
    intervals = np.abs(np.diff(pitches)) if len(pitches) > 1 else np.array([0])
    grid = 60.0 / context["tempo_bpm"] / 4
    on_grid = [abs(n.start / grid - round(n.start / grid)) < 0.1 for n in notes]
    return {
        "valid_midi": readable and bool(notes),
        "grammar_violations": grammar_violations(tokens),
        "n_notes": len(notes),
        "duration_sec": notes[-1].end if notes else 0.0,
        "pitch_min": min(pitches) if pitches else None,
        "pitch_max": max(pitches) if pitches else None,
        "outside_input_pm12": float(np.mean([not (low <= p <= high) for p in pitches])) if pitches else 0.0,
        "in_key": float(np.mean([p % 12 in scale for p in pitches])) if pitches else 0.0,
        "mean_abs_interval": float(intervals.mean()),
        "leaps_gt7": float(np.mean(intervals > 7)),
        "onsets_on_16th_grid": float(np.mean(on_grid)) if on_grid else 0.0,
        "pc_overlap_with_input": float(np.minimum(_pc_hist(pitches), _pc_hist(seed)).sum()),
        "pitches": pitches,
    }


def input_rhythm_shift(context: dict[str, Any], seed_notes: list[tuple[int, float, float]]) -> dict[str, float]:
    """How far quantizing the *input* melody to the estimated-tempo grid would
    move its onsets -- a proxy for how much the grid fights the input rhythm."""
    if not seed_notes:
        return {"mean_shift_ms": 0.0, "moved_over_30ms": 0.0}
    quantized = quantize_notes(seed_notes, context["tempo_bpm"], 4)
    shifts = [abs(q[1] - s[1]) * 1000 for q, s in zip(quantized, seed_notes)]
    return {"mean_shift_ms": float(mean(shifts)), "moved_over_30ms": float(np.mean([s > 30 for s in shifts]))}


def run(
    inputs: list[Path],
    notes: list[Path],
    out_dir: Path,
    seeds: list[int],
    temperatures: list[float],
) -> dict[str, Any]:
    report: dict[str, Any] = {"seeds": seeds, "temperatures": temperatures, "inputs": {}}
    for index, input_path in enumerate(inputs):
        raw = input_path.read_bytes()
        known = json.loads(notes[index].read_text()) if index < len(notes) else None
        context = inference._build_variant_context(raw, known_note_events=known)
        midi_bytes = context["transcription"]["midi_bytes"]
        seed_midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        seed_notes = sorted(
            ((n.pitch, n.start, n.end) for inst in seed_midi.instruments for n in inst.notes),
            key=lambda n: n[1],
        )[:64]
        scale = _scale_for(context["seed_pitches"] or [60])
        input_report: dict[str, Any] = {
            "seed_note_count": context["seed_note_count"],
            "seed_duration_sec": round(context["seed_duration_sec"], 2),
            "seed_pitch_range": [min(context["seed_pitches"], default=None),
                                 max(context["seed_pitches"], default=None)],
            "tempo_bpm": round(context["tempo_bpm"], 1),
            "mood_label": context["transcription"]["mood_label"],
            "seed_source": "known_note_events" if known is not None else "in_process_transcription",
            "quantize_input_shift": input_rhythm_shift(context, seed_notes),
            "profiles": {},
        }
        (out_dir / input_path.stem).mkdir(parents=True, exist_ok=True)
        (out_dir / input_path.stem / "input_transcription.mid").write_bytes(midi_bytes)

        for name, controls in PROFILES.items():
            profile_dir = out_dir / input_path.stem / name
            profile_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for seed in seeds:
                generator = inference._seed_generator(seed)
                for temp in temperatures:
                    tokens = inference._generate_variant_tokens(context, temp, generator, controls)
                    midi = inference._variant_tokens_to_midi_bytes(context, tokens, controls)
                    (profile_dir / f"seed{seed}_T{temp}.mid").write_bytes(midi)
                    rows.append(variant_metrics(midi, tokens, context, scale))

            pitch_seqs = [r.pop("pitches") for r in rows]
            pairs = list(itertools.combinations(pitch_seqs, 2))
            summary = {
                key: round(float(np.mean([r[key] for r in rows if r[key] is not None])), 3)
                for key in rows[0]
                if key not in ("pitch_min", "pitch_max")
            }
            summary["pitch_min"] = min(r["pitch_min"] for r in rows if r["pitch_min"] is not None)
            summary["pitch_max"] = max(r["pitch_max"] for r in rows if r["pitch_max"] is not None)
            summary["pairwise_pitch_edit_distance"] = round(
                float(np.mean([_edit_distance(a, b) for a, b in pairs])), 3)
            summary["pairwise_pc_overlap"] = round(
                float(np.mean([np.minimum(_pc_hist(a), _pc_hist(b)).sum() for a, b in pairs])), 3)
            summary["duplicate_variants"] = len(pitch_seqs) - len({tuple(p) for p in pitch_seqs})
            summary["n_variants"] = len(rows)
            input_report["profiles"][name] = summary
        report["inputs"][input_path.stem] = input_report
    return report


COLUMNS = [
    ("valid_midi", "geçerli"), ("grammar_violations", "gramer ihl."), ("n_notes", "nota"),
    ("duration_sec", "süre s"), ("pitch_min", "min"), ("pitch_max", "max"),
    ("outside_input_pm12", "aralık dışı"), ("in_key", "tonda"), ("mean_abs_interval", "|aralık|"),
    ("leaps_gt7", ">5'li"), ("onsets_on_16th_grid", "16'lık ızgara"),
    ("pc_overlap_with_input", "girdiye benzerlik"), ("pairwise_pitch_edit_distance", "varyant farkı"),
    ("duplicate_variants", "kopya"),
]


def to_markdown(report: dict[str, Any]) -> str:
    lines = [f"Seeds: {report['seeds']} · temperatures: {report['temperatures']}", ""]
    for name, data in report["inputs"].items():
        lines += [
            f"## {name}",
            f"girdi: {data['seed_note_count']} nota, {data['seed_duration_sec']} s, perde "
            f"{data['seed_pitch_range']}, tahmini tempo {data['tempo_bpm']} BPM, mood "
            f"{data['mood_label']} ({data['seed_source']}); girdiyi 16'lık ızgaraya nicemleme: ortalama "
            f"{data['quantize_input_shift']['mean_shift_ms']:.0f} ms kayma, notaların "
            f"%{100 * data['quantize_input_shift']['moved_over_30ms']:.0f}'i >30 ms",
            "",
            "| profil | " + " | ".join(label for _, label in COLUMNS) + " |",
            "|---" * (len(COLUMNS) + 1) + "|",
        ]
        for profile, summary in data["profiles"].items():
            lines.append(f"| {profile} | " + " | ".join(str(summary[key]) for key, _ in COLUMNS) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--notes", action="append", default=[], type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.7, 0.9, 1.0, 1.3])
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    args.out.mkdir(parents=True, exist_ok=True)
    report = run(args.input, args.notes, args.out, args.seeds, args.temperatures)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    (args.out / "report.md").write_text(to_markdown(report))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
