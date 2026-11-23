"""Accuracy, robustness and streaming experiments. Each returns plain JSON-able dicts."""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf

from . import config, data, metrics, noise, prod

Log = Callable[[str], None]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def wav_bytes(signal: np.ndarray, sr: int) -> bytes:
    """32-bit float WAV, so degraded audio is never re-quantised or silently clipped."""
    buf = io.BytesIO()
    sf.write(buf, signal.astype(np.float32), sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def load_clip(clip_id: str) -> tuple[np.ndarray, int, list[dict], Path]:
    midi_path, wav_path = data.excerpt_paths(clip_id)
    audio, sr = sf.read(wav_path, dtype="float64")
    return audio, sr, data.read_ground_truth(midi_path), wav_path


def warm_up(clip_id: str) -> float:
    """First Basic Pitch call is slower (runtime compile); keep it out of the timed runs."""
    audio, sr, _, _ = load_clip(clip_id)
    t0 = time.perf_counter()
    prod.transcribe_chunked(wav_bytes(audio[: sr * 5], sr))
    return time.perf_counter() - t0


def run_robustness(clip_ids: Sequence[str], conditions: Sequence[dict], log: Log = _log) -> list[dict]:
    records: list[dict] = []
    for ci, clip_id in enumerate(clip_ids, 1):
        audio, sr, truth, _ = load_clip(clip_id)
        for cond in conditions:
            rng = noise.stable_rng(clip_id, cond["id"], seed=config.SEED)
            degraded, facts = noise.apply_condition(audio, sr, cond, rng)
            payload = wav_bytes(degraded, sr)
            t0 = time.perf_counter()
            notes, n_chunks = prod.transcribe_chunked(payload)
            seconds = time.perf_counter() - t0
            records.append(
                {"clip": clip_id, "condition": cond["id"], "seconds": seconds, "audio_sec": len(audio) / sr,
                 "n_chunks": n_chunks, **facts, **metrics.transcription_scores(truth, notes)}
            )
        log(f"[robustness] clip {ci}/{len(clip_ids)} ({clip_id}) done")
    return records


# ------------------------------------------------------------------ streaming

def _in_zones(notes: Sequence[dict], zones: Sequence[tuple[float, float]], pad: float = 0.0) -> list[dict]:
    return [n for n in notes if any(lo - pad <= n["start"] <= hi + pad for lo, hi in zones)]


def boundary_zones(duration_sec: float, chunk: dict[str, float]) -> list[tuple[float, float]]:
    """The overlap region of each adjacent chunk pair, where welding decisions are made."""
    import math

    n = max(1, math.ceil((duration_sec - chunk["overlap_sec"]) / chunk["hop_sec"]))
    return [(chunk["hop_sec"] * (k + 1), chunk["hop_sec"] * k + chunk["chunk_sec"]) for k in range(n - 1)]


def _summary(truth: Sequence[dict], notes: Sequence[dict], whole: Sequence[dict], zones: Sequence[tuple[float, float]]) -> dict:
    return {
        "n_notes": len(notes),
        "vs_truth": metrics.transcription_scores(truth, notes),
        "vs_whole_file": metrics.transcription_scores(whole, notes),
        "boundary_vs_truth": metrics.transcription_scores(_in_zones(truth, zones), _in_zones(notes, zones)),
    }


def run_streaming(clip_ids: Sequence[str], log: Log = _log) -> dict:
    chunk = prod.chunk_settings()
    default_eps = chunk["weld_edge_eps"]
    eps_values = sorted({default_eps, *config.WELD_EPS_ALTERNATIVES})
    clips: list[dict] = []
    for clip_id in clip_ids:
        audio, sr, truth, wav_path = load_clip(clip_id)
        payload = wav_bytes(audio, sr)
        duration = len(audio) / sr

        with prod.memoized_predictions() as cache_stats:
            t0 = time.perf_counter()
            with prod.weld_edge_eps(default_eps):
                chunked_notes, n_chunks = prod.transcribe_chunked(payload)
            chunked_seconds = time.perf_counter() - t0  # uncached: every chunk was a miss

            t0 = time.perf_counter()
            whole_notes = prod.transcribe_whole(str(wav_path))
            whole_seconds = time.perf_counter() - t0

            zones = boundary_zones(duration, chunk)
            sweep = []
            for eps in eps_values:
                with prod.weld_edge_eps(eps):
                    notes, _ = prod.transcribe_chunked(payload)  # network results come from the cache
                sweep.append({"edge_eps": eps, "is_production_default": eps == default_eps, **_summary(truth, notes, whole_notes, zones)})
            hits, misses = cache_stats["hits"], cache_stats["misses"]

        clips.append(
            {
                "clip": clip_id, "duration_sec": duration, "n_chunks": n_chunks, "n_boundaries": len(zones),
                "chunked": {"seconds": chunked_seconds, **_summary(truth, chunked_notes, whole_notes, zones)},
                "whole_file": {"seconds": whole_seconds, "n_notes": len(whole_notes),
                               "vs_truth": metrics.transcription_scores(truth, whole_notes),
                               "boundary_vs_truth": metrics.transcription_scores(_in_zones(truth, zones), _in_zones(whole_notes, zones))},
                "weld_sweep": sweep,
                "sweep_cache": {"network_runs": misses, "reused": hits},
            }
        )
        log(f"[streaming] {clip_id} done")
    return {"chunk_settings": chunk, "eps_values": eps_values, "clips": clips}
