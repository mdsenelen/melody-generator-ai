"""Thin wrapper over mir_eval.transcription so the rest of the harness deals in note dicts."""
from __future__ import annotations

from typing import Iterable, Mapping

import mir_eval.transcription as mt
import numpy as np

from . import config

MIN_NOTE_SEC = 1e-3  # mir_eval rejects zero or negative length intervals


def _hz(midi_pitch: np.ndarray) -> np.ndarray:
    return 440.0 * 2.0 ** ((midi_pitch.astype(np.float64) - 69.0) / 12.0)


def notes_to_arrays(notes: Iterable[Mapping[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted((float(n["start"]), float(n["end"]), float(n["pitch"])) for n in notes)
    if not rows:
        return np.zeros((0, 2)), np.zeros(0)
    arr = np.asarray(rows, dtype=np.float64)
    ends = np.maximum(arr[:, 1], arr[:, 0] + MIN_NOTE_SEC)
    return np.stack([arr[:, 0], ends], axis=1), _hz(arr[:, 2])


def transcription_scores(ref_notes: Iterable[Mapping[str, float]], est_notes: Iterable[Mapping[str, float]]) -> dict:
    """Precision, recall, F1 for onset-only and onset+offset matching.

    Pitch tolerance is 50 cents, onset tolerance 50 ms. Notes are matched
    one-to-one by mir_eval (maximum bipartite matching). An empty estimate or
    reference scores 0 across the board."""
    ref_i, ref_p = notes_to_arrays(ref_notes)
    est_i, est_p = notes_to_arrays(est_notes)

    def run(offset_ratio: float | None) -> dict:
        p, r, f, overlap = mt.precision_recall_f1_overlap(
            ref_i, ref_p, est_i, est_p,
            onset_tolerance=config.ONSET_TOLERANCE_SEC,
            pitch_tolerance=50.0,
            offset_ratio=offset_ratio,
            offset_min_tolerance=0.05,
        )
        return {"precision": float(p), "recall": float(r), "f1": float(f)}

    return {
        "n_ref": int(len(ref_p)),
        "n_est": int(len(est_p)),
        "onset": run(None),
        "onset_offset": run(config.OFFSET_RATIO),
    }
