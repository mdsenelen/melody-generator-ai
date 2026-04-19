from __future__ import annotations

from pathlib import Path

import torch


DEFAULT_CHORDS = [
    "C",
    "Cm",
    "D",
    "Dm",
    "E",
    "Em",
    "F",
    "G",
    "Am",
    "Bm",
    "G7",
    "Cmaj7",
    "Am7",
    "Dm7",
]

MODEL_PATH = Path(__file__).resolve().parent / "model" / "weights" / "web_model.pt"


def _load_chord_vocab() -> list[str]:
    if not MODEL_PATH.exists():
        return DEFAULT_CHORDS

    try:  # pragma: no cover
        model_data = torch.load(str(MODEL_PATH), map_location="cpu")
        chord_vocab = model_data.get("chord_vocab", [])
        if isinstance(chord_vocab, list) and chord_vocab:
            return chord_vocab
    except Exception:
        pass

    return DEFAULT_CHORDS


chord_vocab = _load_chord_vocab()


def get_chord_label(index: int) -> str | None:
    if 0 <= index < len(chord_vocab):
        return chord_vocab[index]
    return None


def get_all_chord_labels() -> list[str]:
    return chord_vocab
