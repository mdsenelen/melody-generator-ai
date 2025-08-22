# backend/app/chord_utils.py
import torch
import os

# Path to weights file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'final_vocal2accomp.pth')
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Load chord_vocab from weights file
_model_data = torch.load(MODEL_PATH, map_location='cpu')
chord_vocab = _model_data.get('chord_vocab', [])


def get_chord_label(index: int) -> str:
    """Map chord index to chord label."""
    if 0 <= index < len(chord_vocab):
        return chord_vocab[index]
    return None


def get_all_chord_labels() -> list:
    """Return all chord labels."""
    return chord_vocab
