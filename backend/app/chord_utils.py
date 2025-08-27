# backend/app/chord_utils.py
import torch
import os

# Path to weights file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'final_vocal2accomp.pth')
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Default chord vocabulary for development when model is not available
DEFAULT_CHORD_VOCAB = [
    "C", "Cm", "C7", "Cm7", "Cmaj7", "Cdim", "Caug",
    "D", "Dm", "D7", "Dm7", "Dmaj7", "Ddim", "Daug",
    "E", "Em", "E7", "Em7", "Emaj7", "Edim", "Eaug",
    "F", "Fm", "F7", "Fm7", "Fmaj7", "Fdim", "Faug",
    "G", "Gm", "G7", "Gm7", "Gmaj7", "Gdim", "Gaug",
    "A", "Am", "A7", "Am7", "Amaj7", "Adim", "Aaug",
    "B", "Bm", "B7", "Bm7", "Bmaj7", "Bdim", "Baug"
]

# Load chord_vocab from weights file with fallback
try:
    if os.path.exists(MODEL_PATH):
        _model_data = torch.load(MODEL_PATH, map_location='cpu')
        chord_vocab = _model_data.get('chord_vocab', DEFAULT_CHORD_VOCAB)
        print("✅ Chord vocabulary loaded from model file")
    else:
        chord_vocab = DEFAULT_CHORD_VOCAB
        print("⚠️ Model file not found, using default chord vocabulary for development")
except Exception as e:
    print(f"⚠️ Failed to load model file: {e}, using default chord vocabulary")
    chord_vocab = DEFAULT_CHORD_VOCAB


def get_chord_label(index: int) -> str:
    """Map chord index to chord label."""
    if 0 <= index < len(chord_vocab):
        return chord_vocab[index]
    return None


def get_all_chord_labels() -> list:
    """Return all chord labels."""
    return chord_vocab
