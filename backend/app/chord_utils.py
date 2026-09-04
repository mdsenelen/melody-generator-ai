from __future__ import annotations

from pathlib import Path


DEFAULT_CHORDS = [
    # Major (all 12 roots, both spellings where conventional)
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
    # Minor (all 12 roots)
    "Cm", "C#m", "Dbm", "Dm", "D#m", "Ebm", "Em", "Fm", "F#m", "Gbm", "Gm", "G#m", "Abm", "Am", "A#m", "Bbm", "Bm",
    # Augmented (all 12 roots)
    "Caug", "C#aug", "Daug", "D#aug", "Ebaug", "Eaug", "Faug", "F#aug", "Gaug", "G#aug", "Abaug", "Aaug", "Bbaug", "Baug",
    # Diminished (all 12 roots)
    "Cdim", "C#dim", "Ddim", "D#dim", "Ebdim", "Edim", "Fdim", "F#dim", "Gdim", "G#dim", "Abdim", "Adim", "Bbdim", "Bdim",
    # Power / 5th chords (all 12 roots)
    "C5", "C#5", "D5", "D#5", "Eb5", "E5", "F5", "F#5", "G5", "G#5", "Ab5", "A5", "Bb5", "B5",
    # Dominant 7th (all 12 roots)
    "C7", "C#7", "Db7", "D7", "D#7", "Eb7", "E7", "F7", "F#7", "Gb7", "G7", "G#7", "Ab7", "A7", "A#7", "Bb7", "B7",
    # Major 7th (all 12 roots)
    "Cmaj7", "C#maj7", "Dbmaj7", "Dmaj7", "D#maj7", "Ebmaj7", "Emaj7", "Fmaj7", "F#maj7", "Gbmaj7", "Gmaj7", "G#maj7", "Abmaj7", "Amaj7", "A#maj7", "Bbmaj7", "Bmaj7",
    # Minor 7th (all 12 roots)
    "Cm7", "C#m7", "Dm7", "D#m7", "Ebm7", "Em7", "Fm7", "F#m7", "Gm7", "G#m7", "Abm7", "Am7", "Bbm7", "Bm7",
    # Sus2 (all 12 roots)
    "Csus2", "C#sus2", "Dsus2", "D#sus2", "Ebsus2", "Esus2", "Fsus2", "F#sus2", "Gsus2", "G#sus2", "Absus2", "Asus2", "Bbsus2", "Bsus2",
    # Sus4 (all 12 roots)
    "Csus4", "C#sus4", "Dsus4", "D#sus4", "Ebsus4", "Esus4", "Fsus4", "F#sus4", "Gsus4", "G#sus4", "Absus4", "Asus4", "Bbsus4", "Bsus4",
    # Dominant 9th
    "C9", "D9", "E9", "F9", "G9", "A9", "B9", "Bb9",
    # Major 9th
    "Cmaj9", "Dmaj9", "Emaj9", "Fmaj9", "Gmaj9", "Amaj9", "Bmaj9",
    # Minor 9th
    "Cm9", "Dm9", "Em9", "Fm9", "Gm9", "Am9", "Bm9",
    # Add9
    "Cadd9", "Dadd9", "Eadd9", "Fadd9", "Gadd9", "Aadd9", "Badd9",
]

MODEL_PATH = Path(__file__).resolve().parent / "model" / "weights" / "web_model.pt"


def _load_chord_vocab() -> list[str]:
    if not MODEL_PATH.exists():
        return DEFAULT_CHORDS

    try:  # pragma: no cover
        import torch  # local: only needed if a weights file is actually present

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
