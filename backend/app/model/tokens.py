"""Token / MIDI helpers and the mood heuristic.

Split out of ``colab_parity`` so callers that only need these -- most
importantly the transcription path in ``app.inference`` -- don't transitively
import ``torch`` (which costs ~150-250 MB RSS and is the difference between
fitting and OOM-ing on the 512 MiB deploy). ``colab_parity`` re-exports every
name defined here, so existing ``from .model.colab_parity import ...`` sites
keep working unchanged.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    import pretty_midi
except Exception:  # pragma: no cover - optional dependency
    pretty_midi = None


PITCH_OFFSET = 0
DUR_OFFSET = 128
TEMPO_OFFSET = 160
PAD_TOKEN = 176
VOCAB_SIZE = 177
DUR_BINS = [0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
TEMPO_BINS = list(range(40, 205, 10))
MOOD_LABELS = ["happy", "sad", "neutral"]


def tokens_to_chord_idx(token_seq: list[int]) -> int:
    pitches = [t for t in token_seq if 0 <= t < DUR_OFFSET]
    if not pitches:
        return 24
    hist = [0] * 12
    for p in pitches:
        hist[p % 12] += 1
    total = sum(hist)
    if total == 0:
        return 24
    hist_n = [h / total for h in hist]
    best_score = -1.0
    best_idx = 24
    for root in range(12):
        maj = hist_n[root] + 0.85 * hist_n[(root + 4) % 12] + 0.75 * hist_n[(root + 7) % 12]
        minor = hist_n[root] + 0.85 * hist_n[(root + 3) % 12] + 0.75 * hist_n[(root + 7) % 12]
        if maj > best_score:
            best_score = maj
            best_idx = root
        if minor > best_score:
            best_score = minor
            best_idx = root + 12
    if best_score < 0.55:
        return 24
    return best_idx


def tokens_to_bar_idx(token_seq: list[int], beats_per_bar: int = 4, n_bins: int = 8) -> int:
    beat_pos = 0.0
    for token in token_seq:
        if DUR_OFFSET <= token < TEMPO_OFFSET:
            dur_idx = token - DUR_OFFSET
            beat_pos += DUR_BINS[min(dur_idx, len(DUR_BINS) - 1)] * 2.0
    bar_frac = (beat_pos % beats_per_bar) / beats_per_bar
    return int(bar_frac * n_bins) % n_bins


def midi_to_tokens(midi_path: str | Path, max_notes: int = 64) -> list[int]:
    if pretty_midi is None:
        raise RuntimeError("pretty_midi is required for midi_to_tokens")
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        tempo = midi.estimate_tempo()
    except Exception:
        return []

    tokens = [TEMPO_OFFSET + min(bisect.bisect_left(TEMPO_BINS, tempo), 15)]
    notes = sorted(
        [note for instrument in midi.instruments if not instrument.is_drum for note in instrument.notes],
        key=lambda note: note.start,
    )
    for note in notes[:max_notes]:
        duration = note.end - note.start
        tokens.append(max(0, min(127, note.pitch)))
        tokens.append(DUR_OFFSET + min(bisect.bisect_left(DUR_BINS, duration), 31))
    return tokens


def tokens_to_midi(tokens: list[int], bpm: float = 120.0, out_path: str | Path = "/tmp/gen.mid") -> str:
    if pretty_midi is None:
        raise RuntimeError("pretty_midi is required for tokens_to_midi")
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)
    current_time = 0.0
    pending_pitch: Optional[int] = None
    for token in tokens:
        if PITCH_OFFSET <= token < DUR_OFFSET:
            pending_pitch = token - PITCH_OFFSET
        elif DUR_OFFSET <= token < TEMPO_OFFSET and pending_pitch is not None:
            duration = float(DUR_BINS[min(token - DUR_OFFSET, len(DUR_BINS) - 1)])
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=80,
                    pitch=pending_pitch,
                    start=current_time,
                    end=current_time + duration,
                )
            )
            current_time += duration
            pending_pitch = None
    midi.instruments.append(instrument)
    midi.write(str(out_path))
    return str(out_path)


def heuristic_mood_from_metrics(tempo_bpm: float, average_pitch: float, key_label: str = "") -> tuple[int, str]:
    # key_label may be "" when key detection failed; both flags are False then,
    # which gracefully falls back to the original pitch-only heuristic.
    is_minor = "minor" in key_label.lower()
    is_major = "major" in key_label.lower()

    # Fast + bright key (or high register) → happy
    if tempo_bpm > 110 and (is_major or average_pitch > 65):
        return 0, MOOD_LABELS[0]
    # Slow + dark key (or low register) → sad
    if tempo_bpm < 80 and (is_minor or average_pitch < 60):
        return 1, MOOD_LABELS[1]
    # Minor key at moderate tempo (80–100 BPM) still reads as melancholic
    if is_minor and tempo_bpm < 100:
        return 1, MOOD_LABELS[1]
    return 2, MOOD_LABELS[2]
