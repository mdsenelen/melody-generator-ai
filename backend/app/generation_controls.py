"""Opt-in, inference-only controls for CVAE variant generation.

Nothing here changes the trained model, the checkpoint or the tokenizer. Every
control is off in ``CURRENT_CONTROLS`` and ``generate_iddm_variants`` then runs
exactly the pre-existing code path (``decoder.sample`` + ``tokens_to_midi``),
so production output is byte-for-byte unchanged unless a caller opts in.

Token grammar, derived from what ``tokens.midi_to_tokens`` actually emits
(see tests/test_generation_controls.py for the consistency checks):

* position 0 is a fixed start token (a tempo token in training, the pitch-0
  placeholder ``decoder.sample`` seeds with at inference) -- never sampled;
* a pitch token (0-127) must be followed by a duration token;
* a duration token must be followed by a pitch token or PAD;
* only duration tokens DUR_OFFSET..DUR_OFFSET+len(DUR_BINS) are ever emitted
  by the tokenizer (``bisect_left`` returns 0..len(DUR_BINS));
* tempo tokens never appear after position 0;
* PAD ends the melody -- only PAD may follow it.

Torch is imported inside the functions that need it, so importing this module
from ``app.inference`` keeps the transcription path torch-free.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .model.tokens import DUR_BINS, DUR_OFFSET, PAD_TOKEN, TEMPO_OFFSET

try:  # pragma: no cover - optional dependency
    import pretty_midi
except Exception:  # pragma: no cover - optional dependency
    pretty_midi = None


LATENT_MODES = ("current", "zero", "scaled")
LENGTH_MODES = ("none", "match_notes", "match_duration")

PITCH_TOKENS = range(0, DUR_OFFSET)
# Exactly the duration tokens midi_to_tokens can produce: bisect_left over
# DUR_BINS returns 0..len(DUR_BINS) inclusive.
VALID_DUR_TOKENS = range(DUR_OFFSET, DUR_OFFSET + len(DUR_BINS) + 1)

_EXPECT_PITCH = "pitch"
_EXPECT_DURATION = "duration"
_ENDED = "ended"


@dataclass(frozen=True)
class GenerationControls:
    """Which inference-time controls to apply. The defaults reproduce the
    current production behaviour exactly."""

    # current: z = mu + alpha * delta_z (production)
    # zero:    z = mu (the actor's delta_z is still drawn so the RNG stream
    #          stays aligned with "current", but it is not used)
    # scaled:  z = mu + latent_scale * alpha * delta_z
    latent_mode: str = "current"
    latent_scale: float = 1.0
    grammar_mask: bool = False
    # Semitones beyond the input melody's lowest/highest pitch the generated
    # pitches may reach. None disables the restriction.
    pitch_margin: Optional[int] = None
    length_mode: str = "none"
    # Grid steps per beat (4 = sixteenth notes). None disables quantization.
    quantize_subdivision: Optional[int] = None

    def __post_init__(self) -> None:
        if self.latent_mode not in LATENT_MODES:
            raise ValueError(f"latent_mode must be one of {LATENT_MODES}")
        if self.length_mode not in LENGTH_MODES:
            raise ValueError(f"length_mode must be one of {LENGTH_MODES}")
        if self.latent_scale < 0:
            raise ValueError("latent_scale must be >= 0")
        if self.pitch_margin is not None and self.pitch_margin < 0:
            raise ValueError("pitch_margin must be >= 0")
        if self.quantize_subdivision is not None and self.quantize_subdivision < 1:
            raise ValueError("quantize_subdivision must be >= 1")

    @property
    def needs_custom_sampler(self) -> bool:
        return self.grammar_mask or self.pitch_margin is not None

    @property
    def needs_postprocessing(self) -> bool:
        return self.length_mode != "none" or self.quantize_subdivision is not None


CURRENT_CONTROLS = GenerationControls()


# --- latent ----------------------------------------------------------------


def latent_for_variant(base_mu: Any, alpha: Any, delta_z: Any, controls: GenerationControls) -> Any:
    if controls.latent_mode == "zero":
        return base_mu
    if controls.latent_mode == "scaled":
        return base_mu + controls.latent_scale * alpha * delta_z
    return base_mu + alpha * delta_z


# --- pitch range -----------------------------------------------------------


def pitch_range_for(seed_pitches: Sequence[int], margin: Optional[int]) -> Optional[tuple[int, int]]:
    """Allowed (low, high) pitch range around the input melody, clamped to the
    MIDI range the vocabulary covers. None when disabled or no input pitches."""
    if margin is None or not seed_pitches:
        return None
    low = max(0, min(seed_pitches) - margin)
    high = min(DUR_OFFSET - 1, max(seed_pitches) + margin)
    return low, high


# --- grammar ---------------------------------------------------------------


def _next_state(state: str, token: int) -> str:
    if state == _ENDED or token == PAD_TOKEN:
        return _ENDED
    if 0 <= token < DUR_OFFSET:
        return _EXPECT_DURATION
    return _EXPECT_PITCH


def allowed_tokens(
    state: str,
    vocab_size: int,
    pitch_range: Optional[tuple[int, int]],
    grammar: bool,
    notes_so_far: int,
) -> list[bool]:
    """Boolean mask over the vocabulary for the next token."""
    if not grammar:
        mask = [True] * vocab_size
        if pitch_range is not None:
            low, high = pitch_range
            for p in PITCH_TOKENS:
                if not (low <= p <= high):
                    mask[p] = False
        return mask

    mask = [False] * vocab_size
    if state == _ENDED:
        mask[PAD_TOKEN] = True
    elif state == _EXPECT_DURATION:
        for t in VALID_DUR_TOKENS:
            mask[t] = True
    else:
        low, high = pitch_range if pitch_range is not None else (0, DUR_OFFSET - 1)
        for p in range(low, high + 1):
            mask[p] = True
        # An empty melody is never a useful variant: PAD may only end it
        # once at least one complete note exists.
        if notes_so_far > 0:
            mask[PAD_TOKEN] = True
    return mask


def grammar_violations(tokens: Sequence[int]) -> int:
    """Count transitions that break the grammar above. Position 0 is the
    start token and is not checked."""
    violations = 0
    state = _EXPECT_PITCH
    for token in tokens[1:]:
        token = int(token)
        if state == _ENDED:
            if token != PAD_TOKEN:
                violations += 1
            continue
        if token == PAD_TOKEN:
            if state == _EXPECT_DURATION:
                violations += 1
            state = _ENDED
        elif state == _EXPECT_DURATION:
            if token not in VALID_DUR_TOKENS:
                violations += 1
                # resync on what was actually produced
                state = _EXPECT_DURATION if 0 <= token < DUR_OFFSET else _EXPECT_PITCH
            else:
                state = _EXPECT_PITCH
        else:
            if not (0 <= token < DUR_OFFSET):
                violations += 1
            else:
                state = _EXPECT_DURATION
    return violations


def sample_tokens(
    decoder: Any,
    z_cond: Any,
    seq_len: int,
    temperature: float,
    generator: Any,
    controls: GenerationControls,
    pitch_range: Optional[tuple[int, int]],
) -> Any:
    """Same loop as ``MelodyDecoder.sample`` with a logits mask applied before
    the softmax. Always runs all ``seq_len - 1`` steps (forcing PAD after the
    melody ends) so the RNG stream stays aligned with the unmasked sampler."""
    import torch
    import torch.nn.functional as F

    batch_size = z_cond.size(0)
    device = z_cond.device
    vocab_size = decoder.out.out_features
    hidden = torch.tanh(decoder.fc_z(z_cond)).unsqueeze(0)
    current = torch.zeros(batch_size, dtype=torch.long, device=device)
    outputs = [current]
    states = [_EXPECT_PITCH] * batch_size
    notes = [0] * batch_size
    for _ in range(seq_len - 1):
        embedded = decoder.emb(current).unsqueeze(1)
        context = hidden.squeeze(0).unsqueeze(1).expand(-1, 1, -1)
        output, hidden = decoder.gru(torch.cat([embedded, context], dim=-1), hidden)
        logits = decoder.out(output.squeeze(1)) / max(temperature, 1e-4)
        mask = torch.tensor(
            [
                allowed_tokens(states[b], vocab_size, pitch_range, controls.grammar_mask, notes[b])
                for b in range(batch_size)
            ],
            dtype=torch.bool,
            device=device,
        )
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        current = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        for b in range(batch_size):
            token = int(current[b])
            if states[b] == _EXPECT_DURATION and token in VALID_DUR_TOKENS:
                notes[b] += 1
            states[b] = _next_state(states[b], token)
        outputs.append(current)
    return torch.stack(outputs, dim=1)


# --- tokens -> notes -> MIDI ----------------------------------------------


def tokens_to_notes(tokens: Sequence[int]) -> list[tuple[int, float, float]]:
    """(pitch, start, end) exactly as ``tokens.tokens_to_midi`` places them."""
    notes: list[tuple[int, float, float]] = []
    current_time = 0.0
    pending: Optional[int] = None
    for token in tokens:
        token = int(token)
        if 0 <= token < DUR_OFFSET:
            pending = token
        elif DUR_OFFSET <= token < TEMPO_OFFSET and pending is not None:
            duration = float(DUR_BINS[min(token - DUR_OFFSET, len(DUR_BINS) - 1)])
            notes.append((pending, current_time, current_time + duration))
            current_time += duration
            pending = None
    return notes


def apply_length(
    notes: list[tuple[int, float, float]],
    controls: GenerationControls,
    target_notes: int,
    target_duration_sec: float,
) -> list[tuple[int, float, float]]:
    if controls.length_mode == "match_notes" and target_notes > 0:
        return notes[: max(1, target_notes)]
    if controls.length_mode == "match_duration" and target_duration_sec > 0 and notes:
        kept = [n for n in notes if n[1] < target_duration_sec] or notes[:1]
        last_pitch, last_start, last_end = kept[-1]
        if last_end > target_duration_sec and last_start < target_duration_sec:
            kept[-1] = (last_pitch, last_start, target_duration_sec)
        return kept
    return notes


def quantize_notes(
    notes: list[tuple[int, float, float]], bpm: float, subdivision: int
) -> list[tuple[int, float, float]]:
    """Snap onsets and offsets to a 60/bpm/subdivision grid, keeping notes in
    order, monophonic and at least one grid step long."""
    if not notes or bpm <= 0:
        return notes
    grid = 60.0 / bpm / subdivision
    quantized: list[tuple[int, float, float]] = []
    prev_start_step = -1
    for pitch, start, end in notes:
        start_step = max(int(round(start / grid)), prev_start_step + 1)
        end_step = max(int(round(end / grid)), start_step + 1)
        quantized.append((pitch, start_step * grid, end_step * grid))
        prev_start_step = start_step
    # keep monophonic: a note may not ring past the next onset
    for i in range(len(quantized) - 1):
        pitch, start, end = quantized[i]
        next_start = quantized[i + 1][1]
        if end > next_start:
            quantized[i] = (pitch, start, next_start)
    return quantized


def notes_to_midi_bytes(notes: list[tuple[int, float, float]], bpm: float) -> bytes:
    """Same instrument/velocity/tempo as ``tokens.tokens_to_midi``."""
    if pretty_midi is None:
        raise RuntimeError("pretty_midi is required for notes_to_midi_bytes")
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        instrument.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end))
    midi.instruments.append(instrument)
    buffer = io.BytesIO()
    midi.write(buffer)
    return buffer.getvalue()
