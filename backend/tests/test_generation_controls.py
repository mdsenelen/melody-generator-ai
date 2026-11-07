"""Opt-in generation controls (app/generation_controls.py) and the refactored
generate_iddm_variants path. Uses randomly initialised models, so it runs
without the checkpoint (which isn't in git)."""

from __future__ import annotations

import io
import random

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch

from app import inference
from app.generation_controls import (
    CURRENT_CONTROLS,
    VALID_DUR_TOKENS,
    GenerationControls,
    allowed_tokens,
    apply_length,
    grammar_violations,
    notes_to_midi_bytes,
    pitch_range_for,
    quantize_notes,
    sample_tokens,
    tokens_to_notes,
)
from app.model.colab_parity import ActorCritic, MelodyCVAE, MelStateEncoder
from app.model.tokens import (
    DUR_OFFSET,
    PAD_TOKEN,
    TEMPO_OFFSET,
    VOCAB_SIZE,
    midi_to_tokens,
    tokens_to_midi,
)

COMBINED = GenerationControls(
    latent_mode="scaled",
    latent_scale=0.25,
    grammar_mask=True,
    pitch_margin=12,
    length_mode="match_duration",
    quantize_subdivision=4,
)


def _write_midi(tmp_path, notes, name="seed.mid"):
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    instrument = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        instrument.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))
    midi.instruments.append(instrument)
    path = tmp_path / name
    midi.write(str(path))
    return path


def _read_notes(midi_bytes: bytes):
    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    return [note for instrument in midi.instruments for note in instrument.notes]


# --- grammar is consistent with the tokenizer ------------------------------


@pytest.mark.parametrize(
    "notes",
    [
        # scale with rests
        [(60 + i, i * 0.6, i * 0.6 + 0.5) for i in range(8)],
        # very short and very long durations, beyond the last DUR_BIN
        [(48, 0.0, 0.01), (50, 0.5, 6.5), (127, 7.0, 7.03), (0, 8.0, 9.0)],
        # overlapping (polyphonic) notes, like MAESTRO
        [(60, 0.0, 1.0), (64, 0.0, 1.0), (67, 0.0, 1.0), (72, 0.5, 2.0)],
    ],
)
def test_tokenizer_output_passes_grammar_and_mask(tmp_path, notes):
    tokens = midi_to_tokens(_write_midi(tmp_path, notes), max_notes=64)
    padded = tokens + [PAD_TOKEN] * (129 - len(tokens))

    assert tokens[0] >= TEMPO_OFFSET  # position 0 is the start token
    assert grammar_violations(padded) == 0

    # Replaying the sequence through the mask: every real token is allowed.
    pitch_range = pitch_range_for([n[0] for n in notes], margin=0)
    state, n_notes = "pitch", 0
    for token in padded[1:]:
        mask = allowed_tokens(state, VOCAB_SIZE, pitch_range, grammar=True, notes_so_far=n_notes)
        assert mask[token], f"mask blocked a token the tokenizer emits: {token}"
        if state == "duration":
            n_notes += 1
        state = "ended" if (token == PAD_TOKEN or state == "ended") else (
            "duration" if token < DUR_OFFSET else "pitch")


def test_every_duration_the_tokenizer_can_emit_is_allowed(tmp_path):
    durations = [0.001, 0.05, 0.0625, 0.07, 0.2, 0.5, 1.2, 3.9, 4.0, 4.1, 30.0]
    notes = []
    t = 0.0
    for d in durations:
        notes.append((60, t, t + d))
        t += d + 0.1
    tokens = midi_to_tokens(_write_midi(tmp_path, notes), max_notes=64)
    assert {tok for tok in tokens[2::2]} <= set(VALID_DUR_TOKENS)


@pytest.mark.parametrize(
    "tokens",
    [
        [0, 60, 62, 130],            # pitch followed by pitch
        [0, 130, 60, 130],           # duration without a pitch
        [0, 60, 130, 165, 60, 130],  # tempo token mid-sequence
        [0, 60, 150],                # duration token the tokenizer never emits
        [0, 60, 130, PAD_TOKEN, 62],  # content after the melody ended
        [0, 60, PAD_TOKEN],          # melody ends between pitch and duration
    ],
)
def test_grammar_violations_are_counted(tokens):
    assert grammar_violations(tokens) >= 1


def test_pad_cannot_end_an_empty_melody():
    before = allowed_tokens("pitch", VOCAB_SIZE, None, grammar=True, notes_so_far=0)
    after = allowed_tokens("pitch", VOCAB_SIZE, None, grammar=True, notes_so_far=1)
    assert not before[PAD_TOKEN]
    assert after[PAD_TOKEN]
    assert not any(before[TEMPO_OFFSET:PAD_TOKEN])


def test_pitch_range_follows_the_input_and_can_be_disabled():
    assert pitch_range_for([60, 67], margin=None) is None
    assert pitch_range_for([], margin=12) is None
    assert pitch_range_for([60, 67], margin=12) == (48, 79)
    assert pitch_range_for([3, 125], margin=12) == (0, 127)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latent_mode": "nope"},
        {"length_mode": "nope"},
        {"latent_scale": -1.0},
        {"pitch_margin": -1},
        {"quantize_subdivision": 0},
    ],
)
def test_invalid_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GenerationControls(**kwargs)


# --- tokens -> notes -> MIDI -----------------------------------------------


def test_tokens_to_notes_matches_tokens_to_midi(tmp_path):
    rng = random.Random(3)
    tokens = [0] + [rng.randrange(VOCAB_SIZE) for _ in range(128)]  # includes invalid transitions
    expected = _read_notes(open(tokens_to_midi(tokens, out_path=tmp_path / "ref.mid"), "rb").read())
    actual = _read_notes(notes_to_midi_bytes(tokens_to_notes(tokens), bpm=120.0))
    assert [(n.pitch, round(n.start, 3), round(n.end, 3)) for n in actual] == [
        (n.pitch, round(n.start, 3), round(n.end, 3)) for n in expected
    ]


NOTES = [(60 + i % 7, i * 0.5, i * 0.5 + 0.4) for i in range(20)]  # 10 s


def test_length_match_notes_and_duration():
    by_notes = apply_length(NOTES, GenerationControls(length_mode="match_notes"), 5, 0.0)
    assert len(by_notes) == 5

    by_duration = apply_length(NOTES, GenerationControls(length_mode="match_duration"), 0, 3.2)
    assert by_duration[-1][2] <= 3.2 + 1e-9
    assert len(by_duration) == 7  # onsets 0.0 .. 3.0


def test_length_never_empties_or_extends_the_melody():
    one_note_target = apply_length(NOTES, GenerationControls(length_mode="match_duration"), 0, 0.01)
    assert len(one_note_target) == 1
    longer_target = apply_length(NOTES, GenerationControls(length_mode="match_duration"), 0, 60.0)
    assert longer_target == NOTES
    assert apply_length(NOTES, CURRENT_CONTROLS, 3, 1.0) == NOTES
    assert apply_length([], GenerationControls(length_mode="match_duration"), 0, 3.0) == []


def test_quantize_is_idempotent_on_grid_input():
    grid = 60.0 / 100 / 4
    on_grid = [(60, i * 2 * grid, i * 2 * grid + grid) for i in range(8)]
    np.testing.assert_allclose(np.array(quantize_notes(on_grid, bpm=100, subdivision=4)),
                               np.array(on_grid), atol=1e-9)


def test_quantize_keeps_order_monophony_and_minimum_length():
    rough = [(60, 0.01, 0.02), (62, 0.03, 0.9), (64, 0.031, 0.5), (65, 1.07, 2.3)]
    grid = 60.0 / 120 / 4
    out = quantize_notes(rough, bpm=120, subdivision=4)
    starts = [n[1] for n in out]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    for pitch, start, end in out:
        assert abs(start / grid - round(start / grid)) < 1e-6
        assert end - start >= grid - 1e-9
    for current, following in zip(out, out[1:]):
        assert current[2] <= following[1] + 1e-9
    assert quantize_notes(rough, bpm=0, subdivision=4) == rough


# --- sampler ---------------------------------------------------------------


def _decoder_and_cond(seed=0):
    torch.manual_seed(seed)
    model = MelodyCVAE({})
    model.eval()
    z_cond = torch.randn(1, 32 + 3 + 12 + 8)
    return model.decoder, z_cond


def test_unmasked_sampler_reproduces_decoder_sample():
    decoder, z_cond = _decoder_and_cond()
    with torch.no_grad():
        expected, _, _ = decoder.sample(
            z_cond, seq_len=129, temperature=1.1, generator=torch.Generator().manual_seed(9))
        actual = sample_tokens(
            decoder, z_cond, 129, 1.1, torch.Generator().manual_seed(9),
            GenerationControls(), pitch_range=None)
    assert torch.equal(expected, actual)


def test_grammar_sampler_never_violates_and_yields_readable_midi():
    decoder, z_cond = _decoder_and_cond()
    controls = GenerationControls(grammar_mask=True)
    for seed in range(15):
        with torch.no_grad():
            tokens = sample_tokens(
                decoder, z_cond, 129, 1.5, torch.Generator().manual_seed(seed), controls, None
            ).squeeze(0).tolist()
        assert grammar_violations(tokens) == 0
        notes = _read_notes(notes_to_midi_bytes(tokens_to_notes(tokens), bpm=100))
        assert len(notes) >= 1


def test_pitch_range_is_respected_without_collapsing_variants():
    decoder, z_cond = _decoder_and_cond()
    controls = GenerationControls(grammar_mask=True, pitch_margin=5)
    pitch_range = pitch_range_for([60, 67], 5)
    sequences = []
    for seed in range(6):
        with torch.no_grad():
            tokens = sample_tokens(
                decoder, z_cond, 129, 1.0, torch.Generator().manual_seed(seed), controls, pitch_range
            ).squeeze(0).tolist()
        pitches = [p for p, _, _ in tokens_to_notes(tokens)]
        assert pitches and all(55 <= p <= 72 for p in pitches)
        sequences.append(tuple(pitches))
    assert len(set(sequences)) > 1


# --- generate_iddm_variants end to end (random weights) --------------------


def _fake_bundle():
    torch.manual_seed(0)
    cvae = MelodyCVAE({}).eval()
    return {
        "cfg": {},
        "cvae_model": cvae,
        "state_enc": MelStateEncoder().eval(),
        "ac": ActorCritic().eval(),
        "alpha": torch.tensor(0.1),
        "ready": True,
    }


def _wav_bytes(seconds=2.0):
    sr = 16000
    t = np.arange(int(sr * seconds)) / sr
    buffer = io.BytesIO()
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32), sr, format="WAV")
    return buffer.getvalue()


def _note_events(n, spacing=0.5):
    return [
        {"pitch": 60 + (i * 2) % 12, "start": i * spacing, "end": i * spacing + spacing * 0.8,
         "velocity": 90}
        for i in range(n)
    ]


@pytest.fixture
def fake_models(tmp_path, monkeypatch):
    bundle = _fake_bundle()
    monkeypatch.setattr(inference, "_load_cvae_iddm", lambda: bundle)
    monkeypatch.setattr(inference, "OUTPUT_DIR", tmp_path)
    return bundle


def _run(controls=CURRENT_CONTROLS, seed=11, n_notes=16):
    return inference.generate_iddm_variants(
        audio_bytes=_wav_bytes(),
        n_variants=3,
        temperatures=[0.7, 1.0, 1.3],
        seed=seed,
        known_note_events=_note_events(n_notes),
        controls=controls,
    )


def test_default_path_equals_the_pre_refactor_loop(fake_models):
    """The pre-refactor loop body, verbatim, run on the same context: the
    default controls must produce byte-identical MIDI."""
    result = _run()
    context = inference._build_variant_context(_wav_bytes(), _note_events(16))
    generator = torch.Generator().manual_seed(11)
    expected = []
    with torch.no_grad():
        for temp in [0.7, 1.0, 1.3]:
            dist = context["dist"]
            delta_z = dist.loc + dist.scale * inference._randn_like(dist.loc, generator) * temp
            z_final = context["base_mu"] + context["alpha"] * delta_z
            gen_tokens, _, _ = fake_models["cvae_model"].decoder.sample(
                torch.cat([z_final, context["cond"]], dim=-1),
                seq_len=129, temperature=temp, generator=generator)
            expected.append(inference._token_ids_to_midi_bytes(
                gen_tokens.squeeze(0).tolist(), bpm=context["tempo_bpm"]))
    import base64
    assert [base64.b64decode(v["midi_b64"]) for v in result["variants"]] == expected


def test_response_shape_is_unchanged(fake_models):
    result = _run()
    assert set(result) == {"n_variants", "temperatures", "mood_idx", "mood_label",
                           "model_status", "variants"}
    assert set(result["variants"][0]) == {"index", "temperature", "midi_b64", "midi_filename",
                                         "midi_download_path", "wav_b64", "wav_filename",
                                         "wav_download_path"}


@pytest.mark.parametrize("controls", [CURRENT_CONTROLS, COMBINED])
def test_same_seed_is_reproducible(fake_models, controls):
    first = [v["midi_b64"] for v in _run(controls, seed=5)["variants"]]
    second = [v["midi_b64"] for v in _run(controls, seed=5)["variants"]]
    other = [v["midi_b64"] for v in _run(controls, seed=6)["variants"]]
    assert first == second
    assert first != other


@pytest.mark.parametrize("n_notes", [1, 3, 100])
@pytest.mark.parametrize("controls", [CURRENT_CONTROLS, COMBINED])
def test_short_and_long_inputs_yield_non_empty_readable_midi(fake_models, controls, n_notes):
    import base64

    for variant in _run(controls, n_notes=n_notes)["variants"]:
        notes = _read_notes(base64.b64decode(variant["midi_b64"]))
        assert notes, "variant MIDI has no notes"
        assert all(0 <= n.pitch <= 127 and n.end > n.start for n in notes)
        if controls is COMBINED:
            seed_pitches = [60 + (i * 2) % 12 for i in range(min(n_notes, 64))]
            assert all(min(seed_pitches) - 12 <= n.pitch <= max(seed_pitches) + 12 for n in notes)
