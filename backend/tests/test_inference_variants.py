from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi import UploadFile

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference  # noqa: E402


class _DummyModule:
    def __init__(self, *args, **kwargs):
        self.loaded_state = None

    def to(self, device):
        return self

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state = (state_dict, strict)

    def eval(self):
        return self


class _DummyCVAE(_DummyModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = object()


@pytest.fixture(autouse=True)
def reset_variant_bundle(monkeypatch):
    monkeypatch.setattr(inference, "_CVAE_IDDM_BUNDLE", None)


def test_load_cvae_iddm_success_and_cached(tmp_path, monkeypatch, caplog):
    cvae_path = tmp_path / "cvae_weights.pth"
    iddm_path = tmp_path / "iddm_ppo_weights.pth"
    cvae_path.write_bytes(b"0" * 20001)
    iddm_path.write_bytes(b"1" * 22001)

    load_calls: list[Path] = []

    def fake_torch_load(path, map_location=None):
        load_calls.append(Path(path))
        if Path(path) == cvae_path:
            return {
                "cfg": {
                    "vocab": 177,
                    "emb_dim": 32,
                    "hidden": 64,
                    "latent_dim": 16,
                    "n_moods": 3,
                    "mel_bins": 80,
                    "T_win": 16,
                    "enc_dim": 64,
                    "seq_len": 129,
                },
                "model": {"decoder.weight": torch.zeros(1)},
            }
        return {
            "enc": {"enc.weight": torch.zeros(1)},
            "disc": {"disc.weight": torch.zeros(1)},
            "ac": {"ac.weight": torch.zeros(1)},
            "mine": {"mine.weight": torch.zeros(1)},
        }

    monkeypatch.setattr(inference, "CVAE_WEIGHTS_PATH", cvae_path)
    monkeypatch.setattr(inference, "IDDM_WEIGHTS_PATH", iddm_path)
    monkeypatch.setattr(inference, "torch", type("TorchProxy", (), {"load": staticmethod(fake_torch_load)})())
    monkeypatch.setattr(inference, "MelodyCVAE", _DummyCVAE)
    monkeypatch.setattr(inference, "MelStateEncoder", _DummyModule)
    monkeypatch.setattr(inference, "TransitionDiscriminator", _DummyModule)
    monkeypatch.setattr(inference, "MelodyPPOActorCritic", _DummyModule)
    monkeypatch.setattr(inference, "MINENetwork", _DummyModule)

    with caplog.at_level("INFO"):
        bundle = inference._load_cvae_iddm()
        cached = inference._load_cvae_iddm()

    assert bundle["ready"] is True
    assert cached is bundle
    assert load_calls == [cvae_path, iddm_path]
    assert "CVAE checkpoint keys" in caplog.text
    assert "IDDM-PPO checkpoint keys" in caplog.text


def test_load_cvae_iddm_raises_for_missing_file(tmp_path, monkeypatch):
    cvae_path = tmp_path / "cvae_weights.pth"
    iddm_path = tmp_path / "iddm_ppo_weights.pth"
    cvae_path.write_bytes(b"0" * 20001)

    monkeypatch.setattr(inference, "CVAE_WEIGHTS_PATH", cvae_path)
    monkeypatch.setattr(inference, "IDDM_WEIGHTS_PATH", iddm_path)

    with pytest.raises(HTTPException) as exc:
        inference._load_cvae_iddm()

    assert exc.value.status_code == 503
    assert str(iddm_path) in exc.value.detail


def test_load_cvae_iddm_raises_for_tiny_file(tmp_path, monkeypatch):
    cvae_path = tmp_path / "cvae_weights.pth"
    iddm_path = tmp_path / "iddm_ppo_weights.pth"
    cvae_path.write_bytes(b"0" * 20001)
    iddm_path.write_bytes(b"1" * 16)

    monkeypatch.setattr(inference, "CVAE_WEIGHTS_PATH", cvae_path)
    monkeypatch.setattr(inference, "IDDM_WEIGHTS_PATH", iddm_path)

    with pytest.raises(HTTPException) as exc:
        inference._load_cvae_iddm()

    assert exc.value.status_code == 503
    assert str(iddm_path) in exc.value.detail


def test_transcribe_and_mood_returns_shared_analysis(monkeypatch):
    waveform = np.ones(22050, dtype=np.float32)
    note_events = [
        {"start": 0.0, "end": 0.5, "pitch": 72, "velocity": 96},
        {"start": 0.6, "end": 1.1, "pitch": 76, "velocity": 96},
    ]

    monkeypatch.setattr(inference, "basic_pitch_predict", object())
    monkeypatch.setattr(inference, "_read_audio_bytes", lambda raw, sr: waveform)
    monkeypatch.setattr(inference, "_run_basic_pitch_predict", lambda path: (b"MThd\x00\x00", note_events))
    monkeypatch.setattr(inference, "_estimate_tempo", lambda events, midi_bytes=None: 128.0)
    monkeypatch.setattr(inference, "_extract_key_label", lambda midi_bytes, histogram: "C major")
    monkeypatch.setattr(inference, "_detect_chords_from_audio", lambda audio, sr: ["C", "G"])

    result = inference._transcribe_and_mood(b"fake audio")

    assert result["midi_bytes"] == b"MThd\x00\x00"
    assert result["n_notes"] == 2
    assert result["tempo_bpm"] == 128.0
    assert result["avg_pitch"] == 74.0
    assert result["mood_idx"] == 0
    assert result["mood_label"] == "happy"
    assert result["key"] == "C major"
    assert result["detected_chords"] == ["C", "G"]
    assert len(result["pitch_histogram"]) == 12


def test_generate_variants_route_accepts_formdata(monkeypatch):
    captured = {}

    def fake_generate_iddm_variants(audio_bytes: bytes, n_variants: int, temperatures: list[float]):
        captured["audio_bytes"] = audio_bytes
        captured["n_variants"] = n_variants
        captured["temperatures"] = temperatures
        return {
            "n_variants": n_variants,
            "temperatures": temperatures,
            "mood_idx": 2,
            "mood_label": "neutral",
            "model_status": {
                "cvae": {"path": "cvae", "exists": True, "size_mb": 1.0, "loaded": True},
                "iddm_ppo": {"path": "iddm", "exists": True, "size_mb": 1.0, "loaded": True},
                "device": "cpu",
                "load_error": None,
                "fluidsynth_available": False,
            },
            "variants": [],
        }

    monkeypatch.setattr(inference, "generate_iddm_variants", fake_generate_iddm_variants)

    response = asyncio.run(
        inference.generate_variants_route(
            file=UploadFile(filename="clip.wav", file=io.BytesIO(b"audio-bytes")),
            n_variants=2,
            temperatures="[0.7, 0.9]",
        )
    )

    assert response["temperatures"] == [0.7, 0.9]
    assert captured["audio_bytes"] == b"audio-bytes"
    assert captured["n_variants"] == 2
    assert captured["temperatures"] == [0.7, 0.9]


def test_generate_variants_route_rejects_temperature_length_mismatch(monkeypatch):
    monkeypatch.setattr(inference, "generate_iddm_variants", lambda *args, **kwargs: pytest.fail("should not be called"))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference.generate_variants_route(
                file=UploadFile(filename="clip.wav", file=io.BytesIO(b"audio-bytes")),
                n_variants=2,
                temperatures="[0.7]",
            )
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "temperatures array length must equal n_variants"


def test_transcribe_route_uses_run_basic_pitch(monkeypatch):
    monkeypatch.setattr(
        inference,
        "run_basic_pitch",
        lambda raw, filename: {
            "n_notes": 1,
            "duration_sec": 1.0,
            "midi_b64": "abc",
            "wav_b64": None,
            "midi_filename": "demo.mid",
            "wav_filename": "",
            "mood_label": "neutral",
            "mood_idx": 2,
            "detected_chords": [],
            "key": "C major",
            "pitch_histogram": [0.0] * 12,
            "tempo_bpm": 90.0,
            "average_pitch": 60.0,
        },
    )

    response = asyncio.run(
        inference.transcribe_audio(file=UploadFile(filename="clip.wav", file=io.BytesIO(b"audio-bytes")))
    )

    assert response["midi_filename"] == "demo.mid"


def test_model_status_route_does_not_load_models(monkeypatch):
    def fail_load():
        raise AssertionError("model loader should not be called")

    monkeypatch.setattr(inference, "_load_cvae_iddm", fail_load)
    monkeypatch.setattr(inference, "_fluidsynth_available", lambda: False)
    monkeypatch.setattr(
        inference,
        "_CVAE_IDDM_BUNDLE",
        {"cvae_loaded": True, "iddm_loaded": False, "load_error": "cached error"},
    )

    response = inference.model_status()

    payload = response
    assert payload["cvae"]["loaded"] is True
    assert payload["iddm_ppo"]["loaded"] is False
    assert payload["load_error"] == "cached error"
