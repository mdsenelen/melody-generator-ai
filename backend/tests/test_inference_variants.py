from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi import UploadFile

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference  # noqa: E402
from app.model.colab_parity import ActorCritic, heuristic_mood_from_metrics


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


_NEW_CFG = {
    "vocab": 177,
    "emb_dim": 48,
    "hidden": 128,
    "latent_dim": 32,
    "n_moods": 3,
    "mel_bins": 80,
    "T_win": 32,
    "enc_dim": 96,
    "seq_len": 129,
    "use_chord_cond": True,
    "use_bar_cond": True,
    "chord_emb_dim": 12,
    "bar_emb_dim": 8,
    "n_chord_classes": 25,
    "n_bar_bins": 8,
}


def test_load_cvae_iddm_success_and_cached(tmp_path, monkeypatch, caplog):
    joint_path = tmp_path / "joint_e2e_weights.pth"
    joint_path.write_bytes(b"0" * 20001)

    load_calls: list[Path] = []

    def fake_torch_load(path, map_location=None):
        load_calls.append(Path(path))
        return {
            "cfg": _NEW_CFG,
            "cvae": {"cvae.weight": torch.zeros(1)},
            "state_enc": {"enc.weight": torch.zeros(1)},
            "actor_critic": {"ac.weight": torch.zeros(1)},
            "discriminator": {"disc.weight": torch.zeros(1)},
            "mine": {"mine.weight": torch.zeros(1)},
            "alpha": 0.1,
        }

    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", joint_path)
    monkeypatch.setattr(inference, "torch", type("TorchProxy", (), {"load": staticmethod(fake_torch_load), "tensor": staticmethod(torch.tensor)})())
    monkeypatch.setattr(inference, "MelodyCVAE", _DummyCVAE)
    monkeypatch.setattr(inference, "MelStateEncoder", _DummyModule)
    monkeypatch.setattr(inference, "TransitionDiscriminator", _DummyModule)
    monkeypatch.setattr(inference, "ActorCritic", _DummyModule)
    monkeypatch.setattr(inference, "MINENetwork", _DummyModule)

    with caplog.at_level("INFO"):
        bundle = inference._load_cvae_iddm()
        cached = inference._load_cvae_iddm()

    assert bundle["ready"] is True
    assert cached is bundle
    assert load_calls == [joint_path]
    assert "CVAE checkpoint keys" in caplog.text
    assert "IDDM-PPO checkpoint keys" in caplog.text


def test_load_cvae_iddm_concurrent_calls_load_once(tmp_path, monkeypatch):
    """Two concurrent cold-start callers (now realistic since generation
    routes run in FastAPI's thread pool) must not race: only one should
    actually construct the models, the other waits and reuses the result."""
    joint_path = tmp_path / "joint_e2e_weights.pth"
    joint_path.write_bytes(b"0" * 20001)

    load_started = threading.Event()
    release_load = threading.Event()
    load_call_count = 0
    count_lock = threading.Lock()

    def fake_torch_load(path, map_location=None):
        nonlocal load_call_count
        with count_lock:
            load_call_count += 1
        load_started.set()
        assert release_load.wait(timeout=5), "second caller never reached the lock"
        return {
            "cfg": _NEW_CFG,
            "cvae": {"cvae.weight": torch.zeros(1)},
            "state_enc": {"enc.weight": torch.zeros(1)},
            "actor_critic": {"ac.weight": torch.zeros(1)},
            "discriminator": {"disc.weight": torch.zeros(1)},
            "mine": {"mine.weight": torch.zeros(1)},
            "alpha": 0.1,
        }

    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", joint_path)
    monkeypatch.setattr(inference, "torch", type("TorchProxy", (), {"load": staticmethod(fake_torch_load), "tensor": staticmethod(torch.tensor)})())
    monkeypatch.setattr(inference, "MelodyCVAE", _DummyCVAE)
    monkeypatch.setattr(inference, "MelStateEncoder", _DummyModule)
    monkeypatch.setattr(inference, "TransitionDiscriminator", _DummyModule)
    monkeypatch.setattr(inference, "ActorCritic", _DummyModule)
    monkeypatch.setattr(inference, "MINENetwork", _DummyModule)

    results: list[dict] = []

    def call_load():
        results.append(inference._load_cvae_iddm())

    first = threading.Thread(target=call_load)
    first.start()
    assert load_started.wait(timeout=5), "first caller never started loading"

    second = threading.Thread(target=call_load)
    second.start()
    # Give the second caller time to reach (and block on) the load lock
    # before releasing the in-flight load.
    time.sleep(0.2)
    release_load.set()

    first.join(timeout=5)
    second.join(timeout=5)

    assert load_call_count == 1
    assert len(results) == 2
    assert results[0] is results[1]


def test_load_cvae_iddm_legacy_two_file_format(tmp_path, monkeypatch, caplog):
    cvae_path = tmp_path / "cvae_weights.pth"
    iddm_path = tmp_path / "iddm_ppo_weights.pth"
    missing_joint = tmp_path / "joint_e2e_weights.pth"
    cvae_path.write_bytes(b"0" * 20001)
    iddm_path.write_bytes(b"1" * 22001)

    load_calls: list[Path] = []

    def fake_torch_load(path, map_location=None):
        load_calls.append(Path(path))
        if Path(path) == cvae_path:
            return {
                "cfg": {"vocab": 177, "emb_dim": 32, "hidden": 64, "latent_dim": 16, "n_moods": 3, "enc_dim": 64, "seq_len": 129},
                "model": {"decoder.weight": torch.zeros(1)},
            }
        return {
            "enc": {"enc.weight": torch.zeros(1)},
            "disc": {"disc.weight": torch.zeros(1)},
            "ac": {"ac.weight": torch.zeros(1)},
            "mine": {"mine.weight": torch.zeros(1)},
        }

    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", missing_joint)
    monkeypatch.setattr(inference, "CVAE_WEIGHTS_PATH", cvae_path)
    monkeypatch.setattr(inference, "IDDM_WEIGHTS_PATH", iddm_path)
    monkeypatch.setattr(inference, "torch", type("TorchProxy", (), {"load": staticmethod(fake_torch_load), "tensor": staticmethod(torch.tensor)})())
    monkeypatch.setattr(inference, "MelodyCVAE", _DummyCVAE)
    monkeypatch.setattr(inference, "MelStateEncoder", _DummyModule)
    monkeypatch.setattr(inference, "TransitionDiscriminator", _DummyModule)
    monkeypatch.setattr(inference, "ActorCritic", _DummyModule)
    monkeypatch.setattr(inference, "MINENetwork", _DummyModule)

    with caplog.at_level("INFO"):
        bundle = inference._load_cvae_iddm()

    assert bundle["ready"] is True
    assert load_calls == [cvae_path, iddm_path]
    assert "CVAE checkpoint keys" in caplog.text


def test_load_cvae_iddm_legacy_format_raises_on_schema_drift(tmp_path, monkeypatch):
    """A legacy-format checkpoint whose 'cvae'/'model' state dict doesn't match
    the current MelodyCVAE architecture must fail loudly (503) instead of
    silently loading with random weights for the unmatched parameters."""
    cvae_path = tmp_path / "cvae_weights.pth"
    iddm_path = tmp_path / "iddm_ppo_weights.pth"
    missing_joint = tmp_path / "joint_e2e_weights.pth"
    cvae_path.write_bytes(b"0" * 20001)
    iddm_path.write_bytes(b"1" * 22001)

    def fake_torch_load(path, map_location=None):
        if Path(path) == cvae_path:
            return {
                "cfg": {"vocab": 177, "emb_dim": 32, "hidden": 64, "latent_dim": 16, "n_moods": 3, "enc_dim": 64, "seq_len": 129},
                # Deliberately renamed/incompatible key, simulating checkpoint
                # schema drift against the current MelodyCVAE architecture.
                "model": {"decoder.this_key_does_not_exist": torch.zeros(1)},
            }
        return {"enc": {}, "disc": {}, "ac": {}, "mine": {}}

    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", missing_joint)
    monkeypatch.setattr(inference, "CVAE_WEIGHTS_PATH", cvae_path)
    monkeypatch.setattr(inference, "IDDM_WEIGHTS_PATH", iddm_path)
    monkeypatch.setattr(inference, "torch", type("TorchProxy", (), {"load": staticmethod(fake_torch_load), "tensor": staticmethod(torch.tensor)})())
    # Deliberately do NOT stub out MelodyCVAE here: the real module's
    # load_state_dict is what needs to reject the mismatched keys.
    monkeypatch.setattr(inference, "MelStateEncoder", _DummyModule)
    monkeypatch.setattr(inference, "TransitionDiscriminator", _DummyModule)
    monkeypatch.setattr(inference, "ActorCritic", _DummyModule)
    monkeypatch.setattr(inference, "MINENetwork", _DummyModule)

    with pytest.raises(HTTPException) as exc:
        inference._load_cvae_iddm()

    assert exc.value.status_code == 503


def test_load_cvae_iddm_raises_for_missing_file(tmp_path, monkeypatch):
    cvae_path = tmp_path / "cvae_weights.pth"
    iddm_path = tmp_path / "iddm_ppo_weights.pth"
    missing_joint = tmp_path / "joint_e2e_weights.pth"
    cvae_path.write_bytes(b"0" * 20001)

    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", missing_joint)
    monkeypatch.setattr(inference, "CVAE_WEIGHTS_PATH", cvae_path)
    monkeypatch.setattr(inference, "IDDM_WEIGHTS_PATH", iddm_path)

    with pytest.raises(HTTPException) as exc:
        inference._load_cvae_iddm()

    assert exc.value.status_code == 503
    assert "missing" in exc.value.detail.lower()


def test_load_cvae_iddm_raises_for_tiny_file(tmp_path, monkeypatch):
    cvae_path = tmp_path / "cvae_weights.pth"
    iddm_path = tmp_path / "iddm_ppo_weights.pth"
    missing_joint = tmp_path / "joint_e2e_weights.pth"
    cvae_path.write_bytes(b"0" * 20001)
    iddm_path.write_bytes(b"1" * 16)

    monkeypatch.setattr(inference, "JOINT_WEIGHTS_PATH", missing_joint)
    monkeypatch.setattr(inference, "CVAE_WEIGHTS_PATH", cvae_path)
    monkeypatch.setattr(inference, "IDDM_WEIGHTS_PATH", iddm_path)

    with pytest.raises(HTTPException) as exc:
        inference._load_cvae_iddm()

    assert exc.value.status_code == 503
    assert "corrupt" in exc.value.detail.lower() or "incomplete" in exc.value.detail.lower()


def test_run_generation_returns_result_within_timeout():
    result = asyncio.run(inference._run_generation(lambda: 42))
    assert result == 42


def test_run_generation_times_out(monkeypatch):
    monkeypatch.setattr(inference, "GENERATION_TIMEOUT_SECONDS", 0.05)

    def slow():
        time.sleep(0.5)
        return "should not be reached"

    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference._run_generation(slow))

    assert exc.value.status_code == 504
    assert "timed out" in exc.value.detail.lower()


def test_cleanup_stale_files_removes_old_and_keeps_recent(tmp_path, monkeypatch):
    upload_dir = tmp_path / "recordings"
    output_dir = tmp_path / "generated"
    upload_dir.mkdir()
    output_dir.mkdir()

    old_file = upload_dir / "upload_old.wav"
    old_file.write_bytes(b"old")
    recent_file = output_dir / "generated_recent.wav"
    recent_file.write_bytes(b"recent")

    old_time = time.time() - 48 * 3600
    os.utime(old_file, (old_time, old_time))

    monkeypatch.setattr(inference, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(inference, "OUTPUT_DIR", output_dir)

    removed = inference.cleanup_stale_files(max_age_hours=24)

    assert removed == 1
    assert not old_file.exists()
    assert recent_file.exists()


def test_cleanup_stale_files_disabled_when_non_positive(tmp_path, monkeypatch):
    upload_dir = tmp_path / "recordings"
    upload_dir.mkdir()
    old_file = upload_dir / "upload_old.wav"
    old_file.write_bytes(b"old")
    os.utime(old_file, (time.time() - 100 * 3600, time.time() - 100 * 3600))

    monkeypatch.setattr(inference, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(inference, "OUTPUT_DIR", tmp_path / "generated")
    (tmp_path / "generated").mkdir()

    assert inference.cleanup_stale_files(max_age_hours=0) == 0
    assert old_file.exists()


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
    assert result["note_events"] == note_events
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


# ── heuristic_mood_from_metrics ───────────────────────────────────────────────

@pytest.mark.parametrize("tempo,pitch,key,expected_idx,expected_label", [
    # Fast + major key → happy
    (120.0, 62.0, "C major", 0, "happy"),
    # Fast + high pitch (no key info) → happy
    (115.0, 70.0, "", 0, "happy"),
    # Slow + minor key → sad
    (75.0, 64.0, "A minor", 1, "sad"),
    # Slow + low pitch (no key info) → sad
    (70.0, 55.0, "", 1, "sad"),
    # Minor key at moderate tempo (85 BPM) → sad
    (85.0, 63.0, "D minor", 1, "sad"),
    # Minor key above 100 BPM → neutral (not slow enough for rule 3)
    (105.0, 63.0, "D minor", 2, "neutral"),
    # Unknown key, mid tempo, mid pitch → neutral
    (95.0, 63.0, "", 2, "neutral"),
    # Unknown key string → treated as no-key (neutral)
    (95.0, 63.0, "Unknown", 2, "neutral"),
])
def test_heuristic_mood_from_metrics(tempo, pitch, key, expected_idx, expected_label):
    idx, label = heuristic_mood_from_metrics(tempo, pitch, key)
    assert idx == expected_idx
    assert label == expected_label


def test_heuristic_mood_from_metrics_default_key_arg():
    """key_label defaults to '' so old two-arg callers still work."""
    idx, label = heuristic_mood_from_metrics(120.0, 70.0)
    assert idx == 0
    assert label == "happy"
