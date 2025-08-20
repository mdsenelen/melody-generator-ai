"""Tests for AudioProcessor utility functions."""

from __future__ import annotations
from app.model.utils import AudioProcessor

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import types

# Ensure the backend package is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_postprocess_uses_instance_attributes(monkeypatch):
    """`postprocess` should forward instance parameters to mel_to_audio."""

    captured = {}

    def fake_mel_to_audio(mel_power, sr, n_fft, hop_length, n_iter):  # pragma: no cover - monkeypatched
        captured.update(sr=sr, n_fft=n_fft,
                        hop_length=hop_length, n_iter=n_iter)
        return np.zeros(1)

    fake_inverse = types.SimpleNamespace(mel_to_audio=fake_mel_to_audio)
    monkeypatch.setattr(librosa, "feature",
                        types.SimpleNamespace(inverse=fake_inverse))

    proc = AudioProcessor(sample_rate=12345, n_fft=64, hop_length=16)
    mel = torch.zeros(1, proc.n_mels, 10)
    proc.postprocess(mel)

    assert captured == {"sr": 12345, "n_fft": 64,
                        "hop_length": 16, "n_iter": 32}


def test_save_respects_sample_rate(tmp_path):
    proc = AudioProcessor(sample_rate=12345)
    waveform = np.zeros(100)
    out_file = tmp_path / "test.wav"
    proc.save(waveform, str(out_file))
    _, sr = sf.read(out_file)
    assert sr == 12345
