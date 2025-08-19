import json
import os
import librosa
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

from app.model.vae import WebVAE


APP_DIR = Path(__file__).resolve().parent              # backend/app/model
MODEL_PATH = APP_DIR / "weights" / "web_model.pt"

# Weights are optional during development; handle missing files gracefully
if MODEL_PATH.exists():
    _model_data = torch.load(str(MODEL_PATH), map_location="cpu")
    chord_vocab = _model_data.get("chord_vocab", [])
else:  # pragma: no cover - exercised in environments without weights
    _model_data = {}
    chord_vocab = []


class AudioProcessor:
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512,
                 n_mels=128, fmin=30, fmax=8000):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def preprocess(self, audio_path):
        y, _ = librosa.load(audio_path, sr=self.sample_rate)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
        )
        mel_db = librosa.amplitude_to_db(mel, ref=np.max, top_db=80)
        mel_db = np.clip(mel_db, -40, 0)
        mel_norm = (mel_db + 40) / 40  # Normalize to [0,1]

        # Pad/trim to fixed size
        if mel_norm.shape[1] < 256:
            mel_norm = np.pad(mel_norm, ((0, 0), (0, 256-mel_norm.shape[1])))
        else:
            mel_norm = mel_norm[:, :256]

        return torch.FloatTensor(mel_norm)

    def postprocess(self, mel_tensor: torch.Tensor) -> np.ndarray:
        """Converts mel tensor to waveform using Griffin-Lim"""
        mel = mel_tensor.squeeze().cpu().numpy()

        mel = mel * 80 - 40  # Denormalize
        mel_power = librosa.db_to_power(mel)

        audio = librosa.feature.inverse.mel_to_audio(
            mel_power,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=32,
        )
        return audio

    def save(self, waveform: np.ndarray, out_path: str):
        """Saves waveform to WAV"""
        sf.write(out_path, waveform, self.sample_rate)


def save_model(model, path, metadata=None):
    """Saves VAE model with config and optional metadata"""
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'input_shape': model.input_shape,
            'latent_dim': model.latent_dim
        },
        'metadata': metadata or {}
    }, path)


def load_model(path: str, device=None) -> WebVAE:
    """Loads model from .pt and prepares it for inference"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(path, map_location=device)
    model = WebVAE(
        input_shape=state['config']['input_shape'],
        latent_dim=state['config']['latent_dim']
    )
    model.load_state_dict(state['state_dict'])
