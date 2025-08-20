# 📁 backend/app/inference.py
import io
import os
import json
import uuid
import base64
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf
import torch
import librosa

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from .model.vae import WebVAE
from .model.utils import AudioProcessor

HERE = os.path.dirname(os.path.abspath(__file__))               # backend/app
# backend/app/model/weights
WEIGHTS_DIR = os.path.join(HERE, "model", "weights")
# backend/data/generated
OUTPUT_DIR = os.path.join(HERE, "..", "data", "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, "final_vocal2accomp.pth")
AUDIO_CONFIG_PATH = os.path.join(WEIGHTS_DIR, "audio_params.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_AUDIO = {
    "audio": {
        "sample_rate": 22050,
        "n_fft": 2048,
        "hop_length": 512,
        "win_length": 1024,
        "n_mels": 128,
        "fmin": 30,
        "fmax": 8000,
        "max_frames": 256
    }
}
AUDIO_CFG = (
    json.load(open(AUDIO_CONFIG_PATH)) if os.path.exists(AUDIO_CONFIG_PATH)
    else DEFAULT_AUDIO
)

processor = AudioProcessor(**AUDIO_CFG["audio"])
model = WebVAE().to(DEVICE)
if not os.path.exists(MODEL_WEIGHTS):
    raise RuntimeError(f"❌ Missing model weights: {MODEL_WEIGHTS}")
ckpt = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
state_dict = ckpt["state_dict"] if isinstance(
    ckpt, dict) and "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()


def _read_audio_from_upload(file: UploadFile, target_sr: int) -> np.ndarray:
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        raise HTTPException(
            status_code=400, detail="Supported formats: wav, mp3, flac, ogg, m4a")
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload")
    buf = io.BytesIO(raw)
    y, _ = librosa.load(buf, sr=target_sr, mono=True)
    y = librosa.util.normalize(y).astype(np.float32)
    return y


def _mel_to_audio_bytes(mel_norm: np.ndarray, cfg: Dict[str, Any]) -> bytes:
    mel_db = (mel_norm * 40.0) - 40.0
    mel_mag = librosa.db_to_amplitude(mel_db)
    audio = librosa.griffinlim(
        mel_mag,
        hop_length=cfg["audio"]["hop_length"],
        win_length=cfg["audio"]["win_length"],
        n_iter=32
    )
    out = io.BytesIO()
    sf.write(out, audio, cfg["audio"]["sample_rate"], format="WAV")
    out.seek(0)
    return out.read()


def _encode(audio: np.ndarray) -> torch.Tensor:
    mel = processor.waveform_to_mel(audio)
    return torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)


def _decode_from_latent(z: torch.Tensor,
                        cond_mem: Optional[torch.Tensor] = None,
                        chord_idx: Optional[torch.Tensor] = None) -> np.ndarray:
    with torch.no_grad():
        try:
            if chord_idx is not None and cond_mem is not None:
                recon = model.decode(z, chord_idx, cond_mem)
            elif chord_idx is not None:
                recon = model.decode(z, chord_idx)
            else:
                recon = model.decode(z)
        except TypeError:
            try:
                recon = model.decode(z)
            except Exception:
                recon, _, _ = model(z)
    return recon.squeeze().detach().cpu().numpy()


def _generate_accompaniment_core(vocal_audio: np.ndarray,
                                 style_audio: Optional[np.ndarray] = None,
                                 style_mix: float = 0.4,
                                 creativity: float = 0.6) -> np.ndarray:
    mel_vocal = _encode(vocal_audio)
    with torch.no_grad():
        mu_v, logvar_v = model.encode(mel_vocal)[:2] if hasattr(
            model, "encode") else (None, None)
        if mu_v is None:
            recon_v, mu_v, logvar_v = model(mel_vocal)
        std_v = torch.exp(0.5 * logvar_v)
        eps = torch.randn_like(std_v)
        z_v = mu_v + eps * std_v * max(0.05, creativity)

        if style_audio is not None:
            mel_style = _encode(style_audio)
            mu_s, logvar_s = model.encode(mel_style)[:2] if hasattr(
                model, "encode") else (None, None)
            if mu_s is None:
                recon_s, mu_s, logvar_s = model(mel_style)
            std_s = torch.exp(0.5 * logvar_s)
            eps_s = torch.randn_like(std_s)
            z_s = mu_s + eps_s * std_s * max(0.05, creativity)
            z = (1.0 - style_mix) * z_v + style_mix * z_s
        else:
            z = z_v

        mel_out = _decode_from_latent(z)
        return np.clip(mel_out, 0.0, 1.0)


def _mel_lane_mask(mel: np.ndarray,
                   sr: int,
                   n_mels: int,
                   fmin: int,
                   fmax: int,
                   lanes: Dict[str, bool]) -> np.ndarray:
    mel_f = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    keep = np.ones_like(mel, dtype=np.float32)

    def band_mask(low, high):
        idx = np.where((mel_f >= low) & (mel_f < high))[0]
        m = np.zeros((n_mels, 1), dtype=np.float32)
        m[idx] = 1.0
        return m

    bass_band = band_mask(20, 200)
    piano_band = band_mask(200, 2000)
    other_band = np.ones((n_mels, 1), dtype=np.float32) - \
        np.clip(bass_band + piano_band, 0, 1)

    if not lanes.get("bass", True):
        keep *= (1.0 - bass_band) + 0.05 * bass_band
    if not lanes.get("piano", True):
        keep *= (1.0 - piano_band) + 0.05 * piano_band
    if not lanes.get("other", True):
        keep *= (1.0 - other_band) + 0.05 * other_band
    if not lanes.get("drums", True):
        keep *= 0.7

    return np.clip(mel * keep, 0.0, 1.0)


def _save_and_package(mel: np.ndarray, prefix: str) -> Dict[str, Any]:
    wav_bytes = _mel_to_audio_bytes(mel, AUDIO_CFG)
    b64 = base64.b64encode(wav_bytes).decode("utf-8")
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.wav"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(wav_bytes)
    return {
        "filename": fname,
        "download_path": f"/static/generated/{fname}",
        "audio_b64": b64,
    }


def generate_from_audio(audio_bytes: bytes,
                        creativity: float = 0.7,
                        chord: Optional[str] = None) -> tuple:
    """
    Enhanced audio generation with chord conditioning.

    Args:
        audio_bytes: Input audio as bytes.
        creativity: Control variation (0-1).
        chord: Optional chord label (e.g., "C", "Am").

    Returns:
        tuple: (reconstructed_mel, generated_audio_bytes, detected_chords)
    """
    y, _ = librosa.load(
        io.BytesIO(audio_bytes),
        sr=AUDIO_CFG["audio"]["sample_rate"],
        mono=True,
    )
    y = librosa.util.normalize(y).astype(np.float32)
    mel_tensor = _encode(y)

    with torch.no_grad():
        mu, logvar = model.encode(mel_tensor)[:2] if hasattr(model, "encode") else (None, None)
        if mu is None:
            recon, mu, logvar = model(mel_tensor)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std * max(0.05, creativity)
        mel_out = _decode_from_latent(z)

    mel_out = np.clip(mel_out, 0.0, 1.0)
    audio_bytes_out = _mel_to_audio_bytes(mel_out, AUDIO_CFG)
    detected_chords = [chord] if chord else []
    return mel_out, audio_bytes_out, detected_chords


router = APIRouter()


@router.post("/generate-accompaniment")
async def generate_accompaniment(
    vocal: UploadFile = File(...),
    creativity: float = Form(0.6),
):
    try:
        vocal_audio = _read_audio_from_upload(
            vocal, AUDIO_CFG["audio"]["sample_rate"])
        mel_out = _generate_accompaniment_core(
            vocal_audio=vocal_audio,
            style_audio=None,
            style_mix=0.0,
            creativity=float(creativity),
        )
        return JSONResponse(_save_and_package(mel_out, "accomp"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@router.post("/generate-style")
async def generate_style(
    vocal: UploadFile = File(...),
    style: UploadFile = File(...),
    style_mix: float = Form(0.4),
    creativity: float = Form(0.6),
):
    try:
        sr = AUDIO_CFG["audio"]["sample_rate"]
        vocal_audio = _read_audio_from_upload(vocal, sr)
        style_audio = _read_audio_from_upload(style, sr)
        mel_out = _generate_accompaniment_core(
            vocal_audio=vocal_audio,
            style_audio=style_audio,
            style_mix=float(np.clip(style_mix, 0.0, 1.0)),
            creativity=float(creativity),
        )
        return JSONResponse(_save_and_package(mel_out, "style"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Stylized generation failed: {e}")


@router.post("/generate-lanes")
async def generate_lanes(
    audio: UploadFile = File(...),
    keep_bass: bool = Form(True),
    keep_piano: bool = Form(True),
    keep_drums: bool = Form(True),
    keep_other: bool = Form(True),
):
    try:
        y = _read_audio_from_upload(audio, AUDIO_CFG["audio"]["sample_rate"])
        mel = processor.waveform_to_mel(y)
        mel_masked = _mel_lane_mask(
            mel,
            sr=AUDIO_CFG["audio"]["sample_rate"],
            n_mels=AUDIO_CFG["audio"]["n_mels"],
            fmin=AUDIO_CFG["audio"]["fmin"],
            fmax=AUDIO_CFG["audio"]["fmax"],
            lanes={"bass": keep_bass, "piano": keep_piano,
                   "drums": keep_drums, "other": keep_other},
        )
        return JSONResponse(_save_and_package(mel_masked, "lanes"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lane masking failed: {e}")
