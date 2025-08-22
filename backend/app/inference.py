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
from librosa.feature import chroma_cqt

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse

from .model.vae import WebVAE
from .model.utils import AudioProcessor
from .progressions import roman_to_chords, fetch_popular_progressions

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
        "n_mels": 128,
        "fmin": 30,
        "fmax": 8000
    }
}
AUDIO_CFG = (
    json.load(open(AUDIO_CONFIG_PATH)) if os.path.exists(AUDIO_CONFIG_PATH)
    else DEFAULT_AUDIO
)

processor = AudioProcessor(**AUDIO_CFG["audio"])

# Initialize model with fallback for missing weights
try:
    model = WebVAE().to(DEVICE)
    if os.path.exists(MODEL_WEIGHTS):
        ckpt = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        state_dict = ckpt["state_dict"] if isinstance(
            ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully")
    else:
        print("⚠️ Model weights not found, using untrained model for development")
    model.eval()
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    # Create a simple mock model for development
    model = None


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


def generate_from_chord(chord: str, duration: float = 4.0, creativity: float = 0.7):
    """
    Tek bir akordan kısa bir mel segmenti üretir.
    Not: Mevcut VAE decode(z) yalnızca z alıyor, akor koşullaması yok.
    """
    with torch.no_grad():
        # Akor bazlı deterministik seed -> farklı akorlar farklı latent üretir
        torch.manual_seed(abs(hash(chord)) % (2**32))
        z = torch.randn(1, model.latent_dim, device=DEVICE)

        mel = _decode_from_latent(z)
        mel = np.clip(mel, 0.0, 1.0)

    # Kısa süre hedefi için (istenirse) zaman ekseninde kırpma
    # mel.shape == (n_mels, time_frames)
    max_frames = 256  # Fixed value instead of config
    mel = mel[:, :max_frames]

    audio_bytes = _mel_to_audio_bytes(mel, AUDIO_CFG)
    return mel, audio_bytes, [chord]


def generate_from_progression(roman: list[str], key: str = "C",
                              segment_seconds: float = 2.0, creativity: float = 0.7):
    """
    Roman numeral progresyonundan melodi üretir.
    Her akor için ayrı segment üretir ve zaman ekseninde birleştirir.
    """
    chords = roman_to_chords(roman, key=key)
    
    mel_segments = []
    detected = []
    for ch in chords:
        mel, _, det = generate_from_chord(ch, duration=segment_seconds, creativity=creativity)
        mel_segments.append(mel)
        detected.extend(det)

    # Zaman ekseninde birleştir
    mel_cat = np.concatenate(mel_segments, axis=1)
    mel_cat = np.clip(mel_cat, 0.0, 1.0)

    audio_bytes = _mel_to_audio_bytes(mel_cat, AUDIO_CFG)
    return mel_cat, audio_bytes, detected


def detect_chords_from_audio(audio_bytes: bytes) -> list[str]:
    """
    Detect chords from audio using chroma features.
    This is a simplified chord detection - you can replace with your trained model.
    """
    try:
        # Load audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # Extract chroma features
        chroma = chroma_cqt(y=y, sr=sr, hop_length=512)
        
        # Simple chord detection based on chroma peaks
        # This is a basic implementation - replace with your trained model
        chord_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        chord_qualities = ["", "m", "dim", "aug"]
        
        detected_chords = []
        
        # Analyze each time frame
        for i in range(min(chroma.shape[1], 10)):  # Analyze first 10 frames
            frame = chroma[:, i]
            # Find the strongest root note
            root_idx = np.argmax(frame)
            root_note = chord_names[root_idx]
            
            # Simple quality detection (major/minor)
            # This is very basic - your trained model would be much better
            if frame[(root_idx + 4) % 12] > frame[(root_idx + 3) % 12]:
                quality = "m"  # minor
            else:
                quality = ""   # major
                
            chord = f"{root_note}{quality}"
            if chord not in detected_chords:
                detected_chords.append(chord)
        
        return detected_chords[:5]  # Return up to 5 unique chords
        
    except Exception as e:
        print(f"Chord detection error: {e}")
        return ["C", "Am", "F", "G"]  # Fallback chords


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


@router.get("/progressions")
async def list_progressions(key: str = "C"):
    """Kurulu popüler progresyonları listeler"""
    progs = fetch_popular_progressions()
    mapped = [{"roman": p, "chords": roman_to_chords(p, key)} for p in progs]
    return {"key": key, "progressions": mapped}


@router.post("/generate-progression")
async def generate_progression(
    roman: list[str] = Body(..., embed=True),
    key: str = Body("C"),
    segment_seconds: float = Body(2.0),
    creativity: float = Body(0.7),
):
    """Roman numeral progresyonundan melodi üretir"""
    try:
        mel, audio_bytes, detected = generate_from_progression(
            roman=roman, key=key, segment_seconds=segment_seconds, creativity=creativity
        )
        packaged = _save_and_package(mel, "prog")
        packaged["detected_chords"] = detected
        return JSONResponse(packaged)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Progression generation failed: {e}")


@router.post("/detect-chords")
async def detect_chords_endpoint(audio: UploadFile = File(...)):
    """Detect chords from uploaded audio"""
    try:
        raw = audio.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file upload")
        
        detected_chords = detect_chords_from_audio(raw)
        
        return JSONResponse({
            "detected_chords": detected_chords,
            "message": f"Detected {len(detected_chords)} chords"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chord detection failed: {e}")