from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import subprocess
import tempfile
import traceback
import uuid
import shutil
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .model.colab_parity import (
    ActorCritic,
    MINENetwork,
    MelStateEncoder,
    MelodyCVAE,
    MelodyPPOActorCritic,
    TransitionDiscriminator,
    build_mood_onehot,
    heuristic_mood_from_metrics,
    midi_to_tokens,
    tokens_to_chord_idx,
    tokens_to_bar_idx,
    tokens_to_midi,
)

try:  # pragma: no cover - optional dependency
    import pretty_midi
except Exception:  # pragma: no cover - optional dependency
    pretty_midi = None

try:  # pragma: no cover - optional dependency
    import music21 as m21
except Exception:  # pragma: no cover - optional dependency
    m21 = None

try:  # pragma: no cover - optional dependency
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict as basic_pitch_predict
except Exception:  # pragma: no cover - optional dependency
    ICASSP_2022_MODEL_PATH = None
    basic_pitch_predict = None


router = APIRouter()
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "recordings"
OUTPUT_DIR = DATA_DIR / "generated"
LOG_DIR = DATA_DIR / "logs"
WEIGHTS_DIR = APP_DIR / "model" / "weights"

for directory in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

AUDIO_CONFIG_PATH = WEIGHTS_DIR / "audio_params.json"
CVAE_WEIGHTS_PATH = WEIGHTS_DIR / "cvae_weights.pth"
IDDM_WEIGHTS_PATH = WEIGHTS_DIR / "iddm_ppo_weights.pth"
JOINT_WEIGHTS_PATH = WEIGHTS_DIR / "joint_e2e_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm")
NOTEBOOK_VARIANT_DEFAULT_TEMPERATURES = [0.7, 0.9, 1.0, 1.3]
NOTEBOOK_VARIANT_AUDIO_DEFAULTS = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 256,
    "mel_bins": 80,
    "T_win": 32,
}

DEFAULT_AUDIO_CFG = {
    "audio": {
        "sample_rate": 22050,
        "n_fft": 2048,
        "hop_length": 512,
        "win_length": 1024,
        "n_mels": 128,
        "fmin": 30,
        "fmax": 8000,
        "max_frames": 256,
    }
}

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

_PROGRESSIONS_JSON = Path(__file__).resolve().parent / "data" / "progressions.json"

_PRESET_PROGRESSIONS_FALLBACK: list[dict[str, Any]] = [
    {"id": "pop-1",   "name": "I-V-vi-IV",  "genre": "Pop",  "source": "fallback", "chords": ["C","G","Am","F"],           "song_title": None, "artist": None, "year": None},
    {"id": "jazz-2",  "name": "ii7-V7-I",   "genre": "Jazz", "source": "fallback", "chords": ["Dm7","G7","Cmaj7","A7"],     "song_title": None, "artist": None, "year": None},
    {"id": "rock-1",  "name": "I-vi-IV-V",  "genre": "Rock", "source": "fallback", "chords": ["Em","C","G","D"],            "song_title": None, "artist": None, "year": None},
    {"id": "pop-4",   "name": "I-vi-ii-V",  "genre": "Pop",  "source": "fallback", "chords": ["Am","F","C","G"],            "song_title": None, "artist": None, "year": None},
]


def _load_preset_progressions() -> list[dict[str, Any]]:
    if _PROGRESSIONS_JSON.exists():
        try:
            return json.loads(_PROGRESSIONS_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _PRESET_PROGRESSIONS_FALLBACK


PRESET_PROGRESSIONS: list[dict[str, Any]] = _load_preset_progressions()

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#",
                     "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_CLASS_TO_SEMITONE = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

MAJOR_PROFILE = np.asarray(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.asarray(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_AUDIO_CFG: Optional[dict[str, Any]] = None
_CVAE_IDDM_BUNDLE: Optional[dict[str, Any]] = None


def _load_audio_cfg() -> dict[str, Any]:
    global _AUDIO_CFG
    if _AUDIO_CFG is None:
        if AUDIO_CONFIG_PATH.exists():
            with AUDIO_CONFIG_PATH.open("r", encoding="utf-8") as handle:
                _AUDIO_CFG = json.load(handle)
        else:
            _AUDIO_CFG = DEFAULT_AUDIO_CFG
    return _AUDIO_CFG


def _checkpoint_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 2)


def _fluidsynth_available() -> bool:
    try:
        return shutil.which("fluidsynth") is not None
    except Exception:
        return False


def _variant_model_status() -> dict[str, Any]:
    bundle = _CVAE_IDDM_BUNDLE or {}
    joint_exists = JOINT_WEIGHTS_PATH.exists()
    cvae_path = JOINT_WEIGHTS_PATH if joint_exists else CVAE_WEIGHTS_PATH
    iddm_path = JOINT_WEIGHTS_PATH if joint_exists else IDDM_WEIGHTS_PATH
    return {
        "cvae": {
            "path": str(cvae_path),
            "exists": cvae_path.exists(),
            "size_mb": _checkpoint_size_mb(cvae_path),
            "loaded": bool(bundle.get("cvae_loaded", False)),
        },
        "iddm_ppo": {
            "path": str(iddm_path),
            "exists": iddm_path.exists(),
            "size_mb": _checkpoint_size_mb(iddm_path),
            "loaded": bool(bundle.get("iddm_loaded", False)),
        },
        "device": str(DEVICE),
        "load_error": bundle.get("load_error"),
        "fluidsynth_available": _fluidsynth_available(),
    }


def _raise_variant_service_unavailable(detail: str) -> None:
    raise HTTPException(status_code=503, detail=detail)


def _load_cvae_iddm() -> dict[str, Any]:
    global _CVAE_IDDM_BUNDLE
    if _CVAE_IDDM_BUNDLE is not None:
        if _CVAE_IDDM_BUNDLE.get("load_error"):
            _raise_variant_service_unavailable(
                str(_CVAE_IDDM_BUNDLE["load_error"]))
        return _CVAE_IDDM_BUNDLE

    def _err(detail: str) -> None:
        global _CVAE_IDDM_BUNDLE
        _CVAE_IDDM_BUNDLE = {
            "cfg": {},
            "cvae_loaded": False,
            "iddm_loaded": False,
            "load_error": detail,
            "ready": False,
        }
        _raise_variant_service_unavailable(detail)

    use_joint = JOINT_WEIGHTS_PATH.exists()

    try:
        if use_joint:
            path = JOINT_WEIGHTS_PATH
            file_size = os.path.getsize(path)
            if file_size <= 10_000:
                _err("Model weights file is corrupted or incomplete")
            logger.info("Found joint checkpoint at %s (%.2f MB)", path, file_size / (1024 * 1024))

            ckpt = torch.load(path, map_location=DEVICE)
            logger.info("CVAE checkpoint keys: %s", sorted(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__)
            logger.info("IDDM-PPO checkpoint keys: %s", sorted(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__)

            if not isinstance(ckpt, dict) or "cvae" not in ckpt:
                _err("Joint checkpoint is missing the 'cvae' state dict")

            cfg = dict(ckpt.get("cfg") or {})
            raw_alpha = ckpt.get("alpha", 0.1)
            alpha = torch.tensor(float(raw_alpha), device=DEVICE)

            cvae_sd = ckpt["cvae"]
            state_enc_sd = ckpt.get("state_enc")
            ac_sd = ckpt.get("actor_critic")
            disc_sd = ckpt.get("discriminator")
            mine_sd = ckpt.get("mine")
            strict = True
        else:
            for label, path in [("cvae", CVAE_WEIGHTS_PATH), ("iddm_ppo", IDDM_WEIGHTS_PATH)]:
                if not os.path.exists(path):
                    _err("Required model weights file is missing")
                file_size = os.path.getsize(path)
                if file_size <= 10_000:
                    _err("Model weights file is corrupted or incomplete")
                logger.info("Found %s checkpoint at %s (%.2f MB)", label, path, file_size / (1024 * 1024))

            cvae_ckpt = torch.load(CVAE_WEIGHTS_PATH, map_location=DEVICE)
            iddm_ckpt = torch.load(IDDM_WEIGHTS_PATH, map_location=DEVICE)
            logger.info("CVAE checkpoint keys: %s", sorted(cvae_ckpt.keys()) if isinstance(cvae_ckpt, dict) else type(cvae_ckpt).__name__)
            logger.info("IDDM-PPO checkpoint keys: %s", sorted(iddm_ckpt.keys()) if isinstance(iddm_ckpt, dict) else type(iddm_ckpt).__name__)

            if not isinstance(cvae_ckpt, dict):
                _err("CVAE checkpoint has an unexpected format")
            if not isinstance(iddm_ckpt, dict):
                _err("IDDM-PPO checkpoint has an unexpected format")

            cfg = dict(cvae_ckpt.get("cfg") or {})
            alpha = torch.tensor(0.1, device=DEVICE)

            # Support both old key "model" and new key "cvae"
            cvae_sd = cvae_ckpt.get("cvae") or cvae_ckpt.get("model")
            if cvae_sd is None:
                _err("CVAE checkpoint is missing 'cvae' or 'model' state dict")

            state_enc_sd = iddm_ckpt.get("enc")
            ac_sd = iddm_ckpt.get("ac")
            disc_sd = iddm_ckpt.get("disc")
            mine_sd = iddm_ckpt.get("mine")
            strict = False  # legacy format may have architecture mismatches

        if not cfg:
            logger.warning("cfg not in checkpoint, using notebook defaults")

        latent_dim = int(cfg.get("latent_dim", 32))
        enc_dim = int(cfg.get("enc_dim", 96))
        mel_bins = int(cfg.get("mel_bins", NOTEBOOK_VARIANT_AUDIO_DEFAULTS["mel_bins"]))
        t_win = int(cfg.get("T_win", NOTEBOOK_VARIANT_AUDIO_DEFAULTS["T_win"]))
        n_moods = int(cfg.get("n_moods", 3))

        cvae_model = MelodyCVAE(cfg).to(DEVICE)
        cvae_model.load_state_dict(cvae_sd, strict=strict)
        cvae_model.eval()

        state_encoder = MelStateEncoder(mel_bins=mel_bins, T_win=t_win, enc_dim=enc_dim).to(DEVICE)
        if state_enc_sd is not None:
            state_encoder.load_state_dict(state_enc_sd, strict=strict)
        state_encoder.eval()

        discriminator = TransitionDiscriminator(enc_dim=enc_dim).to(DEVICE)
        if disc_sd is not None:
            discriminator.load_state_dict(disc_sd, strict=strict)
        discriminator.eval()

        actor_critic = ActorCritic(enc_dim=enc_dim, latent_dim=latent_dim).to(DEVICE)
        if ac_sd is not None:
            actor_critic.load_state_dict(ac_sd, strict=strict)
        actor_critic.eval()

        mine = MINENetwork(sa_dim=enc_dim + latent_dim, sp_dim=enc_dim).to(DEVICE)
        if mine_sd is not None:
            mine.load_state_dict(mine_sd, strict=strict)
        mine.eval()

        _CVAE_IDDM_BUNDLE = {
            "cfg": cfg,
            "cvae_model": cvae_model,
            "state_enc": state_encoder,
            "disc": discriminator,
            "ac": actor_critic,
            "mine": mine,
            "alpha": alpha,
            "cvae_loaded": True,
            "iddm_loaded": True,
            "load_error": None,
            "ready": True,
        }
        return _CVAE_IDDM_BUNDLE
    except HTTPException:
        raise
    except Exception as exc:
        detail = "Failed to load model weights"
        logger.exception("Failed to load Colab parity models")
        _CVAE_IDDM_BUNDLE = {
            "cfg": {},
            "cvae_loaded": False,
            "iddm_loaded": False,
            "load_error": detail,
            "ready": False,
        }
        raise HTTPException(status_code=503, detail=detail) from exc


def get_runtime_status() -> dict[str, Any]:
    return {
        "device": str(DEVICE),
        "vae_status": "cvae_iddm",
        "weights_dir": str(WEIGHTS_DIR),
        "basic_pitch_available": basic_pitch_predict is not None,
        "pretty_midi_available": pretty_midi is not None,
        "music21_available": m21 is not None,
        "variant_status": _variant_model_status(),
    }


def get_model_info() -> dict[str, Any]:
    cfg = _load_audio_cfg()
    return {
        "model": "MelodyCVAE+IDDM-PPO",
        "latent_dim": 16,
        "audio_config": cfg["audio"],
        "device": str(DEVICE),
        "status": "cvae_iddm",
        "weights_dir": str(WEIGHTS_DIR),
    }


def get_available_chords() -> list[str]:
    return DEFAULT_CHORDS


def _audio_cfg() -> dict[str, Any]:
    return _load_audio_cfg()


def _waveform_to_mel(audio: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    audio_cfg = cfg["audio"]
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        win_length=audio_cfg["win_length"],
        n_mels=audio_cfg["n_mels"],
        fmin=audio_cfg["fmin"],
        fmax=audio_cfg["fmax"],
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, -40, 0)
    mel_norm = (mel_db + 40) / 40
    max_frames = int(audio_cfg.get("max_frames", 256))
    if mel_norm.shape[1] < max_frames:
        mel_norm = np.pad(
            mel_norm, ((0, 0), (0, max_frames - mel_norm.shape[1])))
    else:
        mel_norm = mel_norm[:, :max_frames]
    return mel_norm.astype(np.float32)


def _mel_to_audio_bytes(mel_norm: np.ndarray, cfg: dict[str, Any]) -> bytes:
    audio_cfg = cfg["audio"]
    mel = np.clip(mel_norm, 0.0, 1.0) * 80.0 - 40.0
    mel_power = librosa.db_to_power(mel)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        win_length=audio_cfg["win_length"],
        n_iter=32,
    )
    return _waveform_to_wav_bytes(audio, audio_cfg["sample_rate"])


def _waveform_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    out = io.BytesIO()
    sf.write(out, np.asarray(audio, dtype=np.float32),
             sample_rate, format="WAV")
    out.seek(0)
    return out.read()


def _read_audio_bytes(raw: bytes, target_sr: int) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    buf = io.BytesIO(raw)
    try:
        audio, _ = librosa.load(buf, sr=target_sr, mono=True)
    except Exception:
        # Fallback: convert via ffmpeg (handles WebM/Opus from browser recordings)
        tmp_in = tmp_out = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(raw)
                tmp_in = f.name
            tmp_out = tmp_in + "_converted.wav"
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in,
                 "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp_out],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise HTTPException(
                    status_code=400,
                    detail="Could not decode audio (ffmpeg: " + (
                        result.stderr.decode(errors="replace").strip().splitlines() or ["(no output)"]
                    )[-1] + ")",
                )
            with open(tmp_out, "rb") as f:
                wav_bytes = f.read()
            audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=target_sr, mono=True)
        finally:
            for p in (tmp_in, tmp_out):
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except Exception:
                        pass
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Could not decode audio")
    return librosa.util.normalize(audio).astype(np.float32)


def _ensure_supported_audio_filename(filename: Optional[str]) -> str:
    if not filename or not filename.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
        raise HTTPException(
            status_code=400, detail="Supported formats: wav, mp3, flac, ogg, m4a, webm")
    return filename


async def _read_upload_bytes(file: UploadFile) -> bytes:
    _ensure_supported_audio_filename(file.filename)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    return raw


def _read_audio_from_upload(file: UploadFile, target_sr: int) -> np.ndarray:
    _ensure_supported_audio_filename(file.filename)
    raw = file.file.read()
    return _read_audio_bytes(raw, target_sr)


def _save_bytes(raw: bytes, prefix: str, extension: str) -> tuple[str, Path]:
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}{extension}"
    path = OUTPUT_DIR / filename
    path.write_bytes(raw)
    return filename, path


def _package_audio_bytes(raw: bytes, prefix: str) -> dict[str, Any]:
    filename, _ = _save_bytes(raw, prefix, ".wav")
    return {
        "audio_b64": base64.b64encode(raw).decode("utf-8"),
        "filename": filename,
        "download_path": f"/api/download/{filename}",
    }


def _resolve_upload_path(filename: Optional[str], upload_id: Optional[str]) -> Optional[Path]:
    if filename:
        candidate = (UPLOAD_DIR / Path(filename).name).resolve()
        if candidate.exists() and candidate.parent == UPLOAD_DIR.resolve():
            return candidate

    if upload_id:
        matches = sorted(UPLOAD_DIR.glob(f"upload_{upload_id}.*"))
        if matches:
            return matches[0]

    return None


def _encode(audio: np.ndarray, cfg: dict[str, Any]) -> torch.Tensor:
    mel = _waveform_to_mel(audio, cfg)
    return torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)


def _decode_from_latent(model: MelodyCVAE, z: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        recon = model.decode(z)
    return recon.squeeze().detach().cpu().numpy()


def _infer_chord_from_chroma(chroma_vector: np.ndarray) -> str:
    quality_templates = {
        "": np.array([1.0, 0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0]),
        "m": np.array([1.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0]),
        "7": np.array([1.0, 0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.7, 0.0, 0.0, 0.6, 0.0]),
        "maj7": np.array([1.0, 0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.6]),
        "m7": np.array([1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.6, 0.0]),
    }
    chroma = np.asarray(chroma_vector, dtype=np.float32)
    if chroma.sum() <= 0:
        return "C"
    chroma /= chroma.sum()

    best_score = float("-inf")
    best_label = "C"
    for root_idx, root in enumerate(PITCH_CLASS_NAMES):
        for quality, template in quality_templates.items():
            rolled = np.roll(template, root_idx)
            score = float(np.dot(chroma, rolled) - 0.12 *
                          np.dot(chroma, 1.0 - rolled))
            label = f"{root}{quality}"
            if score > best_score:
                best_score = score
                best_label = label
    return best_label


def _detect_chords_from_audio(audio: np.ndarray, sample_rate: int) -> list[str]:
    try:
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)

    if chroma.shape[1] == 0:
        return []

    segment_count = min(8, max(1, chroma.shape[1] // 8))
    boundaries = np.linspace(
        0, chroma.shape[1], num=segment_count + 1, dtype=int)
    chords: list[str] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        label = _infer_chord_from_chroma(np.mean(chroma[:, start:end], axis=1))
        if not chords or chords[-1] != label:
            chords.append(label)
    return chords


def _parse_chord_notes(chord: str) -> list[int]:
    chord = chord.strip()
    if not chord:
        return [60, 64, 67]

    root = chord[0].upper()
    suffix = chord[1:]
    if suffix.startswith(("#", "b")):
        root += suffix[0]
        suffix = suffix[1:]

    root_semitone = PITCH_CLASS_TO_SEMITONE.get(root, 0)
    quality = suffix.lower()
    intervals = [0, 4, 7]
    if quality in {"m", "min"}:
        intervals = [0, 3, 7]
    elif quality == "7":
        intervals = [0, 4, 7, 10]
    elif quality in {"maj7"}:
        intervals = [0, 4, 7, 11]
    elif quality in {"m7", "min7"}:
        intervals = [0, 3, 7, 10]
    elif quality == "dim":
        intervals = [0, 3, 6]
    elif quality == "aug":
        intervals = [0, 4, 8]
    elif quality == "5":
        intervals = [0, 7]
    elif quality == "sus2":
        intervals = [0, 2, 7]
    elif quality == "sus4":
        intervals = [0, 5, 7]
    elif quality == "9":
        intervals = [0, 4, 7, 10, 14]
    elif quality == "maj9":
        intervals = [0, 4, 7, 11, 14]
    elif quality in {"m9", "min9"}:
        intervals = [0, 3, 7, 10, 14]
    elif quality == "add9":
        intervals = [0, 4, 7, 14]

    root_midi = 60 + root_semitone
    notes = [root_midi - 12] + [root_midi + interval for interval in intervals]
    return sorted(notes)


def _note_events_for_progression(
    progression: list[str],
    bpm: float,
    temperature: float = 1.0,
    variant_seed: int = 0,
    chord_beats: float = 4.0,
) -> list[dict[str, float | int]]:
    randomizer = random.Random(
        f"{variant_seed}:{temperature}:{','.join(progression)}")
    seconds_per_beat = 60.0 / max(bpm, 1.0)
    chord_duration = seconds_per_beat * chord_beats
    note_events: list[dict[str, float | int]] = []

    for chord_index, chord in enumerate(progression):
        chord_start = chord_index * chord_duration
        chord_end = chord_start + chord_duration * 0.96
        notes = _parse_chord_notes(chord)
        variation_mode = "held"
        if temperature > 1.15:
            variation_mode = "arpeggio"
        if temperature > 1.45 and randomizer.random() > 0.5:
            variation_mode = "strum"

        if variation_mode == "held":
            for note in notes:
                note_events.append({
                    "start": chord_start,
                    "end": chord_end,
                    "pitch": note,
                    "velocity": 78 if note == notes[0] else 92,
                })
        else:
            subdivision = 4 if variation_mode == "arpeggio" else 6
            step_duration = chord_duration / subdivision
            playable_notes = notes[1:] if len(notes) > 1 else notes
            for step in range(subdivision):
                note = playable_notes[step % len(playable_notes)]
                if variation_mode == "strum" and randomizer.random() > 0.65:
                    note = playable_notes[randomizer.randrange(
                        0, len(playable_notes))]
                start = chord_start + step * step_duration
                end = min(chord_end, start + step_duration *
                          (0.75 + randomizer.random() * 0.2))
                note_events.append({
                    "start": start,
                    "end": end,
                    "pitch": note,
                    "velocity": 74 + int(randomizer.random() * 40),
                })
            note_events.append({
                "start": chord_start,
                "end": chord_end,
                "pitch": notes[0],
                "velocity": 88,
            })

    return note_events


def _encode_var_len(value: int) -> bytes:
    buffer = value & 0x7F
    bytes_out = [buffer]
    value >>= 7
    while value:
        buffer = (value & 0x7F) | 0x80
        bytes_out.insert(0, buffer)
        value >>= 7
    return bytes(bytes_out)


def _tokens_to_midi_bytes(tokens: list[dict[str, float | int]], bpm: float, instrument: int) -> bytes:
    ticks_per_beat = 480
    tempo_microseconds = int(60_000_000 / max(bpm, 1.0))

    timeline: list[tuple[int, str, int, int]] = []
    for token in tokens:
        start_tick = int(float(token["start"]) * bpm / 60.0 * ticks_per_beat)
        end_tick = int(float(token["end"]) * bpm / 60.0 * ticks_per_beat)
        pitch = int(token["pitch"])
        velocity = int(token.get("velocity", 90))
        timeline.append((start_tick, "on", pitch, velocity))
        timeline.append((max(end_tick, start_tick + 1), "off", pitch, 0))

    timeline.sort(key=lambda item: (
        item[0], 0 if item[1] == "off" else 1, item[2]))
    track = bytearray()
    track.extend(_encode_var_len(0))
    track.extend(b"\xFF\x51\x03")
    track.extend(tempo_microseconds.to_bytes(3, byteorder="big"))

    track.extend(_encode_var_len(0))
    track.extend(bytes([0xC0, max(0, min(127, int(instrument)))]))

    previous_tick = 0
    for tick, event_type, pitch, velocity in timeline:
        track.extend(_encode_var_len(max(0, tick - previous_tick)))
        status = 0x90 if event_type == "on" else 0x80
        track.extend(bytes([status, pitch, velocity]))
        previous_tick = tick

    track.extend(_encode_var_len(0))
    track.extend(b"\xFF\x2F\x00")

    midi = bytearray()
    midi.extend(b"MThd")
    midi.extend((6).to_bytes(4, byteorder="big"))
    midi.extend((0).to_bytes(2, byteorder="big"))
    midi.extend((1).to_bytes(2, byteorder="big"))
    midi.extend(ticks_per_beat.to_bytes(2, byteorder="big"))
    midi.extend(b"MTrk")
    midi.extend(len(track).to_bytes(4, byteorder="big"))
    midi.extend(track)
    return bytes(midi)


def _synthesize_note_events_to_waveform(
    note_events: list[dict[str, float | int]],
    sample_rate: int,
    tail_seconds: float = 0.3,
) -> np.ndarray:
    if not note_events:
        return np.zeros(sample_rate, dtype=np.float32)

    total_duration = max(float(event["end"])
                         for event in note_events) + tail_seconds
    waveform = np.zeros(int(total_duration * sample_rate), dtype=np.float32)

    for event in note_events:
        start = max(0, int(float(event["start"]) * sample_rate))
        end = max(start + 1, int(float(event["end"]) * sample_rate))
        pitch = int(event["pitch"])
        velocity = int(event.get("velocity", 90))
        frequency = librosa.midi_to_hz(pitch)
        t = np.linspace(0.0, (end - start) / sample_rate,
                        end - start, endpoint=False)
        envelope = np.minimum(1.0, t / 0.015) * np.exp(-2.8 * t)
        tone = (
            np.sin(2 * np.pi * frequency * t)
            + 0.25 * np.sin(2 * np.pi * frequency * 2 * t)
            + 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
        )
        waveform[start:end] += (velocity / 127.0) * 0.18 * envelope * tone

    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform /= peak
    return waveform.astype(np.float32)


def _midi_bytes_to_wav_b64(
    midi_bytes: bytes,
    sample_rate: int,
    note_events: Optional[list[dict[str, float | int]]] = None,
    prefer_fluidsynth_only: bool = False,
) -> Optional[str]:
    if pretty_midi is not None:  # pragma: no branch
        temp_midi_path: Optional[Path] = None
        try:  # pragma: no cover
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
                handle.write(midi_bytes)
                temp_midi_path = Path(handle.name)
            midi_obj = pretty_midi.PrettyMIDI(str(temp_midi_path))
            _SF2 = os.environ.get(
                "SOUNDFONT_PATH",
                str(APP_DIR / "soundfonts" / "GeneralUser-GS.sf2")
            )

            rendered = midi_obj.fluidsynth(
                fs=sample_rate,
                sf2_path=_SF2 if Path(_SF2).exists() else None
            )
            return base64.b64encode(_waveform_to_wav_bytes(rendered, sample_rate)).decode("utf-8")
        except Exception as e:
            logger.warning("FluidSynth render failed: %s", e)
        finally:
            if temp_midi_path and temp_midi_path.exists():
                temp_midi_path.unlink(missing_ok=True)

    if prefer_fluidsynth_only:
        return None

    # Lazily extract note events from midi_bytes when not provided, so callers
    # don't pay the parsing cost when FluidSynth succeeds.
    effective = note_events if note_events else _note_events_from_midi_bytes(midi_bytes)
    if not effective:
        return None

    rendered = _synthesize_note_events_to_waveform(effective, sample_rate)
    return base64.b64encode(_waveform_to_wav_bytes(rendered, sample_rate)).decode("utf-8")


def _extract_midi_bytes(midi_data: Any) -> Optional[bytes]:
    if midi_data is None:
        return None
    if isinstance(midi_data, (bytes, bytearray)):
        return bytes(midi_data)
    if isinstance(midi_data, str) and os.path.exists(midi_data):
        return Path(midi_data).read_bytes()
    if hasattr(midi_data, "write"):
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
                temp_path = Path(handle.name)
            midi_data.write(str(temp_path))
            return temp_path.read_bytes()
        except Exception:
            return None
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
    return None


def _coerce_note_events(raw_note_events: Any) -> list[dict[str, float | int]]:
    events: list[dict[str, float | int]] = []
    if not raw_note_events:
        return events

    for item in raw_note_events:
        if isinstance(item, dict):
            start = float(item.get("start_time", item.get("start", 0.0)))
            end = float(item.get("end_time", item.get("end", start + 0.25)))
            pitch = int(item.get("pitch", 60))
            amplitude = float(
                item.get("amplitude", item.get("velocity", 0.85)))
            velocity = amplitude if amplitude > 1 else amplitude * 127
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start = float(item[0])
            end = float(item[1])
            pitch = int(item[2])
            amplitude = float(item[3]) if len(item) > 3 else 0.85
            velocity = amplitude if amplitude > 1 else amplitude * 127
        else:
            continue
        events.append({
            "start": start,
            "end": max(start + 0.05, end),
            "pitch": pitch,
            "velocity": int(max(1, min(127, round(velocity)))),
        })

    return events


def _run_basic_pitch_predict(audio_path: str) -> tuple[Optional[bytes], list[dict[str, float | int]]]:
    if basic_pitch_predict is None:
        return None, []

    invocation_attempts = [
        {"args": (audio_path,), "kwargs": {}},
        {"args": (audio_path,), "kwargs": {
            "model_or_model_path": ICASSP_2022_MODEL_PATH}},
        {"args": (audio_path, ICASSP_2022_MODEL_PATH), "kwargs": {}},
    ]

    last_error: Optional[Exception] = None
    for attempt in invocation_attempts:
        try:
            result = basic_pitch_predict(*attempt["args"], **attempt["kwargs"])
            if isinstance(result, tuple) and len(result) >= 3:
                _, midi_data, note_events = result[:3]
            else:
                midi_data, note_events = None, []
            return _extract_midi_bytes(midi_data), _coerce_note_events(note_events)
        except TypeError as exc:
            last_error = exc
        except Exception as exc:  # pragma: no cover
            last_error = exc
            break

    if last_error is not None:
        raise last_error
    return None, []


def _fallback_note_events_from_audio(audio: np.ndarray, sample_rate: int) -> list[dict[str, float | int]]:
    hop_length = int(_audio_cfg()["audio"]["hop_length"])
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)
    note_events: list[dict[str, float | int]] = []

    start_index: Optional[int] = None
    current_pitches: list[float] = []
    for index, (pitch_hz, is_voiced) in enumerate(zip(f0, voiced_flag)):
        valid_pitch = bool(
            is_voiced) and pitch_hz is not None and not np.isnan(pitch_hz)
        if valid_pitch:
            if start_index is None:
                start_index = index
                current_pitches = []
            current_pitches.append(float(librosa.hz_to_midi(float(pitch_hz))))
            continue

        if start_index is not None and current_pitches:
            start = float(times[start_index])
            end = float(times[index]) if index < len(
                times) else len(audio) / sample_rate
            if end - start >= 0.12:
                note_events.append({
                    "start": start,
                    "end": end,
                    "pitch": int(round(float(np.median(current_pitches)))),
                    "velocity": 96,
                })
        start_index = None
        current_pitches = []

    if start_index is not None and current_pitches:
        start = float(times[start_index])
        end = len(audio) / sample_rate
        if end - start >= 0.12:
            note_events.append({
                "start": start,
                "end": end,
                "pitch": int(round(float(np.median(current_pitches)))),
                "velocity": 96,
            })

    if note_events:
        return note_events

    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    average = np.mean(chroma, axis=1)
    dominant_pitch = int(np.argmax(average))
    return [{
        "start": 0.0,
        "end": max(0.4, len(audio) / sample_rate),
        "pitch": 60 + dominant_pitch,
        "velocity": 90,
    }]


def _estimate_tempo(note_events: list[dict[str, float | int]], midi_bytes: Optional[bytes] = None) -> float:
    if pretty_midi is not None and midi_bytes:
        temp_midi_path: Optional[Path] = None
        try:  # pragma: no cover
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
                handle.write(midi_bytes)
                temp_midi_path = Path(handle.name)
            midi_obj = pretty_midi.PrettyMIDI(str(temp_midi_path))
            raw_bpm = float(midi_obj.estimate_tempo())
            while raw_bpm > 180:
                raw_bpm /= 2
            while raw_bpm < 60:
                raw_bpm *= 2
            return raw_bpm
        except Exception:
            pass
        finally:
            if temp_midi_path and temp_midi_path.exists():
                temp_midi_path.unlink(missing_ok=True)

    starts = sorted({float(event["start"]) for event in note_events})
    if len(starts) < 2:
        return 90.0
    intervals = np.diff(starts)
    valid = intervals[(intervals > 0.08) & (intervals < 2.0)]
    if valid.size == 0:
        return 90.0
    bpm = float(60.0 / np.median(valid))
    while bpm < 60:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return bpm


def _note_events_from_midi_bytes(midi_bytes: bytes) -> list[dict[str, float | int]]:
    if pretty_midi is None:
        return []
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
            handle.write(midi_bytes)
            temp_path = Path(handle.name)
        pm = pretty_midi.PrettyMIDI(str(temp_path))
        events: list[dict[str, float | int]] = []
        for inst in pm.instruments:
            for note in inst.notes:
                events.append({"start": note.start, "end": note.end, "pitch": note.pitch, "velocity": note.velocity})
        return events
    except Exception:
        return []
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _pitch_histogram(note_events: list[dict[str, float | int]]) -> list[float]:
    histogram = np.zeros(12, dtype=np.float32)
    for event in note_events:
        histogram[int(event["pitch"]) % 12] += 1
    total = float(histogram.sum())
    if total > 0:
        histogram /= total
    return histogram.round(4).tolist()


def _key_from_histogram(histogram: list[float]) -> str:
    hist = np.asarray(histogram, dtype=np.float32)
    if hist.sum() <= 0:
        return "Unknown"

    # Yeni kontrol: tüm değerler aynıysa korelasyon hesaplanamaz
    if np.std(hist) == 0:
        return "Unknown"

    best_label = "Unknown"
    best_score = float("-inf")

    for root_idx, root_name in enumerate(PITCH_CLASS_NAMES):
        major_score = float(np.corrcoef(
            hist, np.roll(MAJOR_PROFILE, root_idx))[0, 1])
        minor_score = float(np.corrcoef(
            hist, np.roll(MINOR_PROFILE, root_idx))[0, 1])

        # Güvenli karşılaştırma — NaN gelirse -inf'e çekilmeli
        if np.isnan(major_score):
            major_score = float("-inf")
        if np.isnan(minor_score):
            minor_score = float("-inf")

        if major_score > best_score:
            best_score = major_score
            best_label = f"{root_name} major"
        if minor_score > best_score:
            best_score = minor_score
            best_label = f"{root_name} minor"

    return best_label


def _extract_key_label(midi_bytes: bytes, histogram: list[float]) -> str:
    if m21 is None:
        return _key_from_histogram(histogram)

    temp_midi_path: Optional[Path] = None
    try:  # pragma: no cover
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
            handle.write(midi_bytes)
            temp_midi_path = Path(handle.name)
        parsed = m21.converter.parse(str(temp_midi_path))
        return str(parsed.analyze("key"))
    except Exception:
        return _key_from_histogram(histogram)
    finally:
        if temp_midi_path and temp_midi_path.exists():
            temp_midi_path.unlink(missing_ok=True)




def _default_variant_temperatures(n_variants: int) -> list[float]:
    if n_variants == 4:
        return NOTEBOOK_VARIANT_DEFAULT_TEMPERATURES.copy()
    return [round(0.7 + 0.2 * index, 2) for index in range(n_variants)]


def _parse_variant_temperatures(raw_temperatures: Optional[str], n_variants: int) -> list[float]:
    if raw_temperatures is None or not raw_temperatures.strip():
        return _default_variant_temperatures(n_variants)

    try:
        parsed = json.loads(raw_temperatures)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid temperatures JSON") from exc

    if not isinstance(parsed, list) or not all(isinstance(value, (int, float)) for value in parsed):
        raise HTTPException(
            status_code=400, detail="Invalid temperatures JSON")

    temperatures = [float(value) for value in parsed]
    if len(temperatures) != n_variants:
        raise HTTPException(
            status_code=400, detail="temperatures array length must equal n_variants")
    return temperatures


def _audio_to_iddm_mel(audio_bytes: bytes, cfg: dict[str, Any]) -> torch.Tensor:
    sample_rate = NOTEBOOK_VARIANT_AUDIO_DEFAULTS["sample_rate"]
    n_fft = NOTEBOOK_VARIANT_AUDIO_DEFAULTS["n_fft"]
    hop_length = NOTEBOOK_VARIANT_AUDIO_DEFAULTS["hop_length"]
    mel_bins = int(
        cfg.get("mel_bins", NOTEBOOK_VARIANT_AUDIO_DEFAULTS["mel_bins"]))
    t_win = int(cfg.get("T_win", NOTEBOOK_VARIANT_AUDIO_DEFAULTS["T_win"]))

    audio = _read_audio_bytes(audio_bytes, sample_rate)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mel_bins,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    mel_window = mel_db[:, :t_win]
    if mel_window.shape[1] < t_win:
        mel_window = np.pad(
            mel_window, ((0, 0), (0, t_win - mel_window.shape[1])))
    return torch.tensor(mel_window[np.newaxis, ...], dtype=torch.float32, device=DEVICE)


def _token_ids_to_midi_bytes(token_ids: list[int], bpm: float) -> bytes:
    temp_midi_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
            temp_midi_path = Path(handle.name)
        tokens_to_midi(token_ids, bpm=bpm, out_path=temp_midi_path)
        return temp_midi_path.read_bytes()
    finally:
        if temp_midi_path and temp_midi_path.exists():
            temp_midi_path.unlink(missing_ok=True)


def _transcribe_and_mood(audio_bytes: bytes) -> dict[str, Any]:
    sample_rate = NOTEBOOK_VARIANT_AUDIO_DEFAULTS["sample_rate"]
    audio = _read_audio_bytes(audio_bytes, sample_rate)
    duration_sec = float(len(audio) / sample_rate)

    temp_audio_path: Optional[Path] = None
    note_events: list[dict[str, float | int]] = []
    midi_bytes: Optional[bytes] = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_audio_path = Path(handle.name)
        sf.write(str(temp_audio_path), np.asarray(
            audio, dtype=np.float32), sample_rate)
        if basic_pitch_predict is not None:
            try:
                midi_bytes, note_events = _run_basic_pitch_predict(
                    str(temp_audio_path))
            except Exception:
                logger.exception(
                    "Basic Pitch inference failed, falling back to heuristic notes")
                midi_bytes = None
                note_events = []
    finally:
        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)

    if not note_events:
        note_events = _fallback_note_events_from_audio(audio, sample_rate)

    if midi_bytes is None:
        estimated_tempo = _estimate_tempo(note_events)
        midi_bytes = _tokens_to_midi_bytes(
            note_events, bpm=estimated_tempo, instrument=0)

    tempo_bpm = float(_estimate_tempo(note_events, midi_bytes))
    avg_pitch = float(np.mean([int(note["pitch"])
                      for note in note_events])) if note_events else 60.0
    pitch_histogram = _pitch_histogram(note_events)
    key = _extract_key_label(midi_bytes, pitch_histogram)
    mood_idx, mood_label = heuristic_mood_from_metrics(tempo_bpm, avg_pitch, key)
    detected_chords = _detect_chords_from_audio(audio, sample_rate)

    return {
        "midi_bytes": midi_bytes,
        "note_events": note_events,
        "n_notes": len(note_events),
        "duration_sec": duration_sec,
        "tempo_bpm": tempo_bpm,
        "avg_pitch": avg_pitch,
        "mood_idx": mood_idx,
        "mood_label": mood_label,
        "key": key,
        "pitch_histogram": pitch_histogram,
        "detected_chords": detected_chords,
    }


def generate_from_audio(audio_bytes_input: bytes, creativity: float = 0.7, chord: Optional[str] = None) -> tuple[list[list[float]], bytes, list[str]]:
    cfg = _load_audio_cfg()
    sample_rate = int(cfg["audio"]["sample_rate"])
    audio = _read_audio_bytes(audio_bytes_input, sample_rate)
    detected_chords = _detect_chords_from_audio(audio, sample_rate)
    mel_input = _waveform_to_mel(audio, cfg)

    try:
        bundle = _load_cvae_iddm()
    except HTTPException:
        return mel_input.tolist(), _waveform_to_wav_bytes(audio, sample_rate), detected_chords

    try:
        mel_tensor = _encode(audio, cfg)
        with torch.no_grad():
            mu, logvar = bundle["cvae_model"].encode(mel_tensor)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std * max(0.05, float(creativity))
            mel_output = _decode_from_latent(bundle["cvae_model"], z)
        return mel_output.tolist(), _mel_to_audio_bytes(mel_output, cfg), detected_chords
    except Exception as exc:
        logger.warning("CVAE inference failed: %s", exc, exc_info=True)
        return mel_input.tolist(), _waveform_to_wav_bytes(audio, sample_rate), detected_chords


def generate_from_chord(
    chord: str,
    duration: float = 5.0,
    bpm: float = 120.0,
    instrument: int = 0,
) -> tuple[list[list[float]], bytes, list[str]]:
    cfg = _audio_cfg()
    sample_rate = int(cfg["audio"]["sample_rate"])
    chord_duration_beats = max(1.0, duration / (60.0 / max(bpm, 1.0)))
    note_events = _note_events_for_progression(
        [chord], bpm=bpm, temperature=1.0, chord_beats=chord_duration_beats)
    waveform = _synthesize_note_events_to_waveform(note_events, sample_rate)
    return _waveform_to_mel(waveform, cfg).tolist(), _waveform_to_wav_bytes(waveform, sample_rate), [chord]


def _generate_progression_payload(progression: list[str], bpm: float, instrument: int, prefix: str) -> dict[str, Any]:
    cfg = _audio_cfg()
    sample_rate = int(cfg["audio"]["sample_rate"])
    note_events = _note_events_for_progression(progression, bpm=bpm)
    midi_bytes = _tokens_to_midi_bytes(
        note_events, bpm=bpm, instrument=instrument)
    midi_filename, _ = _save_bytes(midi_bytes, prefix, ".mid")

    waveform = _synthesize_note_events_to_waveform(note_events, sample_rate)
    wav_bytes = _waveform_to_wav_bytes(waveform, sample_rate)
    wav_filename, _ = _save_bytes(wav_bytes, prefix, ".wav")

    return {
        "progression": progression,
        "bpm": bpm,
        "instrument": instrument,
        "audio_b64": base64.b64encode(wav_bytes).decode("utf-8"),
        "wav_b64": base64.b64encode(wav_bytes).decode("utf-8"),
        "filename": wav_filename,
        "download_path": f"/api/download/{wav_filename}",
        "midi_b64": base64.b64encode(midi_bytes).decode("utf-8"),
        "midi_filename": midi_filename,
        "midi_download_path": f"/api/download/{midi_filename}",
        "wav_filename": wav_filename,
        "wav_download_path": f"/api/download/{wav_filename}",
        "detected_chords": progression,
    }


def generate_iddm_variants(
    audio_bytes: bytes,
    n_variants: int,
    temperatures: list[float],
) -> dict[str, Any]:
    try:
        bundle = _load_cvae_iddm()
        transcription = _transcribe_and_mood(audio_bytes)
        mel_window = _audio_to_iddm_mel(audio_bytes, bundle["cfg"])
        cfg = bundle["cfg"]
        latent_dim = int(cfg.get("latent_dim", 32))
        n_moods = int(cfg.get("n_moods", 3))
        seq_len = int(cfg.get("seq_len", 129))
        n_bar_bins = int(cfg.get("n_bar_bins", 8))
        sample_rate = NOTEBOOK_VARIANT_AUDIO_DEFAULTS["sample_rate"]
        variants: list[dict[str, Any]] = []

        # Build seed tokens from transcription MIDI (used for base_mu and chord/bar conditioning)
        seed_tokens: list[int] = []
        if pretty_midi is not None and transcription.get("midi_bytes"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                    tmp.write(transcription["midi_bytes"])
                    tmp_path = Path(tmp.name)
                seed_tokens = midi_to_tokens(tmp_path, max_notes=64)
                tmp_path.unlink(missing_ok=True)
            except Exception:
                seed_tokens = []

        # Pad or truncate seed tokens to seq_len
        if len(seed_tokens) < seq_len:
            from .model.colab_parity import PAD_TOKEN
            seed_tokens = seed_tokens + [PAD_TOKEN] * (seq_len - len(seed_tokens))
        else:
            seed_tokens = seed_tokens[:seq_len]

        seed_t = torch.tensor(seed_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
        mood_idx = int(transcription["mood_idx"])
        chord_idx_val = tokens_to_chord_idx(seed_tokens)
        bar_idx_val = tokens_to_bar_idx(seed_tokens, n_bins=n_bar_bins)

        with torch.no_grad():
            s_enc = bundle["state_enc"](mel_window)
            dist, _ = bundle["ac"].dist_and_value(s_enc)
            base_mu = bundle["cvae_model"].encode_mu(seed_t)
            mood_oh = build_mood_onehot([mood_idx], n_moods=n_moods, device=DEVICE)
            chord_t = torch.tensor([chord_idx_val], dtype=torch.long, device=DEVICE)
            bar_t = torch.tensor([bar_idx_val], dtype=torch.long, device=DEVICE)
            cond = bundle["cvae_model"].build_cond(mood_oh, chord_t, bar_t)
            alpha = bundle["alpha"].clamp(0.0, 1.0)

            for index, temperature in enumerate(temperatures[:n_variants]):
                temp = float(temperature)
                delta_z = dist.loc + dist.scale * torch.randn_like(dist.loc) * temp
                z_final = base_mu + alpha * delta_z
                gen_tokens, _, _ = bundle["cvae_model"].decoder.sample(
                    torch.cat([z_final, cond], dim=-1),
                    seq_len=seq_len,
                    temperature=temp,
                )
                midi_bytes = _token_ids_to_midi_bytes(
                    gen_tokens.squeeze(0).detach().cpu().tolist(),
                    bpm=120.0,
                )
                midi_filename, _ = _save_bytes(
                    midi_bytes, f"variant_{index + 1}", ".mid")
                wav_b64 = _midi_bytes_to_wav_b64(
                    midi_bytes,
                    sample_rate=sample_rate,
                    note_events=None,
                    prefer_fluidsynth_only=False,
                )
                wav_filename = ""
                wav_download_path = ""
                if wav_b64 is not None:
                    wav_bytes = base64.b64decode(wav_b64)
                    wav_filename, _ = _save_bytes(
                        wav_bytes, f"variant_{index + 1}", ".wav")
                    wav_download_path = f"/api/download/{wav_filename}"
                variants.append({
                    "index": index,
                    "temperature": temp,
                    "midi_b64": base64.b64encode(midi_bytes).decode("utf-8"),
                    "midi_filename": midi_filename,
                    "midi_download_path": f"/api/download/{midi_filename}",
                    "wav_b64": wav_b64,
                    "wav_filename": wav_filename,
                    "wav_download_path": wav_download_path,
                })

        return {
            "n_variants": n_variants,
            "temperatures": temperatures[:n_variants],
            "mood_idx": transcription["mood_idx"],
            "mood_label": transcription["mood_label"],
            "model_status": _variant_model_status(),
            "variants": variants,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected Colab parity inference error:\n%s",
                     traceback.format_exc())
        raise HTTPException(
            status_code=500, detail="Variant generation failed") from exc


def run_basic_pitch(file_bytes: bytes, original_filename: str) -> dict[str, Any]:
    _ = original_filename
    transcription = _transcribe_and_mood(file_bytes)
    sample_rate = NOTEBOOK_VARIANT_AUDIO_DEFAULTS["sample_rate"]
    midi_filename, _ = _save_bytes(
        transcription["midi_bytes"], "transcription", ".mid")
    wav_b64 = _midi_bytes_to_wav_b64(
        transcription["midi_bytes"],
        sample_rate=sample_rate,
        note_events=transcription.get("note_events"),  # parsed lazily if FluidSynth unavailable
        prefer_fluidsynth_only=False,
    )
    wav_filename = ""
    if wav_b64 is not None:
        wav_bytes = base64.b64decode(wav_b64)
        wav_filename, _ = _save_bytes(wav_bytes, "transcription", ".wav")

    return {
        "n_notes": int(transcription["n_notes"]),
        "duration_sec": round(float(transcription["duration_sec"]), 2),
        "midi_b64": base64.b64encode(transcription["midi_bytes"]).decode("utf-8"),
        "wav_b64": wav_b64,
        "midi_filename": midi_filename,
        "wav_filename": wav_filename,
        "mood_label": transcription["mood_label"],
        "mood_idx": int(transcription["mood_idx"]),
        "detected_chords": transcription["detected_chords"],
        "key": transcription["key"],
        "pitch_histogram": transcription["pitch_histogram"],
        "tempo_bpm": round(float(transcription["tempo_bpm"]), 2),
        "average_pitch": round(float(transcription["avg_pitch"]), 2),
    }


async def handle_generate_request(
    filename: Optional[str] = None,
    upload_id: Optional[str] = None,
    chord: Optional[str] = None,
    creativity: float = 0.7,
    duration: Optional[float] = None,
    bpm: float = 120.0,
    instrument: int = 0,
) -> dict[str, Any]:
    if chord and not filename and not upload_id:
        mel, audio_bytes, detected_chords = generate_from_chord(
            chord=chord,
            duration=duration or 5.0,
            bpm=bpm,
            instrument=instrument,
        )
        payload = _package_audio_bytes(audio_bytes, "chord")
        payload["detected_chords"] = detected_chords
        payload["mel"] = mel
        return payload

    input_path = _resolve_upload_path(filename, upload_id)
    if input_path is None:
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    mel, audio_bytes, detected_chords = generate_from_audio(
        input_path.read_bytes(),
        creativity=creativity,
        chord=chord,
    )
    payload = _package_audio_bytes(audio_bytes, "generated")
    payload["detected_chords"] = detected_chords
    payload["mel"] = mel
    return payload


def _rerender_midi_bytes(midi_bytes: bytes, instrument: int, bpm: float) -> bytes:
    if pretty_midi is None:
        return midi_bytes

    temp_in: Optional[Path] = None
    temp_out: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
            handle.write(midi_bytes)
            temp_in = Path(handle.name)

        src = pretty_midi.PrettyMIDI(str(temp_in))
        _, tempo_vals = src.get_tempo_changes()
        original_bpm = float(tempo_vals[0]) if len(tempo_vals) > 0 else 120.0
        time_scale = original_bpm / max(bpm, 1.0)

        dst = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
        program = max(0, min(127, int(instrument)))

        for src_inst in src.instruments:
            dst_inst = pretty_midi.Instrument(
                program=program,
                is_drum=src_inst.is_drum,
                name=src_inst.name,
            )
            for note in src_inst.notes:
                dst_inst.notes.append(pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start * time_scale,
                    end=note.end * time_scale,
                ))
            dst.instruments.append(dst_inst)

        if not dst.instruments:
            dst_inst = pretty_midi.Instrument(program=program)
            dst.instruments.append(dst_inst)

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
            temp_out = Path(handle.name)
        dst.write(str(temp_out))
        return temp_out.read_bytes()
    except Exception:
        logger.warning("_rerender_midi_bytes failed, returning original", exc_info=True)
        return midi_bytes
    finally:
        if temp_in and temp_in.exists():
            temp_in.unlink(missing_ok=True)
        if temp_out and temp_out.exists():
            temp_out.unlink(missing_ok=True)


def download_generated_file(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Generated file not found")
    media_type = "audio/wav"
    if file_path.suffix.lower() == ".mid":
        media_type = "audio/midi"
    return FileResponse(file_path, filename=safe_name, media_type=media_type)


@router.get("/model-status")
def model_status() -> dict[str, Any]:
    return _variant_model_status()


@router.get("/chords")
def chords() -> dict[str, list[str]]:
    return {"chords": get_available_chords()}


@router.get("/progressions")
def progressions(genre: Optional[str] = None) -> dict[str, Any]:
    result = PRESET_PROGRESSIONS
    if genre:
        result = [p for p in result if p.get("genre", "").lower() == genre.lower()]
    return {"progressions": result}


@router.get("/download/{filename}")
def download(filename: str) -> FileResponse:
    return download_generated_file(filename)


@router.post("/generate")
async def generate_audio_route(
    filename: Optional[str] = Body(None),
    id: Optional[str] = Body(None),
    chord: Optional[str] = Body(None),
    creativity: float = Body(0.7),
    duration: Optional[float] = Body(None),
    bpm: float = Body(120.0),
    instrument: int = Body(0),
) -> dict[str, Any]:
    return await handle_generate_request(
        filename=filename,
        upload_id=id,
        chord=chord,
        creativity=creativity,
        duration=duration,
        bpm=bpm,
        instrument=instrument,
    )


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await _read_upload_bytes(file)
    return run_basic_pitch(raw, file.filename or "audio.wav")


@router.post("/generate-progression")
async def generate_progression_route(
    progression: list[str] = Body(...),
    bpm: float = Body(120.0),
    instrument: int = Body(0),
) -> dict[str, Any]:
    if not progression:
        raise HTTPException(
            status_code=400, detail="Progression must include at least one chord")
    cleaned_progression = [chord.strip()
                           for chord in progression if chord.strip()]
    if not cleaned_progression:
        raise HTTPException(
            status_code=400, detail="Progression must include at least one chord")
    return _generate_progression_payload(cleaned_progression, bpm=float(bpm), instrument=int(instrument), prefix="progression")


@router.post("/generate-variants")
async def generate_variants_route(
    file: UploadFile = File(...),
    n_variants: int = Form(4),
    temperatures: Optional[str] = Form(None),
) -> dict[str, Any]:
    count = max(1, min(8, int(n_variants)))
    raw = await _read_upload_bytes(file)
    return generate_iddm_variants(
        audio_bytes=raw,
        n_variants=count,
        temperatures=_parse_variant_temperatures(temperatures, count),
    )


@router.post("/generate-accompaniment")
async def generate_accompaniment(
    vocal: UploadFile = File(...),
    creativity: float = Form(0.6),
) -> JSONResponse:
    try:
        raw = vocal.file.read()
        _, audio_bytes, _ = generate_from_audio(
            raw, creativity=float(creativity))
        return JSONResponse(_package_audio_bytes(audio_bytes, "accomp"))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Accompaniment generation failed", exc_info=exc)
        raise HTTPException(
            status_code=500, detail="Accompaniment generation failed") from exc


@router.post("/generate-style")
async def generate_style(
    vocal: UploadFile = File(...),
    style: UploadFile = File(...),
    style_mix: float = Form(0.4),
    creativity: float = Form(0.6),
) -> JSONResponse:
    try:
        cfg = _audio_cfg()
        sample_rate = int(cfg["audio"]["sample_rate"])
        vocal_audio = _read_audio_from_upload(vocal, sample_rate)
        style_audio = _read_audio_from_upload(style, sample_rate)
        mixed_audio = librosa.util.normalize(
            (1.0 - float(np.clip(style_mix, 0.0, 1.0))) *
            vocal_audio[: min(len(vocal_audio), len(style_audio))]
            + float(np.clip(style_mix, 0.0, 1.0)) *
            style_audio[: min(len(vocal_audio), len(style_audio))]
        )
        audio_bytes = _waveform_to_wav_bytes(mixed_audio, sample_rate)
        return JSONResponse(_package_audio_bytes(audio_bytes, "style"))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Stylized generation failed", exc_info=exc)
        raise HTTPException(
            status_code=500, detail="Stylized generation failed") from exc


@router.post("/render-midi")
async def render_midi_route(
    midi_b64: str = Body(...),
    instrument: int = Body(0),
    bpm: float = Body(120.0),
) -> dict[str, Any]:
    try:
        midi_bytes = base64.b64decode(midi_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid midi_b64") from exc

    sample_rate = NOTEBOOK_VARIANT_AUDIO_DEFAULTS["sample_rate"]
    rerendered = _rerender_midi_bytes(midi_bytes, instrument=instrument, bpm=bpm)
    wav_b64 = _midi_bytes_to_wav_b64(rerendered, sample_rate=sample_rate, prefer_fluidsynth_only=True)
    return {"wav_b64": wav_b64}


@router.post("/generate-lanes")
async def generate_lanes(
    audio: UploadFile = File(...),
    keep_bass: bool = Form(True),
    keep_piano: bool = Form(True),
    keep_drums: bool = Form(True),
    keep_other: bool = Form(True),
) -> JSONResponse:
    try:
        cfg = _audio_cfg()
        sample_rate = int(cfg["audio"]["sample_rate"])
        waveform = _read_audio_from_upload(audio, sample_rate)
        harmonic, percussive = librosa.effects.hpss(waveform)
        mix = np.zeros_like(waveform)
        if keep_drums:
            mix += percussive
        if keep_bass or keep_piano or keep_other:
            mix += harmonic
        mix = librosa.util.normalize(mix if np.any(mix) else waveform)
        return JSONResponse(_package_audio_bytes(_waveform_to_wav_bytes(mix, sample_rate), "lanes"))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Lane masking failed", exc_info=exc)
        raise HTTPException(
            status_code=500, detail="Lane masking failed") from exc
