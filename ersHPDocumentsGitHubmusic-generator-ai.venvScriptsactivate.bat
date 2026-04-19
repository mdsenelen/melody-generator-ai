diff --git a/README.md b/README.md
index e69de29..7b75f21 100644
--- a/README.md
+++ b/README.md
@@ -0,0 +1,6 @@
+# Music Generator AI
+
+## Backend notes
+
+- Install the FluidSynth system package and a compatible SoundFont if you want MIDI-to-WAV preview synthesis from transcription outputs.
+- CPU PyTorch wheels are referenced from `backend/requirements.txt` via the PyTorch CPU index.
diff --git a/backend/app/__pycache__/inference.cpython-312.pyc b/backend/app/__pycache__/inference.cpython-312.pyc
index 1a7a2b7..f1323f6 100644
Binary files a/backend/app/__pycache__/inference.cpython-312.pyc and b/backend/app/__pycache__/inference.cpython-312.pyc differ
diff --git a/backend/app/__pycache__/main.cpython-312.pyc b/backend/app/__pycache__/main.cpython-312.pyc
index d1f8f8c..8eb2236 100644
Binary files a/backend/app/__pycache__/main.cpython-312.pyc and b/backend/app/__pycache__/main.cpython-312.pyc differ
diff --git a/backend/app/chord_utils.py b/backend/app/chord_utils.py
index 99e2626..b25c1ed 100644
--- a/backend/app/chord_utils.py
+++ b/backend/app/chord_utils.py
@@ -1,23 +1,53 @@
-# backend/app/chord_utils.py
-import torch
-import os
-
-# Path to web_model.pt
-MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'web_model.pt')
-MODEL_PATH = os.path.abspath(MODEL_PATH)
-
-# Load chord_vocab from web_model.pt
-_model_data = torch.load(MODEL_PATH, map_location='cpu')
-chord_vocab = _model_data.get('chord_vocab', [])
-
-
-def get_chord_label(index: int) -> str:
-    """Map chord index to chord label."""
-    if 0 <= index < len(chord_vocab):
-        return chord_vocab[index]
-    return None
-
-
-def get_all_chord_labels() -> list:
-    """Return all chord labels."""
-    return chord_vocab
+from __future__ import annotations
+
+from pathlib import Path
+
+import torch
+
+
+DEFAULT_CHORDS = [
+    "C",
+    "Cm",
+    "D",
+    "Dm",
+    "E",
+    "Em",
+    "F",
+    "G",
+    "Am",
+    "Bm",
+    "G7",
+    "Cmaj7",
+    "Am7",
+    "Dm7",
+]
+
+MODEL_PATH = Path(__file__).resolve().parent / "model" / "weights" / "web_model.pt"
+
+
+def _load_chord_vocab() -> list[str]:
+    if not MODEL_PATH.exists():
+        return DEFAULT_CHORDS
+
+    try:  # pragma: no cover
+        model_data = torch.load(str(MODEL_PATH), map_location="cpu")
+        chord_vocab = model_data.get("chord_vocab", [])
+        if isinstance(chord_vocab, list) and chord_vocab:
+            return chord_vocab
+    except Exception:
+        pass
+
+    return DEFAULT_CHORDS
+
+
+chord_vocab = _load_chord_vocab()
+
+
+def get_chord_label(index: int) -> str | None:
+    if 0 <= index < len(chord_vocab):
+        return chord_vocab[index]
+    return None
+
+
+def get_all_chord_labels() -> list[str]:
+    return chord_vocab
diff --git a/backend/app/inference.py b/backend/app/inference.py
index 0cc90f0..66a3424 100644
--- a/backend/app/inference.py
+++ b/backend/app/inference.py
@@ -1,262 +1,1473 @@
-# 📁 backend/app/inference.py
-import io
-import os
-import json
-import uuid
-import base64
-from typing import Optional, Dict, Any
-
-import numpy as np
-import soundfile as sf
-import torch
-import librosa
-
-from fastapi import APIRouter, UploadFile, File, Form, HTTPException
-from fastapi.responses import JSONResponse
-
-from .model.vae import WebVAE
-from .model.utils import AudioProcessor
-
-HERE = os.path.dirname(os.path.abspath(__file__))               # backend/app
-# backend/app/model/weights
-WEIGHTS_DIR = os.path.join(HERE, "model", "weights")
-# backend/data/generated
-OUTPUT_DIR = os.path.join(HERE, "..", "data", "generated")
-os.makedirs(OUTPUT_DIR, exist_ok=True)
-
-MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, "final_vocal2accomp.pth")
-AUDIO_CONFIG_PATH = os.path.join(WEIGHTS_DIR, "audio_params.json")
-DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
-
-DEFAULT_AUDIO = {
-    "audio": {
-        "sample_rate": 22050,
-        "n_fft": 2048,
-        "hop_length": 512,
-        "win_length": 1024,
-        "n_mels": 128,
-        "fmin": 30,
-        "fmax": 8000,
-        "max_frames": 256
-    }
-}
-AUDIO_CFG = (
-    json.load(open(AUDIO_CONFIG_PATH)) if os.path.exists(AUDIO_CONFIG_PATH)
-    else DEFAULT_AUDIO
-)
-
-processor = AudioProcessor(**AUDIO_CFG["audio"])
-model = WebVAE().to(DEVICE)
-if not os.path.exists(MODEL_WEIGHTS):
-    raise RuntimeError(f"❌ Missing model weights: {MODEL_WEIGHTS}")
-ckpt = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
-state_dict = ckpt["state_dict"] if isinstance(
-    ckpt, dict) and "state_dict" in ckpt else ckpt
-model.load_state_dict(state_dict)
-model.eval()
-
-
-def _read_audio_from_upload(file: UploadFile, target_sr: int) -> np.ndarray:
-    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
-        raise HTTPException(
-            status_code=400, detail="Supported formats: wav, mp3, flac, ogg, m4a")
-    raw = file.file.read()
-    if not raw:
-        raise HTTPException(status_code=400, detail="Empty file upload")
-    buf = io.BytesIO(raw)
-    y, _ = librosa.load(buf, sr=target_sr, mono=True)
-    y = librosa.util.normalize(y).astype(np.float32)
-    return y
-
-
-def _mel_to_audio_bytes(mel_norm: np.ndarray, cfg: Dict[str, Any]) -> bytes:
-    mel_db = (mel_norm * 40.0) - 40.0
-    mel_mag = librosa.db_to_amplitude(mel_db)
-    audio = librosa.griffinlim(
-        mel_mag,
-        hop_length=cfg["audio"]["hop_length"],
-        win_length=cfg["audio"]["win_length"],
-        n_iter=32
-    )
-    out = io.BytesIO()
-    sf.write(out, audio, cfg["audio"]["sample_rate"], format="WAV")
-    out.seek(0)
-    return out.read()
-
-
-def _encode(audio: np.ndarray) -> torch.Tensor:
-    mel = processor.waveform_to_mel(audio)
-    return torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
-
-
-def _decode_from_latent(z: torch.Tensor,
-                        cond_mem: Optional[torch.Tensor] = None,
-                        chord_idx: Optional[torch.Tensor] = None) -> np.ndarray:
-    with torch.no_grad():
-        try:
-            if chord_idx is not None and cond_mem is not None:
-                recon = model.decode(z, chord_idx, cond_mem)
-            elif chord_idx is not None:
-                recon = model.decode(z, chord_idx)
-            else:
-                recon = model.decode(z)
-        except TypeError:
-            try:
-                recon = model.decode(z)
-            except Exception:
-                recon, _, _ = model(z)
-    return recon.squeeze().detach().cpu().numpy()
-
-
-def _generate_accompaniment_core(vocal_audio: np.ndarray,
-                                 style_audio: Optional[np.ndarray] = None,
-                                 style_mix: float = 0.4,
-                                 creativity: float = 0.6) -> np.ndarray:
-    mel_vocal = _encode(vocal_audio)
-    with torch.no_grad():
-        mu_v, logvar_v = model.encode(mel_vocal)[:2] if hasattr(
-            model, "encode") else (None, None)
-        if mu_v is None:
-            recon_v, mu_v, logvar_v = model(mel_vocal)
-        std_v = torch.exp(0.5 * logvar_v)
-        eps = torch.randn_like(std_v)
-        z_v = mu_v + eps * std_v * max(0.05, creativity)
-
-        if style_audio is not None:
-            mel_style = _encode(style_audio)
-            mu_s, logvar_s = model.encode(mel_style)[:2] if hasattr(
-                model, "encode") else (None, None)
-            if mu_s is None:
-                recon_s, mu_s, logvar_s = model(mel_style)
-            std_s = torch.exp(0.5 * logvar_s)
-            eps_s = torch.randn_like(std_s)
-            z_s = mu_s + eps_s * std_s * max(0.05, creativity)
-            z = (1.0 - style_mix) * z_v + style_mix * z_s
-        else:
-            z = z_v
-
-        mel_out = _decode_from_latent(z)
-        return np.clip(mel_out, 0.0, 1.0)
-
-
-def _mel_lane_mask(mel: np.ndarray,
-                   sr: int,
-                   n_mels: int,
-                   fmin: int,
-                   fmax: int,
-                   