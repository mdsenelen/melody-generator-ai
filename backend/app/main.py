# 📁 backend/main.py
import os
import uuid
import json
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from typing import Optional
import base64

from app.model.vae import WebVAE
from app.model.utils import AudioProcessor
from app.inference import generate_from_audio, generate_from_chord
from app.chord_utils import get_all_chord_labels

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent             # backend/
APP_DIR = BASE_DIR / "app"                             # backend/app
MODEL_DIR = APP_DIR / "model" / "weights"              # backend/app/model/weights

MODEL_CONFIG = {
    "input_shape": (1, 128, 256),
    "latent_dim": 256,
    "model_path": str(MODEL_DIR / "web_model.pt"),
    # adjust if different
    "config_path": str(APP_DIR / "model" / "weights" / "audio_params.json"),
}

app = FastAPI(
    title="Musical VAE Playground",
    description="AI-powered audio processing for musicians",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory Setup
UPLOAD_DIR = "data/recordings"
OUTPUT_DIR = "data/generated"
LOG_DIR = "data/logs"
MODEL_DIR = os.path.join("backend", "app", "model", "weights")  # Updated path

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="data"), name="static")

# Configuration
MODEL_CONFIG = {
    "input_shape": (1, 128, 256),  # Updated from (1, 64, 256)
    "latent_dim": 256,              # Updated from 128
    "model_path": os.path.join(MODEL_DIR, "web_model.pt"),
    "config_path": os.path.join(MODEL_DIR, "audio_params.json")
}


def log_event(event_id: str, data: dict):
    """Enhanced logging with error handling"""
    try:
        path = os.path.join(LOG_DIR, f"{event_id}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to log event: {str(e)}")


def load_model():
    """Updated model loading with new architecture"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load model config
        with open(MODEL_CONFIG["config_path"], "r") as f:
            audio_config = json.load(f)

        # Initialize model
        model = WebVAE().to(device)  # Now uses default constructor

        # Load weights
        checkpoint = torch.load(
            MODEL_CONFIG["model_path"], map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # Initialize processor with loaded config
        processor = AudioProcessor(**audio_config["audio"])

        return model, processor, device, audio_config

    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")


# Initialize model
try:
    model, audio_processor, device, audio_config = load_model()
    print("✅ Model loaded successfully")
    print(f"Model configuration: {audio_config}")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    raise


@app.post("/upload/")
async def upload_audio(
    file: UploadFile = File(...),
    sample_rate: Optional[int] = None
):
    """Handle audio uploads with validation"""
    if not file.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(400, "Only .wav or .mp3 files are accepted")

    unique_id = uuid.uuid4().hex
    filename = f"upload_{unique_id}{os.path.splitext(file.filename)[1]}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        log_data = {
            "id": unique_id,
            "event": "upload",
            "filename": filename,
            "sample_rate": sample_rate or audio_config["audio"]["sample_rate"],
            "timestamp": datetime.now().isoformat()
        }
        log_event(unique_id, log_data)

        return JSONResponse(
            status_code=201,
            content={"id": unique_id, "filename": filename}
        )
    except Exception as e:
        raise HTTPException(500, f"File upload failed: {str(e)}")


@app.post("/api/upload")
async def api_upload_audio(
    file: UploadFile = File(...),
    sample_rate: Optional[int] = None
):
    # Reuse the existing upload_audio logic
    return await upload_audio(file=file, sample_rate=sample_rate)


@app.get("/model-info")
def get_model_info():
    """Updated model information endpoint"""
    return {
        "model": "WebVAE",
        "latent_dim": MODEL_CONFIG["latent_dim"],
        "input_shape": MODEL_CONFIG["input_shape"],
        "audio_config": audio_config["audio"],
        "device": str(device),
        "status": "active"
    }


@app.get("/chords", response_class=JSONResponse)
def get_chords():
    """API endpoint to get all chord labels."""
    return {"chords": get_all_chord_labels()}


@app.post("/generate")
async def generate_audio(
    id: Optional[str] = Body(None, embed=True),
    chord: Optional[str] = Body(None),
    creativity: float = Body(0.7),
    duration: Optional[float] = Body(None)
):
    """Enhanced generation endpoint with chord support and chord detection"""
    try:
        if chord and not id:
            mel, audio_bytes, detected_chords = generate_from_chord(
                chord=chord,
                duration=duration or 5.0
            )
            output_filename = f"chord_{chord}_{uuid.uuid4().hex[:6]}.wav"
        else:
            if not id:
                raise HTTPException(
                    400, "ID is required unless generating from chord only")

            input_filename = f"upload_{id}.wav"
            input_path = os.path.join(UPLOAD_DIR, input_filename)

            if not os.path.exists(input_path):
                raise HTTPException(404, "Uploaded file not found")

            with open(input_path, "rb") as f:
                audio_bytes_input = f.read()

            mel, audio_bytes, detected_chords = generate_from_audio(
                audio_bytes_input,
                creativity=creativity,
                chord=chord
            )
            output_filename = f"generated_{id}.wav"

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        log_event(f"gen_{id or chord}", {
            "type": "chord" if chord and not id else "audio",
            "id": id or chord,
            "output_file": output_filename,
            "timestamp": datetime.now().isoformat(),
            "creativity": creativity,
            "duration": duration,
            "chords": detected_chords
        })

        return {
            "mel": mel,
            "audio": base64.b64encode(audio_bytes).decode('utf-8'),
            "download_url": f"/download/{output_filename}",
            "chords": detected_chords
        }
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.post("/process/")
async def process_audio(
    id: str = Body(...),
    intensity: float = 0.5,
    creativity: float = 0.7
):
    """Legacy endpoint - consider deprecating"""
    return await generate_audio(id=id, creativity=creativity)
