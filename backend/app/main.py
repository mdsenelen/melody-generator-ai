from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import inference
from app.chord_utils import get_all_chord_labels


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "recordings"
OUTPUT_DIR = DATA_DIR / "generated"
LOG_DIR = DATA_DIR / "logs"

for directory in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Musical VAE Playground",
    description="AI-powered audio processing for musicians",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")
app.include_router(inference.router, prefix="/api")


def log_event(event_id: str, data: dict[str, Any]) -> None:
    try:
        path = LOG_DIR / f"{event_id}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        print(f"Failed to log event {event_id}: {exc}")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "runtime": inference.get_runtime_status(),
    }


@app.post("/upload/")
async def upload_audio(
    file: UploadFile = File(...),
    sample_rate: Optional[int] = None,
) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm")):
        raise HTTPException(status_code=400, detail="Supported formats: wav, mp3, flac, ogg, m4a, webm")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    unique_id = uuid.uuid4().hex
    suffix = Path(file.filename).suffix.lower() or ".wav"
    filename = f"upload_{unique_id}{suffix}"
    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(raw)

    log_event(unique_id, {
        "id": unique_id,
        "event": "upload",
        "filename": filename,
        "sample_rate": sample_rate or inference.get_model_info()["audio_config"]["sample_rate"],
        "timestamp": datetime.now().isoformat(),
    })

    return JSONResponse(status_code=201, content={"id": unique_id, "filename": filename})


@app.post("/api/upload")
async def api_upload_audio(
    file: UploadFile = File(...),
    sample_rate: Optional[int] = None,
) -> JSONResponse:
    return await upload_audio(file=file, sample_rate=sample_rate)


@app.get("/model-info")
def get_model_info() -> dict[str, Any]:
    return inference.get_model_info()


@app.get("/chords")
def get_chords() -> dict[str, list[str]]:
    return {"chords": get_all_chord_labels()}


@app.get("/download/{filename}")
def download_generated_file(filename: str):
    return inference.download_generated_file(filename)


@app.post("/generate")
async def generate_audio(
    filename: Optional[str] = Body(None),
    id: Optional[str] = Body(None),
    chord: Optional[str] = Body(None),
    creativity: float = Body(0.7),
    duration: Optional[float] = Body(None),
    bpm: float = Body(120.0),
    instrument: int = Body(0),
) -> dict[str, Any]:
    return await inference.handle_generate_request(
        filename=filename,
        upload_id=id,
        chord=chord,
        creativity=creativity,
        duration=duration,
        bpm=bpm,
        instrument=instrument,
    )


@app.post("/process/")
async def process_audio(
    id: str = Body(...),
    intensity: float = Body(0.5),
    creativity: float = Body(0.7),
) -> dict[str, Any]:
    return await inference.handle_generate_request(
        upload_id=id,
        creativity=creativity,
    )
