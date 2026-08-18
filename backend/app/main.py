from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import inference
from app.schemas import ProcessRequest


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "recordings"
OUTPUT_DIR = DATA_DIR / "generated"
LOG_DIR = DATA_DIR / "logs"

for directory in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    inference.cleanup_stale_files()
    cleanup_task = asyncio.create_task(inference.run_periodic_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(
    title="Musical VAE Playground",
    description="AI-powered audio processing for musicians",
    version="1.0.0",
    lifespan=lifespan,
)

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# No public static mount: data/recordings, data/generated, and data/logs all
# hold user-uploaded/generated content and were previously exposed under
# /static with no auth or sanitization, despite nothing in the app ever
# using that route. Generated files are served through the sanitized
# /api/download/{filename} endpoint instead.
#
# All routes are canonicalized under /api/*, served by inference.router
# below (chords, download, generate, generate-variants, generate-progression,
# transcribe). /api/upload, /model-info, /process/, and /health are the only
# routes registered directly on `app`.
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


@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(...),
    sample_rate: Optional[int] = None,
) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm")):
        raise HTTPException(
            status_code=400, detail="Supported formats: wav, mp3, flac, ogg, m4a, webm")

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


@app.get("/model-info")
def get_model_info() -> dict[str, Any]:
    return inference.get_model_info()


@app.post("/process/")
async def process_audio(payload: ProcessRequest) -> dict[str, Any]:
    return await inference.handle_generate_request(
        upload_id=payload.id,
        creativity=payload.creativity,
    )
