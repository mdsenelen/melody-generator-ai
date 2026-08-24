from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import inference
from app.jobs import routes as job_routes
from app.jobs.service import get_job_queue, get_job_store, get_object_storage
from app.jobs.worker import run_worker_loop
from app.request_limits import MaxBodySizeMiddleware, RequestBodyTooLarge
from app.schemas import ProcessRequest

logger = logging.getLogger(__name__)

# Dev/single-service default: run the transcription worker on a background
# thread inside the web process, so `uvicorn app.main:app --reload` alone
# is enough locally. Production must set this to "false" on the WEB
# service and run `python -m app.worker_main` as its own Render Background
# Worker service instead -- otherwise Basic Pitch inference is right back
# to running on a thread the web process spun up as a fire-and-forget
# workaround, which is the exact thing this job architecture replaces. See
# CLAUDE.md.
RUN_WORKER_IN_PROCESS = os.environ.get("RUN_WORKER_IN_PROCESS", "true").strip().lower() not in (
    "false", "0", "",
)
# Best-effort wait for the current job to reach a checkpoint on shutdown;
# the worker loop itself isn't forcibly interruptible mid-job (same
# fundamental limitation as GENERATION_TIMEOUT_SECONDS -- see
# inference.py's _run_generation), so this just bounds how long shutdown
# waits before moving on regardless.
DEFAULT_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0

# /api/upload used to read the whole request body into memory before any
# size check ran; a large-enough upload could exhaust memory or (now that
# B2 storage is billed) storage cost before any cap applied. Also covers
# the raw-file-upload fallback on /api/transcribe and /api/generate-variants,
# which had the same unbounded-read shape -- see MaxBodySizeMiddleware.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))


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

    # Only warm up (load) Basic Pitch in a process that will actually run
    # transcription work. In production with RUN_WORKER_IN_PROCESS=false,
    # the web process never runs the pipeline via the job workflow, so
    # eagerly loading the model here would be pure waste on an instance
    # that's supposed to stay light. (The legacy synchronous
    # /api/transcribe route is a separate, narrower caveat: if that route
    # is actually called, the web process will still lazy-load the model
    # on demand regardless of this flag -- gating the eager warm-up here
    # doesn't change that, only the startup-time cost.)
    warm_up_task = (
        asyncio.create_task(inference.warm_up_basic_pitch()) if RUN_WORKER_IN_PROCESS else None
    )

    worker_stop_event = threading.Event()
    worker_thread: Optional[threading.Thread] = None
    if RUN_WORKER_IN_PROCESS:
        worker_thread = threading.Thread(
            target=run_worker_loop,
            kwargs=dict(
                store=get_job_store(),
                queue=get_job_queue(),
                storage=get_object_storage(),
                stop_event=worker_stop_event,
            ),
            name="transcription-worker",
            daemon=True,
        )
        worker_thread.start()

    try:
        yield
    finally:
        cleanup_task.cancel()
        if warm_up_task is not None:
            warm_up_task.cancel()
        worker_stop_event.set()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task
        if warm_up_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await warm_up_task
        if worker_thread is not None:
            worker_thread.join(timeout=DEFAULT_WORKER_SHUTDOWN_TIMEOUT_SECONDS)


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

# Added after CORSMiddleware so it ends up outermost (Starlette's
# add_middleware inserts each new middleware ahead of the previous ones) --
# a body over the cap should be rejected before any other request handling,
# not just before the route body reads it.
app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_UPLOAD_BYTES)


@app.exception_handler(RequestBodyTooLarge)
async def handle_request_body_too_large(request: Any, exc: RequestBodyTooLarge) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": f"Request body exceeds the {exc.max_bytes // (1024 * 1024)}MB upload limit"},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Any, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )

# No public static mount: data/recordings, data/generated, and data/logs all
# hold user-uploaded/generated content and were previously exposed under
# /static with no auth or sanitization, despite nothing in the app ever
# using that route. Generated files are served through the sanitized
# /api/download/{filename} endpoint instead.
#
# All routes are canonicalized under /api/*, served by inference.router
# below (chords, download, generate, generate-variants, generate-progression)
# and job_routes.router (the async transcription endpoints: POST /transcribe,
# GET /transcribe/{job_id} -- inference.router used to also serve a
# synchronous POST /transcribe; that route is retired, so job_routes.router
# is now the sole owner of that path). /api/upload, /model-info, /process/,
# and /health are the only routes registered directly on `app`.
app.include_router(inference.router, prefix="/api")
app.include_router(job_routes.router, prefix="/api")


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
