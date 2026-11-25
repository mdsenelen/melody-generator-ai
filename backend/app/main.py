from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app import inference
from app.jobs import routes as job_routes
from app.jobs.service import get_job_queue, get_job_store, get_object_storage
from app.jobs.worker import run_worker_loop
from app.request_limits import MaxBodySizeMiddleware, RequestBodyTooLarge
from app.schemas import ProcessRequest

# Every app.* module just does logging.getLogger(__name__) with no handler
# of its own, and nothing here ever called logging.basicConfig/dictConfig --
# uvicorn configures its OWN loggers (uvicorn/uvicorn.access) but its config
# has no "root" entry, so it never touches the root logger either. Net
# result, confirmed directly against both a real local run and the live
# Render log stream: every logger.info() call in this codebase (22 of them
# at the time this was found) was silently dropped -- not delayed, not
# buffered, just never handled anywhere up to Python's logging.lastResort,
# which only fires at WARNING+. This must run before `app` and the FastAPI
# lifespan below are defined, so it's in place before uvicorn's own
# configure_logging() runs later at server start (which doesn't touch the
# root logger either, so this survives it rather than being overwritten).
#
# force=True is required, not optional: basic_pitch's own __init__.py calls
# logging.basicConfig() itself at import time (confirmed by testing, not
# guessed -- that's also why its "X is not installed" lines already print
# in the bare "WARNING:root:..." format), and basicConfig() is a documented
# no-op once the root logger already has a handler. Without force=True,
# this call above would silently do nothing whenever anything that pulls in
# basic_pitch gets imported first, which `from app import inference` above
# already does -- i.e. always, in this file.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

# A handful of third-party libraries emit their own INFO-level noise that
# would otherwise compete with our own log lines in Render's stream.
# Checked each of numba/librosa/resampy/soundfile/basic_pitch directly
# (static source scan + exercising the real decode/predict warm-up path
# with level=INFO) -- none of them actually emit at INFO in this app's real
# code paths (numba's few INFO call sites are CUDA/tracing/AOT-compile only,
# none of which this CPU-only deploy ever touches; TensorFlow isn't even
# installed). botocore is the one that is: it logs "Found credentials in
# environment variables" at INFO every time a boto3 client is constructed
# (jobs/storage.py's B2/S3 client, built from JOB_STORAGE_* env vars) --
# confirmed firing, and confirmed silenced by this.
logging.getLogger("botocore").setLevel(logging.WARNING)

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
# How often the in-process worker's watchdog checks that its thread is still
# alive. Only a thread.is_alive() call in the common case, so a short interval
# costs nothing and bounds how long a dead thread can sit there with new jobs
# queueing up behind it.
WORKER_WATCHDOG_INTERVAL_SECONDS = 30.0

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


class InProcessWorker:
    """Owns the in-process transcription worker thread, plus the liveness
    signal /health reports for it.

    run_worker_loop guards every per-job path, but nothing catches a failure
    in the loop itself (a MemoryError on the 512MB tier, or anything the
    store/queue adapters raise outside those guards). It runs as a daemon
    thread, so such a failure is completely silent: the process keeps
    serving, /health keeps saying ok, and every new job queues up behind a
    worker that no longer exists. Hence a supervisor that can restart the
    thread, and a heartbeat (the loop's own tick, via run_worker_loop's
    on_tick) so /health can distinguish "idle" from "not running".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_tick_at: Optional[float] = None
        self._restarts = 0

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        self._thread = threading.Thread(
            target=run_worker_loop,
            kwargs=dict(
                store=get_job_store(),
                queue=get_job_queue(),
                storage=get_object_storage(),
                stop_event=self._stop_event,
                on_tick=self._record_tick,
            ),
            name="transcription-worker",
            daemon=True,
        )
        self._thread.start()

    def _record_tick(self) -> None:
        self._last_tick_at = time.monotonic()

    def ensure_running(self) -> bool:
        """Restart the thread if it died without being asked to stop.
        Returns True when a restart actually happened."""
        with self._lock:
            if self._stop_event.is_set():
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            logger.warning(
                "In-process transcription worker thread is not running; restarting it"
            )
            self._restarts += 1
            self._start_locked()
            return True

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=DEFAULT_WORKER_SHUTDOWN_TIMEOUT_SECONDS)

    def status(self) -> dict[str, Any]:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            last_tick_at = self._last_tick_at
            restarts = self._restarts
        return {
            "mode": "in_process",
            "alive": alive,
            # None until the loop's first tick. Grows while a job is being
            # processed (the loop is inside process_one_job, not ticking), so
            # `alive` is the authority on whether the worker exists -- this
            # only says whether the loop itself has made progress recently.
            "last_tick_age_seconds": (
                None if last_tick_at is None else round(time.monotonic() - last_tick_at, 1)
            ),
            "restarts": restarts,
        }


_in_process_worker = InProcessWorker()


def _worker_status() -> dict[str, Any]:
    if not RUN_WORKER_IN_PROCESS:
        # The worker is a separate process (a Render Background Worker in
        # production), so this process has nothing to report on and must not
        # claim it's dead -- see RUN_WORKER_IN_PROCESS above.
        return {"mode": "external"}
    return _in_process_worker.status()


def _queue_status() -> dict[str, Any]:
    """Best-effort: /health is what the keep-warm ping and any uptime monitor
    hit, so a store hiccup must not turn "is the web process up" into a 500.
    Reported as reachable=false instead, with the age unknown."""
    try:
        age = get_job_store().oldest_queued_job_age_seconds()
    except Exception:
        logger.warning("Could not read the job queue for /health", exc_info=True)
        return {"reachable": False, "oldest_queued_job_age_seconds": None}
    return {
        "reachable": True,
        "oldest_queued_job_age_seconds": None if age is None else round(age, 1),
    }


async def _watch_in_process_worker() -> None:
    while True:
        await asyncio.sleep(WORKER_WATCHDOG_INTERVAL_SECONDS)
        _in_process_worker.ensure_running()


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

    # Model weights, unlike Basic Pitch, are needed by generation -- which
    # always runs in the web process regardless of RUN_WORKER_IN_PROCESS --
    # so this prefetch is unconditional. Moves the (tiny, ~1.4MB) download
    # off the first generation request's critical path; a failed/slow fetch
    # here just means that request pays the same retry it always has.
    weights_warm_up_task = asyncio.create_task(inference.warm_up_model_weights())

    worker_watchdog_task: Optional[asyncio.Task] = None
    if RUN_WORKER_IN_PROCESS:
        _in_process_worker.start()
        worker_watchdog_task = asyncio.create_task(_watch_in_process_worker())

    try:
        yield
    finally:
        cleanup_task.cancel()
        if warm_up_task is not None:
            warm_up_task.cancel()
        weights_warm_up_task.cancel()
        if worker_watchdog_task is not None:
            worker_watchdog_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task
        if warm_up_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await warm_up_task
        with contextlib.suppress(asyncio.CancelledError):
            await weights_warm_up_task
        if worker_watchdog_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await worker_watchdog_task
        if RUN_WORKER_IN_PROCESS:
            _in_process_worker.stop()


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


class UnhandledErrorMiddleware:
    """Turns any unhandled exception into a sanitized JSON 500.

    Not an @app.exception_handler(Exception): Starlette routes that handler
    through ServerErrorMiddleware, which always sits outside every user
    middleware -- so its 500 skipped CORSMiddleware and went out with no
    Access-Control-Allow-Origin. The browser (which uploads straight to this
    backend, cross-origin) then hid it behind an opaque "Failed to fetch",
    and the frontend could only say "Couldn't reach the server". Added before
    CORSMiddleware so CORS wraps it, same reasoning as the 413 above.
    """

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def tracking_send(message: Any) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            logger.exception("Unhandled error on %s %s", scope.get("method"), scope.get("path"))
            if response_started:
                raise
            response = JSONResponse(
                status_code=500,
                content={"detail": "An unexpected error occurred. Please try again."},
            )
            await response(scope, receive, send)


# Starlette's add_middleware inserts each new middleware ahead of the previous
# ones, so the last one added is outermost. The body-size cap is added first
# and CORS last: the cap's 413 for an oversized upload is sent directly from
# the middleware, and if CORS sat inside it that response would carry no
# Access-Control-Allow-Origin header -- the browser would hide it behind an
# opaque "Failed to fetch" instead of showing the real "too large" message.
app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_UPLOAD_BYTES)

app.add_middleware(UnhandledErrorMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestBodyTooLarge)
async def handle_request_body_too_large(request: Any, exc: RequestBodyTooLarge) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": f"Request body exceeds the {exc.max_bytes // (1024 * 1024)}MB upload limit"},
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
        "worker": _worker_status(),
        "queue": _queue_status(),
    }


UPLOAD_COPY_CHUNK_BYTES = 1024 * 1024


def _stream_upload_to_disk(source: BinaryIO, dest_path: Path) -> int:
    """Copy an upload to dest_path one chunk at a time; returns bytes written."""
    with dest_path.open("wb") as dest:
        shutil.copyfileobj(source, dest, UPLOAD_COPY_CHUNK_BYTES)
        return dest.tell()


@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(...),
    sample_rate: Optional[int] = None,
) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm")):
        raise HTTPException(
            status_code=400, detail="Supported formats: wav, mp3, flac, ogg, m4a, webm")

    unique_id = uuid.uuid4().hex
    suffix = Path(file.filename).suffix.lower() or ".wav"
    filename = f"upload_{unique_id}{suffix}"
    file_path = UPLOAD_DIR / filename

    # Stream the (already spooled) body to disk in chunks rather than
    # `await file.read()`: uploads aren't serialized by HEAVY_WORK_LOCK, so a
    # 50MB body materialized in RAM during a transcription, or on a container
    # left at the post-generation floor, could OOM-kill the 512MB instance
    # mid-request. Anything rejected below is deleted before raising.
    try:
        written = await run_in_threadpool(_stream_upload_to_disk, file.file, file_path)
        if not written:
            raise HTTPException(status_code=400, detail="Empty upload")

        # Best-effort duration cap before a job exists. The header probe only
        # reads wav/flac/ogg; mp3/m4a/webm return None and are hard-capped
        # later in the chunked transcribe path (MAX_UPLOAD_DURATION_SEC).
        probed = inference._probe_source_duration_sec(file_path)
        if probed is not None and probed > inference.MAX_UPLOAD_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"Audio is too long (max {inference.MAX_UPLOAD_DURATION_SEC / 60:.0f} minutes)",
            )
    except BaseException:
        file_path.unlink(missing_ok=True)
        raise

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
