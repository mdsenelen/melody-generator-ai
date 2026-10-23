from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.rate_limit import rate_limiter
from app.schemas import AnalyzeRequest, RegisterTranscriptionRequest

from .models import COMPLETED, EXPIRED, FAILED, Job
from .service import create_completed_job, create_transcription_job, get_job_store

router = APIRouter()
logger = logging.getLogger(__name__)

# Both trigger real Basic Pitch inference cost with no auth in front of
# them -- see CLAUDE.md's Production Readiness/Security notes.
TRANSCRIBE_RATE_LIMIT = int(os.environ.get("TRANSCRIBE_RATE_LIMIT", "5"))
TRANSCRIBE_RATE_WINDOW_SECONDS = float(os.environ.get("TRANSCRIBE_RATE_WINDOW_SECONDS", "300"))

# POST /api/transcriptions gets its own bucket rather than sharing
# "transcribe": a browser attempt whose registration 422s falls back to
# POST /api/transcribe, and sharing would count that one user action twice
# (a surprise 429 on the fallback). Higher limit than "transcribe" because
# registration is cheap -- MIDI-from-notes, no Basic Pitch, no heavy lock.
REGISTER_TRANSCRIPTION_RATE_LIMIT = int(os.environ.get("REGISTER_TRANSCRIPTION_RATE_LIMIT", "10"))


@router.post(
    "/transcribe",
    dependencies=[Depends(rate_limiter("transcribe", TRANSCRIBE_RATE_LIMIT, TRANSCRIBE_RATE_WINDOW_SECONDS))],
)
async def create_transcribe_job(
    file: Optional[UploadFile] = File(None),
    upload_id: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
) -> JSONResponse:
    # Deliberately mirrors inference.py's _read_upload_bytes /
    # _resolve_upload_path pair (same validation, same "existing upload"
    # resolution) rather than duplicating that logic here.
    from app import inference

    if file is not None:
        raw = await inference._read_upload_bytes(file)
        source_filename = file.filename or "audio.wav"
    else:
        input_path = inference._resolve_upload_path(filename, upload_id)
        if input_path is None:
            raise HTTPException(
                status_code=404,
                detail="No file uploaded and no matching prior upload found",
            )
        raw = input_path.read_bytes()
        source_filename = input_path.name

    # File I/O (storage write) and DB access are blocking; keep them off
    # the event loop like every other route in this app (see
    # inference.py's _run_generation).
    job = await run_in_threadpool(create_transcription_job, raw, source_filename, upload_id)

    # Reuse Job.to_status_payload()'s status normalization (it maps the
    # internal-only 'creating' state to 'queued') rather than reading
    # job.status directly -- an idempotent reuse can return a job that's
    # still in 'creating' if the original request is still mid-flight.
    return JSONResponse(status_code=202, content={"job_id": job.id, "status": job.to_status_payload()["status"]})


@router.post(
    "/transcriptions",
    dependencies=[
        Depends(
            rate_limiter(
                "register_transcription",
                REGISTER_TRANSCRIPTION_RATE_LIMIT,
                TRANSCRIBE_RATE_WINDOW_SECONDS,
            )
        )
    ],
)
async def register_transcription(payload: RegisterTranscriptionRequest) -> dict[str, Any]:
    """Register note events the browser already transcribed (Basic Pitch via
    TF.js) as a completed job -- same result shape as a server transcription,
    so /api/analyze, GET /transcribe/{id} and generate-variants' jobId
    shortcut need no changes. Additive: the server path is untouched and
    remains the frontend's fallback."""
    from app import inference  # lazy import, see create_transcribe_job above

    note_events = [n.model_dump() for n in payload.note_events]
    result = await run_in_threadpool(
        inference.register_browser_transcription,
        note_events,
        payload.source_duration_sec,
        payload.tfjs_backend,
    )
    job = await run_in_threadpool(create_completed_job, payload.source_filename, result)
    logger.info(
        "Registered browser transcription job %s (n_notes=%d, duration_sec=%.2f, tfjs_backend=%s)",
        job.id, result["n_notes"], result["source_duration_sec"], payload.tfjs_backend,
    )
    return job.to_status_payload()


def _is_past_retention(job: Job, retention_hours: float) -> bool:
    """A terminal job's generated MIDI/WAV files live on local disk (or,
    once GP3's synchronous-generation jobs exist, may never have had a
    stored input at all) and are swept by inference.cleanup_stale_files
    after DATA_RETENTION_HOURS -- but the job row itself is never deleted.
    Without this check, an old job would keep reporting 'completed'/'failed'
    with download links that 404, instead of reading as gone."""
    if job.status not in (COMPLETED, FAILED):
        return False
    reference = job.completed_at or job.updated_at
    try:
        finished_at = datetime.fromisoformat(reference)
    except (TypeError, ValueError):
        return False
    if finished_at.tzinfo is None:
        finished_at = finished_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - finished_at > timedelta(hours=retention_hours)


@router.get("/transcribe/{job_id}")
async def get_transcribe_job(job_id: str) -> dict[str, Any]:
    store = get_job_store()
    job = await run_in_threadpool(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    from app import inference  # lazy import, see create_transcribe_job above

    if _is_past_retention(job, inference.DATA_RETENTION_HOURS):
        return {"job_id": job.id, "status": EXPIRED, "progress": job.progress, "result": None, "error": None}

    return job.to_status_payload()


@router.post("/analyze")
async def analyze_clip_route(payload: AnalyzeRequest) -> dict[str, Any]:
    """Mood / key / tempo / chords / pitch histogram for a clip of a completed
    transcribe job -- computed from its stored note events, re-runnable against
    any window without re-transcribing. No audio, no Basic Pitch, no
    HEAVY_WORK_LOCK (see docs/PLAN-full-transcription-split.md step 2)."""
    from app import inference  # lazy import, see create_transcribe_job above

    store = get_job_store()
    job = await run_in_threadpool(store.get_job, payload.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != COMPLETED or not job.result:
        raise HTTPException(
            status_code=409, detail="Transcription is not complete for this job yet"
        )

    note_events = job.result.get("note_events")
    if not note_events:
        raise HTTPException(
            status_code=409,
            detail="This transcription has no stored note data — re-transcribe to analyze a clip.",
        )

    start = max(0.0, float(payload.clip_start_sec))
    end = payload.clip_end_sec
    if end is not None:
        end = float(end)
        if end <= start:
            raise HTTPException(
                status_code=400, detail="clip_end_sec must be greater than clip_start_sec"
            )

    return await run_in_threadpool(inference.analyze_clip, note_events, start, end)
