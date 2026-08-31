from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.rate_limit import rate_limiter

from .models import COMPLETED, EXPIRED, FAILED, Job
from .service import create_transcription_job, get_job_store

router = APIRouter()

# Both trigger real Basic Pitch inference cost with no auth in front of
# them -- see CLAUDE.md's Production Readiness/Security notes.
TRANSCRIBE_RATE_LIMIT = int(os.environ.get("TRANSCRIBE_RATE_LIMIT", "5"))
TRANSCRIBE_RATE_WINDOW_SECONDS = float(os.environ.get("TRANSCRIBE_RATE_WINDOW_SECONDS", "300"))


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
