from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from pathlib import Path
from typing import Optional

from .models import QUEUED, Job
from .queue import JobQueue, create_job_queue_from_env
from .storage import ObjectStorage, create_object_storage_from_env
from .store import JobStore, create_job_store_from_env

logger = logging.getLogger(__name__)

# Lazy module-level singletons, same pattern as inference.py's
# _CVAE_IDDM_BUNDLE / _BASIC_PITCH_MODEL: cheap to check, initialized once,
# swappable in tests via monkeypatch.
_JOB_STORE: Optional[JobStore] = None
_JOB_QUEUE: Optional[JobQueue] = None
_OBJECT_STORAGE: Optional[ObjectStorage] = None
_INIT_LOCK = threading.Lock()


def get_job_store() -> JobStore:
    global _JOB_STORE
    if _JOB_STORE is None:
        with _INIT_LOCK:
            if _JOB_STORE is None:
                _JOB_STORE = create_job_store_from_env()
    return _JOB_STORE


def get_job_queue() -> JobQueue:
    global _JOB_QUEUE
    if _JOB_QUEUE is None:
        with _INIT_LOCK:
            if _JOB_QUEUE is None:
                _JOB_QUEUE = create_job_queue_from_env()
    return _JOB_QUEUE


def get_object_storage() -> ObjectStorage:
    global _OBJECT_STORAGE
    if _OBJECT_STORAGE is None:
        with _INIT_LOCK:
            if _OBJECT_STORAGE is None:
                _OBJECT_STORAGE = create_object_storage_from_env()
    return _OBJECT_STORAGE


def create_transcription_job(
    raw: bytes,
    source_filename: str,
    upload_id: Optional[str] = None,
) -> Job:
    """Persist the input audio, record job metadata, and hand the job off
    to the queue. Runs on the web process; the worker (in-process thread in
    dev, or a separate process in production) does the actual transcription.

    Idempotency key defaults to the upload_id when the client referenced an
    existing upload (so retrying the same upload_id reuses the in-flight/
    completed job), or a content hash of the raw bytes otherwise (so an
    identical direct-upload retry -- e.g. after a network timeout where the
    client can't tell if the first attempt landed -- doesn't double up).

    Three separate steps have to succeed for a job to actually be
    runnable -- the DB row, the storage write, and the queue push -- and
    nothing here makes all three atomic. Instead the row starts in the
    internal 'creating' state (see models.CREATING) and is only flipped to
    'queued' after the storage write succeeds; if this process dies in
    between (or the enqueue call itself fails), the job is left in
    'creating' rather than a 'queued' row pointing at input bytes that may
    not exist. Worker.reconcile_stuck_creating_jobs, run periodically by
    every worker regardless of which process created the job, is what
    resolves that afterward: it checks whether the bytes actually made it
    to storage and either releases the job to 'queued' (re-enqueuing it)
    or fails it outright if they never did. This works whether the web
    service and worker are the same process or, in production, entirely
    separate ones -- the reconciliation reads the same shared store/storage
    the creator wrote to, nothing web-process-local is required."""
    store = get_job_store()
    idempotency_key = upload_id or hashlib.sha256(raw).hexdigest()
    ext = Path(source_filename).suffix.lower() or ".wav"
    job_id = uuid.uuid4().hex
    storage_key = f"transcribe-jobs/{job_id}/input{ext}"

    # create_job itself decides whether this reuses an existing job for the
    # idempotency key; only write bytes / enqueue for a genuinely new one,
    # so a client retry never duplicates the (potentially large) upload.
    job, created = store.create_job(
        job_id=job_id,
        source_filename=source_filename,
        storage_key=storage_key,
        upload_id=upload_id,
        idempotency_key=idempotency_key,
    )
    if not created:
        logger.info("Reusing existing transcription job %s for idempotency key", job.id)
        return job

    get_object_storage().write_bytes(storage_key, raw)
    # If this process dies between the write above and here, or the
    # enqueue call below fails, the row stays 'creating' -- recovered by
    # reconcile_stuck_creating_jobs rather than lost.
    store.mark_creation_ready(job.id)
    get_job_queue().enqueue(job.id)
    job.status = QUEUED
    return job


def create_completed_job(source_filename: str, result: dict[str, Any]) -> Job:
    """Persist an already-computed result -- from a synchronous generation
    endpoint (generate-variants, generate-progression) that has no timeout
    problem to fix and no async work left to do -- as a completed job, so it
    gets a shareable job_id backed by the same jobs table and result page as
    transcription jobs (see GP3 in docs/GUIDED-PASS.md). No queue or object
    storage involvement: this drives the store straight through its normal
    create -> ready -> claim -> complete sequence so the row ends up
    indistinguishable from one a real worker processed, without adding any
    new store method."""
    store = get_job_store()
    job, _ = store.create_job(
        job_id=uuid.uuid4().hex, source_filename=source_filename, storage_key=""
    )
    store.mark_creation_ready(job.id)
    claimed = store.claim_job(job.id, lease_seconds=60.0)
    assert claimed is not None and claimed.lease_token is not None
    store.mark_completed(job.id, result, lease_token=claimed.lease_token)
    completed = store.get_job(job.id)
    assert completed is not None
    return completed
