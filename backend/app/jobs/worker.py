from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import HTTPException

from .queue import JobQueue
from .storage import ObjectStorage
from .store import JobStore

logger = logging.getLogger(__name__)

# How long a claimed job's lease is valid before the worker loop assumes
# its worker died and reclaims it. No longer bounded by
# GENERATION_TIMEOUT_SECONDS -- transcription jobs aren't a synchronous HTTP
# request anymore, so what this actually needs margin above is
# inference.MAX_ANALYSIS_DURATION_SEC (240s as of this default): at
# "roughly realtime speed" Basic Pitch inference on a 240s clip plus decode/
# resample/chord-and-key analysis overhead could itself approach 240s, so
# the lease needs real headroom above that, not just above the old 60s
# request timeout. 900s (15 minutes) keeps a legitimately-slow job from
# ever being reclaimed as if its worker had died.
DEFAULT_LEASE_SECONDS = 900.0
DEFAULT_STUCK_CREATING_AFTER_SECONDS = 30.0
DEFAULT_POLL_TIMEOUT_SECONDS = 5.0


def process_one_job(
    job_id: str,
    store: JobStore,
    storage: ObjectStorage,
    queue: JobQueue,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
) -> None:
    """Claim and run a single job. A claim miss (already claimed/handled by
    another worker, or no longer queued) is not an error -- just a no-op."""
    from app import inference  # local import: avoids a hard import-time
    # dependency from this lightweight jobs package on inference.py's much
    # heavier one (torch, basic-pitch, librosa...), so job-store/queue unit
    # tests don't need those installed.

    job = store.claim_job(job_id, lease_seconds=lease_seconds)
    if job is None:
        logger.info("Job %s was not claimable (already handled?)", job_id)
        return
    lease_token = job.lease_token
    assert lease_token is not None

    logger.info("Processing transcription job %s (%s)", job.id, job.source_filename)
    try:
        raw = storage.read_bytes(job.storage_key)
        result = inference.run_basic_pitch(raw, job.source_filename)
        _upload_generated_files(result, storage)
        if not store.mark_completed(job.id, result, lease_token=lease_token):
            # Our lease was reclaimed (this attempt ran too long, another
            # worker already picked the job back up) before we finished --
            # a newer attempt owns the job now. Discard, don't overwrite.
            logger.warning(
                "Job %s completed but lease %s is no longer current; discarding result",
                job.id, lease_token,
            )
        else:
            logger.info("Transcription job %s completed", job.id)
    except HTTPException as exc:
        # 4xx from the pipeline (bad/undecodable audio, empty payload) is a
        # permanent failure -- retrying won't change the input. 5xx (model/
        # service temporarily unavailable) is the same category as an
        # unhandled exception below: worth retrying. Either way exc.detail
        # is already a short, curated, user-safe string (see inference.py),
        # never a raw exception message or traceback.
        retryable = exc.status_code >= 500
        logger.warning(
            "Transcription job %s failed (%s, retryable=%s): %s",
            job.id, exc.status_code, retryable, exc.detail,
        )
        _mark_failed_and_maybe_requeue(
            store, queue, job.id, str(exc.detail), retry=retryable, lease_token=lease_token
        )
    except Exception:
        # Unexpected/transient failure: retry up to the job's max_attempts,
        # and never leak internals (stack trace, exception text) to the
        # client -- only the sanitized message below.
        logger.exception("Transcription job %s failed unexpectedly", job.id)
        _mark_failed_and_maybe_requeue(
            store, queue, job.id, "Transcription failed unexpectedly",
            retry=True, lease_token=lease_token,
        )


def _mark_failed_and_maybe_requeue(
    store: JobStore, queue: JobQueue, job_id: str, error: str, *, retry: bool, lease_token: str
) -> None:
    if not store.mark_failed(job_id, error, retry=retry, lease_token=lease_token):
        logger.warning(
            "Job %s failure write discarded; lease %s no longer current", job_id, lease_token,
        )
        return
    # mark_failed with retry=True moves the job back to 'queued' with a
    # backoff delay (next_attempt_at), not immediately runnable -- the
    # idle-tick sweep in run_worker_loop is what actually re-enqueues it
    # once that delay elapses. Nothing needs to happen here beyond the
    # store write; this function exists mainly so the two call sites above
    # share the discarded-lease logging.


def _upload_generated_files(result: dict[str, Any], storage: ObjectStorage) -> None:
    """run_basic_pitch (via inference._save_bytes) always writes the
    generated MIDI/WAV to local disk, regardless of which process calls
    it -- fine for the synchronous /api/transcribe route (same process
    serves /api/download), not fine here, where the worker may be a
    separate Render service from the web process serving /api/download.
    Mirror both files into the same object storage used for job input, so
    /api/download can fall back to it. No-op (and no local-disk
    dependency introduced) when object storage isn't configured, i.e.
    LocalFilesystemStorage is in use -- local dev keeps today's
    single-process, local-disk-only behavior unchanged."""
    from app import inference  # see process_one_job for why this is local

    if not storage.is_shared:
        return

    for filename_key, bytes_key in (("midi_filename", "midi_bytes"), ("wav_filename", "wav_bytes")):
        filename = result.get(filename_key)
        if not filename:
            continue
        local_path = inference.OUTPUT_DIR / filename
        try:
            data = local_path.read_bytes()
        except OSError:
            logger.warning("Expected generated file %s missing on local disk", local_path)
            continue
        storage.write_bytes(f"generated/{filename}", data)


def run_worker_loop(
    *,
    store: JobStore,
    queue: JobQueue,
    storage: ObjectStorage,
    stop_event: threading.Event,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
    stuck_creating_after_seconds: float = DEFAULT_STUCK_CREATING_AFTER_SECONDS,
) -> None:
    """Runs until stop_event is set. Safe to run as a daemon thread (dev, or
    a single-service deployment) or as the entrypoint of a standalone
    process (production; see worker_main.py) -- it only depends on the
    store/queue/storage adapters passed in, not on how it's hosted.

    Bootstraps by re-enqueueing anything already 'queued' in the store: an
    InProcessJobQueue restart loses unclaimed queue messages (it's pure
    in-memory), so a job created just before a dev-server reload would
    otherwise wait forever with nothing to wake a worker for it.
    """
    for job_id in store.get_queued_job_ids():
        queue.enqueue(job_id)

    while not stop_event.is_set():
        try:
            job_id = queue.dequeue(timeout=poll_timeout)
        except Exception:
            # dequeue() itself isn't guarded by process_one_job's try/except
            # below -- a transient queue-backend hiccup (e.g. RedisJobQueue's
            # BRPOP hitting a socket read timeout) would otherwise propagate
            # straight out of this loop and silently kill the whole worker
            # thread (it's a daemon thread in-process, so nothing restarts
            # it). Treat it like an empty poll: log and fall through to the
            # idle tick, then retry on the next iteration instead of dying.
            logger.exception("Error dequeuing from job queue; will retry next poll")
            job_id = None
        if job_id:
            try:
                process_one_job(job_id, store, storage, queue, lease_seconds=lease_seconds)
            except Exception:
                # process_one_job already catches pipeline errors into
                # mark_failed; this is a last-resort guard so one broken
                # job can't kill the whole worker loop (e.g. the storage
                # backend itself is unreachable).
                logger.exception("Unhandled error processing job %s", job_id)
            continue

        # Idle tick: reconcile jobs whose worker died mid-processing,
        # re-enqueue retries whose backoff delay has elapsed, and resolve
        # jobs stuck in 'creating' (see service.create_transcription_job).
        for requeued_id in store.reclaim_stale_processing_jobs():
            logger.warning("Reclaimed stale job %s back to queued", requeued_id)
            queue.enqueue(requeued_id)

        for ready_id in store.get_ready_retry_job_ids():
            logger.info("Re-enqueueing retry %s after backoff", ready_id)
            queue.enqueue(ready_id)

        for released_id in store.reconcile_stuck_creating_jobs(
            stuck_creating_after_seconds, storage.exists
        ):
            logger.info("Released stuck-creating job %s", released_id)
            queue.enqueue(released_id)
