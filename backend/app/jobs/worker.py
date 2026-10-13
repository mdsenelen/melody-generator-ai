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
# its worker died and reclaims it. This is now a short "is the worker
# process still alive" window, not a budget sized to cover a whole job's
# compute time end to end: the background heartbeat thread started in
# process_one_job below renews it on a fixed clock for as long as the main
# thread is actually alive, so a legitimately slow job is never mistaken
# for a dead one no matter how long processing takes (see
# HEARTBEAT_INTERVAL_SECONDS).
#
# What this value actually bounds is the worst case for detecting a
# genuinely dead worker (e.g. an OOM kill) that dies the instant after
# claiming a job, before its heartbeat thread gets a chance to run even
# once. 120s (plus ~DEFAULT_POLL_TIMEOUT_SECONDS for the next idle tick to
# notice) keeps that worst case around two minutes. It used to be 900s,
# because the lease was only ever renewed from inside the main processing
# thread, at transcription-chunk boundaries (inference.py's on_progress
# callback, called only *after* each ~30s chunk finishes) -- a worker that
# died before reaching the end of its first chunk (including a long wait
# on HEAVY_WORK_LOCK, WORKER_HEAVY_WORK_WAIT_SEC=180s, before any chunk
# work even starts) got no renewal at all and sat on the full 900s.
DEFAULT_LEASE_SECONDS = 120.0
DEFAULT_STUCK_CREATING_AFTER_SECONDS = 30.0
DEFAULT_POLL_TIMEOUT_SECONDS = 5.0
# How often the background heartbeat thread (started in process_one_job)
# renews the lease while a job is being processed. ~3x margin under
# DEFAULT_LEASE_SECONDS (120s / 40s) so a renewal delayed by GIL scheduling
# jitter or a slow DB round-trip still lands well before the lease would
# actually expire -- renews 3 times within one lease window even if two
# consecutive renewals are missed.
HEARTBEAT_INTERVAL_SECONDS = 40.0
# How long a job waits before being re-tried after inference.WarmingUpError
# (the boot warm-up marker was still fresh). Deliberately short and fixed,
# not exponential backoff like a real failure's -- this isn't a failure, so
# there's no reason to back off, only to give the warm-up a little more
# time before trying again. Termination is bounded by WARMUP_MAX_SECONDS
# itself (inference.py), not by anything here or by attempt_count, which
# this path never increments.
WARMUP_REQUEUE_DELAY_SECONDS = 10.0


def process_one_job(
    job_id: str,
    store: JobStore,
    storage: ObjectStorage,
    queue: JobQueue,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
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

    # Progress is recorded here, from the main thread, at each
    # transcription-chunk boundary (inference.py's on_progress callback);
    # the heartbeat thread below reads it on its own clock, independent of
    # chunk timing -- a plain dict behind a lock since the two threads
    # touch it on different schedules.
    progress_state = {"value": 0}
    progress_lock = threading.Lock()

    def _record_progress(progress: int) -> bool:
        with progress_lock:
            progress_state["value"] = max(0, min(99, int(progress)))
        return True

    stop_heartbeat = threading.Event()

    def _heartbeat_loop() -> None:
        # wait() returns True only when stop_heartbeat is set, so this
        # renews on a fixed interval and exits promptly once the main
        # thread is done, without an extra tick after the last real one.
        while not stop_heartbeat.wait(timeout=heartbeat_interval_seconds):
            with progress_lock:
                current_progress = progress_state["value"]
            ok = store.heartbeat(
                job.id,
                lease_token=lease_token,
                progress=current_progress,
                lease_seconds=lease_seconds,
            )
            if not ok:
                logger.warning(
                    "Job %s heartbeat rejected -- lease %s no longer current", job.id, lease_token,
                )
                return

    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    try:
        raw = storage.read_bytes(job.storage_key)
        result = inference.run_basic_pitch(raw, job.source_filename, on_progress=_record_progress)
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
    except inference.WarmingUpError:
        # Not a failure -- the boot warm-up marker was still fresh when
        # this attempt tried to run. Put the job back in 'queued' with a
        # short fixed delay, WITHOUT counting it as a failed attempt (see
        # store.requeue_after_warmup's own docstring for why that's a
        # separate method from mark_failed). Termination is bounded by
        # WARMUP_MAX_SECONDS itself, not by anything here.
        logger.info(
            "Job %s requeued: boot warm-up still in progress (retrying in %.0fs, "
            "attempt not counted)",
            job.id, WARMUP_REQUEUE_DELAY_SECONDS,
        )
        if not store.requeue_after_warmup(
            job.id, lease_token=lease_token, delay_seconds=WARMUP_REQUEUE_DELAY_SECONDS
        ):
            logger.warning(
                "Job %s warm-up requeue write discarded; lease %s no longer current",
                job.id, lease_token,
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
    finally:
        # Event.set() wakes the thread immediately (wait() doesn't block
        # out its remaining timeout), so this join is just a safety bound,
        # not a real wait in practice.
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=5.0)


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
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
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
                process_one_job(
                    job_id, store, storage, queue,
                    lease_seconds=lease_seconds,
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                )
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
