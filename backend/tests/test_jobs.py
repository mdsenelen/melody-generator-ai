from __future__ import annotations

import asyncio
import io
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference  # noqa: E402
from app.jobs import routes as job_routes  # noqa: E402
from app.jobs import service  # noqa: E402
from app.jobs.models import COMPLETED, CREATING, FAILED, PROCESSING, QUEUED  # noqa: E402
from app.jobs.queue import InProcessJobQueue  # noqa: E402
from app.jobs.storage import LocalFilesystemStorage, ObjectStorage  # noqa: E402
from app.jobs.store import create_sqlite_job_store  # noqa: E402
from app.jobs.worker import process_one_job, run_worker_loop  # noqa: E402

LEASE_SECONDS = 60.0


@pytest.fixture
def store(tmp_path):
    return create_sqlite_job_store(str(tmp_path / "jobs.db"))


@pytest.fixture
def storage(tmp_path):
    return LocalFilesystemStorage(tmp_path / "storage")


@pytest.fixture
def queue():
    return InProcessJobQueue()


@pytest.fixture(autouse=True)
def reset_service_singletons(monkeypatch):
    monkeypatch.setattr(service, "_JOB_STORE", None)
    monkeypatch.setattr(service, "_JOB_QUEUE", None)
    monkeypatch.setattr(service, "_OBJECT_STORAGE", None)


def _fake_transcription_result(**overrides):
    result = {
        "n_notes": 3,
        "duration_sec": 1.0,
        "source_duration_sec": 1.0,
        "truncated": False,
        "midi_b64": "AAA=",
        "wav_b64": None,
        "midi_filename": "transcription.mid",
        "wav_filename": "",
        "mood_label": "happy",
        "mood_idx": 0,
        "detected_chords": ["C"],
        "key": "C major",
        "pitch_histogram": [0.1] * 12,
        "tempo_bpm": 120.0,
        "average_pitch": 60.0,
    }
    result.update(overrides)
    return result


def _create_queued_job(store, **kwargs):
    """create_job alone leaves a row in the internal 'creating' state (see
    models.CREATING) until mark_creation_ready runs -- most store/worker
    tests don't care about that distinction and just want a queued job."""
    job, created = store.create_job(**kwargs)
    assert created
    assert store.mark_creation_ready(job.id)
    return store.get_job(job.id)


# --- store: creation & idempotency -----------------------------------------


def test_create_job_starts_in_internal_creating_state(store):
    job, created = store.create_job(source_filename="clip.wav", storage_key="jobs/1/input.wav")
    assert created is True
    assert job.status == CREATING
    # Never exposed externally as anything but "queued".
    assert job.to_status_payload()["status"] == "queued"


def test_mark_creation_ready_transitions_to_queued(store):
    job, _ = store.create_job(source_filename="clip.wav", storage_key="jobs/1/input.wav")
    assert store.mark_creation_ready(job.id) is True
    assert store.get_job(job.id).status == QUEUED
    # Not re-transitionable once already queued.
    assert store.mark_creation_ready(job.id) is False


def test_create_job_is_idempotent_on_key(store):
    job1, created1 = store.create_job(
        source_filename="clip.wav", storage_key="jobs/1/input.wav", idempotency_key="dup"
    )
    job2, created2 = store.create_job(
        source_filename="clip.wav", storage_key="jobs/2/input.wav", idempotency_key="dup"
    )
    assert created1 is True
    assert created2 is False
    assert job1.id == job2.id


def test_create_job_allows_fresh_row_after_permanent_failure(store):
    job1 = _create_queued_job(
        store, source_filename="clip.wav", storage_key="jobs/1/input.wav", idempotency_key="dup"
    )
    claimed = store.claim_job(job1.id, lease_seconds=LEASE_SECONDS)
    store.mark_failed(job1.id, "bad audio", retry=False, lease_token=claimed.lease_token)

    job2, created2 = store.create_job(
        source_filename="clip.wav", storage_key="jobs/2/input.wav", idempotency_key="dup"
    )
    assert created2 is True
    assert job2.id != job1.id


def test_create_job_idempotency_is_atomic_across_concurrent_instances(tmp_path):
    """Simulates two separate web processes/instances sharing one database
    (two SQLJobStore instances, not one) racing to create a job for the
    same idempotency key at essentially the same instant. Only the
    database-level partial unique index + atomic upsert can make this
    correct -- a Python-level lock inside a single instance cannot, since
    each instance has its own."""
    db_path = str(tmp_path / "jobs.db")
    store_a = create_sqlite_job_store(db_path)
    store_b = create_sqlite_job_store(db_path)

    results: dict[str, tuple[str, bool]] = {}
    barrier = threading.Barrier(2)

    def run(store, label):
        barrier.wait()
        job, created = store.create_job(
            source_filename="clip.wav", storage_key=f"jobs/{label}/input.wav",
            idempotency_key="race-key",
        )
        results[label] = (job.id, created)

    t1 = threading.Thread(target=run, args=(store_a, "a"))
    t2 = threading.Thread(target=run, args=(store_b, "b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["a"][0] == results["b"][0]
    assert results["a"][1] != results["b"][1]  # exactly one of them actually created it


# --- store: claim / complete / fail -----------------------------------------


def test_claim_job_transitions_queued_to_processing_and_issues_lease(store):
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    claimed = store.claim_job(job.id, lease_seconds=LEASE_SECONDS)
    assert claimed is not None
    assert claimed.status == PROCESSING
    assert claimed.started_at is not None
    assert claimed.lease_token


def test_claim_job_only_lets_one_claimant_succeed(store):
    """Two workers racing for the same job_id: the conditional
    UPDATE ... WHERE status='queued' means only one claim can win."""
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    first = store.claim_job(job.id, lease_seconds=LEASE_SECONDS)
    second = store.claim_job(job.id, lease_seconds=LEASE_SECONDS)
    assert first is not None
    assert second is None
    assert first.lease_token != second  # sanity: distinct tokens would never collide anyway


def test_mark_completed_requires_matching_lease(store):
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    claimed = store.claim_job(job.id, lease_seconds=LEASE_SECONDS)

    ok = store.mark_completed(job.id, _fake_transcription_result(), lease_token=claimed.lease_token)
    assert ok is True

    refreshed = store.get_job(job.id)
    assert refreshed.status == COMPLETED
    assert refreshed.progress == 100
    assert refreshed.result["key"] == "C major"


def test_mark_completed_with_stale_lease_is_discarded(store):
    """The exact scenario the fencing token exists for: a worker's job was
    reclaimed as stale (its lease cleared, a new one issued to whoever
    picks it up next) while it was still running. When the original,
    slow-but-still-alive worker finally finishes and tries to complete the
    job with its now-stale lease token, the write must not take effect."""
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    # A negative lease is already expired the instant it's issued -- avoids
    # sleeping in the test to make reclaim_stale_processing_jobs's
    # lease_expires_at < now() check trigger.
    original = store.claim_job(job.id, lease_seconds=-1.0)

    reclaimed_ids = store.reclaim_stale_processing_jobs()
    assert reclaimed_ids == [job.id]

    new_worker = store.claim_job(job.id, lease_seconds=LEASE_SECONDS)
    assert new_worker is not None
    assert new_worker.lease_token != original.lease_token

    # The original (now-stale) worker finishes and tries to complete.
    ok = store.mark_completed(
        job.id, _fake_transcription_result(key="OLD RESULT"), lease_token=original.lease_token
    )
    assert ok is False

    # The newer attempt's eventual completion must be the one that sticks.
    ok2 = store.mark_completed(
        job.id, _fake_transcription_result(key="NEW RESULT"), lease_token=new_worker.lease_token
    )
    assert ok2 is True
    assert store.get_job(job.id).result["key"] == "NEW RESULT"


def test_mark_failed_without_retry_is_terminal(store):
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    claimed = store.claim_job(job.id, lease_seconds=LEASE_SECONDS)
    store.mark_failed(job.id, "bad audio", retry=False, lease_token=claimed.lease_token)

    refreshed = store.get_job(job.id)
    assert refreshed.status == FAILED
    assert refreshed.error == "bad audio"
    assert refreshed.attempt_count == 1


def test_mark_failed_with_retry_requeues_with_backoff_until_max_attempts(store):
    job = _create_queued_job(
        store, source_filename="clip.wav", storage_key="jobs/1/input.wav", max_attempts=2
    )

    claimed = store.claim_job(job.id, lease_seconds=LEASE_SECONDS)
    store.mark_failed(job.id, "transient", retry=True, lease_token=claimed.lease_token)
    after_first = store.get_job(job.id)
    assert after_first.status == QUEUED
    assert after_first.attempt_count == 1
    assert after_first.next_attempt_at is not None  # scheduled, not immediately retriable
    # Not yet claimable again -- it's not due until next_attempt_at.
    assert store.claim_job(job.id, lease_seconds=LEASE_SECONDS) is not None  # claim itself doesn't check backoff...

    # ...but get_queued_job_ids (used for the "no message in queue" recovery
    # path) deliberately excludes it, and get_ready_retry_job_ids is what
    # surfaces it once due.
    store.mark_failed(job.id, "transient", retry=True, lease_token=store.get_job(job.id).lease_token)
    after_second = store.get_job(job.id)
    assert after_second.status == FAILED
    assert after_second.attempt_count == 2


def test_get_ready_retry_job_ids_only_returns_due_jobs_and_clears_backoff(store):
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    claimed = store.claim_job(job.id, lease_seconds=LEASE_SECONDS)
    store.mark_failed(job.id, "transient", retry=True, lease_token=claimed.lease_token)

    assert store.get_job(job.id).status == QUEUED
    assert store.get_queued_job_ids() == []  # excluded: has a pending backoff

    # Backdate next_attempt_at into the past by re-running mark_failed's
    # backoff math isn't exposed directly; instead simulate elapsed time by
    # reclaiming with a zero threshold isn't applicable here (status is
    # queued, not processing) -- so directly poke next_attempt_at via a
    # second claim+fail cycle is unnecessary; use the store's own
    # dev-only escape hatch: reach into the connection to backdate it,
    # mirroring how a real clock would eventually satisfy the condition.
    conn = store._connect()
    conn.execute(
        "UPDATE transcription_jobs SET next_attempt_at = '2000-01-01T00:00:00+00:00' WHERE id = ?",
        (job.id,),
    )
    conn.commit()
    conn.close()

    ready = store.get_ready_retry_job_ids()
    assert ready == [job.id]
    assert store.get_job(job.id).next_attempt_at is None
    # A second call shouldn't return it again before it's claimed.
    assert store.get_ready_retry_job_ids() == []


def test_reclaim_stale_processing_jobs_requeues_and_reports_ids(store):
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    # A negative lease is already expired the instant it's issued.
    store.claim_job(job.id, lease_seconds=-1.0)

    requeued = store.reclaim_stale_processing_jobs()

    assert requeued == [job.id]
    refreshed = store.get_job(job.id)
    assert refreshed.status == QUEUED
    assert refreshed.lease_token is None


def test_reclaim_stale_processing_jobs_ignores_fresh_jobs(store):
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    store.claim_job(job.id, lease_seconds=LEASE_SECONDS)  # not yet expired

    requeued = store.reclaim_stale_processing_jobs()

    assert requeued == []
    assert store.get_job(job.id).status == PROCESSING


def test_get_queued_job_ids(store):
    job1 = _create_queued_job(store, source_filename="a.wav", storage_key="jobs/1/input.wav")
    job2 = _create_queued_job(store, source_filename="b.wav", storage_key="jobs/2/input.wav")
    store.claim_job(job2.id, lease_seconds=LEASE_SECONDS)  # no longer queued

    assert store.get_queued_job_ids() == [job1.id]


# --- store: stuck-creating reconciliation -----------------------------------


def test_reconcile_stuck_creating_jobs_releases_when_bytes_exist(store):
    job, _ = store.create_job(source_filename="clip.wav", storage_key="jobs/1/input.wav")
    # Simulate: storage write succeeded, but the process died before
    # mark_creation_ready/enqueue ran.
    released = store.reconcile_stuck_creating_jobs(stale_after_seconds=0, exists=lambda key: True)
    assert released == [job.id]
    assert store.get_job(job.id).status == QUEUED


def test_reconcile_stuck_creating_jobs_fails_when_bytes_missing(store):
    job, _ = store.create_job(source_filename="clip.wav", storage_key="jobs/1/input.wav")
    # Simulate: process died before the storage write ever happened.
    released = store.reconcile_stuck_creating_jobs(stale_after_seconds=0, exists=lambda key: False)
    assert released == []
    refreshed = store.get_job(job.id)
    assert refreshed.status == FAILED
    assert "never stored" in refreshed.error


def test_reconcile_stuck_creating_jobs_ignores_fresh_rows(store):
    job, _ = store.create_job(source_filename="clip.wav", storage_key="jobs/1/input.wav")
    released = store.reconcile_stuck_creating_jobs(stale_after_seconds=30, exists=lambda key: True)
    assert released == []
    assert store.get_job(job.id).status == CREATING


# --- worker: process_one_job (direct) ---------------------------------------


def test_process_one_job_completes_successfully(store, storage, queue, monkeypatch):
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    storage.write_bytes(job.storage_key, b"fake-audio-bytes")
    monkeypatch.setattr(inference, "run_basic_pitch", lambda raw, name: _fake_transcription_result())

    process_one_job(job.id, store, storage, queue)

    refreshed = store.get_job(job.id)
    assert refreshed.status == COMPLETED
    assert refreshed.result["tempo_bpm"] == 120.0


def test_process_one_job_marks_400_as_permanent_failure(store, storage, queue, monkeypatch):
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    storage.write_bytes(job.storage_key, b"garbage")

    def _raise(raw, name):
        raise HTTPException(status_code=400, detail="Could not decode audio")

    monkeypatch.setattr(inference, "run_basic_pitch", _raise)

    process_one_job(job.id, store, storage, queue)

    refreshed = store.get_job(job.id)
    assert refreshed.status == FAILED
    assert refreshed.error == "Could not decode audio"
    assert refreshed.attempt_count == 1


def test_process_one_job_retries_503_as_transient(store, storage, queue, monkeypatch):
    job = _create_queued_job(
        store, source_filename="clip.wav", storage_key="jobs/1/input.wav", max_attempts=2
    )
    storage.write_bytes(job.storage_key, b"fake-audio-bytes")

    def _raise(raw, name):
        raise HTTPException(status_code=503, detail="Model temporarily unavailable")

    monkeypatch.setattr(inference, "run_basic_pitch", _raise)

    process_one_job(job.id, store, storage, queue)

    after_first = store.get_job(job.id)
    assert after_first.status == QUEUED  # retried, not permanently failed
    assert after_first.attempt_count == 1


def test_process_one_job_retries_unexpected_errors_then_gives_up(store, storage, queue, monkeypatch):
    job = _create_queued_job(
        store, source_filename="clip.wav", storage_key="jobs/1/input.wav", max_attempts=2
    )
    storage.write_bytes(job.storage_key, b"fake-audio-bytes")

    def _raise(raw, name):
        raise RuntimeError("boom")

    monkeypatch.setattr(inference, "run_basic_pitch", _raise)

    process_one_job(job.id, store, storage, queue)
    after_first = store.get_job(job.id)
    assert after_first.status == QUEUED
    assert after_first.attempt_count == 1
    assert after_first.error == "Transcription failed unexpectedly"  # no traceback leaked

    # Directly re-claim (bypassing backoff/queue timing, which the
    # run_worker_loop-level test below covers) to drive the second attempt.
    process_one_job(job.id, store, storage, queue)
    after_second = store.get_job(job.id)
    assert after_second.status == FAILED
    assert after_second.attempt_count == 2


def test_process_one_job_is_a_noop_for_already_claimed_job(store, storage, queue, monkeypatch):
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    storage.write_bytes(job.storage_key, b"fake-audio-bytes")
    store.claim_job(job.id, lease_seconds=LEASE_SECONDS)  # simulate another worker already claiming it

    calls = []
    monkeypatch.setattr(
        inference, "run_basic_pitch", lambda raw, name: calls.append(1) or _fake_transcription_result()
    )

    process_one_job(job.id, store, storage, queue)

    assert calls == []
    assert store.get_job(job.id).status == PROCESSING


# --- worker: run_worker_loop (the real retry-delivery path) -----------------


def _run_loop_until(stop_event, predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_run_worker_loop_automatically_retries_a_transient_failure(store, storage, queue, monkeypatch):
    """This is the test that would have caught the original bug: a
    transient failure moves a job back to 'queued' in the store
    (mark_failed(retry=True)), but nothing re-enqueues it unless the
    worker LOOP itself (not a direct process_one_job call) is driving
    delivery. Uses a real run_worker_loop on a background thread with a
    real InProcessJobQueue -- no manual re-invocation of process_one_job."""
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    storage.write_bytes(job.storage_key, b"fake-audio-bytes")
    queue.enqueue(job.id)

    attempts = []

    def flaky_run_basic_pitch(raw, name):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("transient blip")
        return _fake_transcription_result()

    monkeypatch.setattr(inference, "run_basic_pitch", flaky_run_basic_pitch)

    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_worker_loop,
        kwargs=dict(
            store=store, queue=queue, storage=storage, stop_event=stop_event,
            poll_timeout=0.2,
        ),
        daemon=True,
    )
    thread.start()
    try:
        completed = _run_loop_until(
            stop_event, lambda: (store.get_job(job.id).status == COMPLETED), timeout=10.0
        )
        assert completed, f"job never completed automatically; last state: {store.get_job(job.id)}"
        assert len(attempts) == 2
    finally:
        stop_event.set()
        thread.join(timeout=5.0)


def test_run_worker_loop_survives_a_dequeue_error(store, storage, queue, monkeypatch):
    """Regression test for a production bug: RedisJobQueue.dequeue() can
    raise (e.g. a BRPOP socket read timeout) rather than returning None on
    an empty poll, and run_worker_loop's dequeue call wasn't wrapped in a
    try/except -- unlike process_one_job's call a few lines below it. Since
    the loop runs as a daemon thread with nothing supervising it, one
    dequeue() exception silently killed the whole worker permanently: jobs
    kept getting created and enqueued, but nothing ever claimed them again.
    Verified against production: a real job sat in 'queued' indefinitely
    after the in-process worker thread's first BRPOP raised
    redis.exceptions.TimeoutError."""
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    storage.write_bytes(job.storage_key, b"fake-audio-bytes")

    real_dequeue = queue.dequeue
    calls = []

    def flaky_dequeue(timeout):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("simulated transient queue backend error")
        return real_dequeue(timeout)

    monkeypatch.setattr(queue, "dequeue", flaky_dequeue)
    monkeypatch.setattr(inference, "run_basic_pitch", lambda raw, name: _fake_transcription_result())

    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_worker_loop,
        kwargs=dict(
            store=store, queue=queue, storage=storage, stop_event=stop_event,
            poll_timeout=0.2,
        ),
        daemon=True,
    )
    thread.start()
    try:
        # The job is only enqueued after the loop's first (failing) dequeue
        # attempt, so it can't be picked up by that doomed first call --
        # this proves the loop is still alive and polling afterward, not
        # just that a pre-enqueued job survived one bad poll.
        assert _run_loop_until(stop_event, lambda: bool(calls), timeout=5.0), \
            "loop never called dequeue"
        queue.enqueue(job.id)
        completed = _run_loop_until(
            stop_event, lambda: (store.get_job(job.id).status == COMPLETED), timeout=10.0
        )
        assert completed, f"job never completed; loop likely died on the dequeue error"
        assert len(calls) >= 2
    finally:
        stop_event.set()
        thread.join(timeout=5.0)


def test_run_worker_loop_reclaims_stale_job_and_old_attempt_cannot_overwrite(
    store, storage, queue, monkeypatch
):
    """End-to-end version of the lease-fencing scenario: a job whose lease
    has already expired sits in 'processing' with nothing in the queue
    (simulating a worker that silently died). The loop's idle-tick
    reclaims it and re-enqueues it; a *second*, independent worker loop
    picks it up and completes it; the "original" attempt then tries to
    complete with its now-stale lease and must be discarded."""
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    storage.write_bytes(job.storage_key, b"fake-audio-bytes")
    # A negative lease is already expired -- the idle-tick sweep reclaims
    # it on its very first pass, no need to wait out a real lease duration.
    original_claim = store.claim_job(job.id, lease_seconds=-1.0)
    assert original_claim is not None
    # No queue message for this job -- it's "stuck" in processing, exactly
    # as if its worker had crashed after claiming but before finishing.

    monkeypatch.setattr(inference, "run_basic_pitch", lambda raw, name: _fake_transcription_result())

    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_worker_loop,
        kwargs=dict(
            store=store, queue=queue, storage=storage, stop_event=stop_event,
            poll_timeout=0.2,
        ),
        daemon=True,
    )
    thread.start()
    try:
        completed = _run_loop_until(
            stop_event, lambda: (store.get_job(job.id).status == COMPLETED), timeout=10.0
        )
        assert completed
        winning_result = store.get_job(job.id).result
    finally:
        stop_event.set()
        thread.join(timeout=5.0)

    # The "original" worker finally finishes and tries to complete with
    # its stale lease -- must not overwrite the real result.
    ok = store.mark_completed(
        job.id, _fake_transcription_result(key="STALE OVERWRITE ATTEMPT"),
        lease_token=original_claim.lease_token,
    )
    assert ok is False
    assert store.get_job(job.id).result == winning_result
    assert store.get_job(job.id).result["key"] != "STALE OVERWRITE ATTEMPT"


# --- service.create_transcription_job ------------------------------------


def test_create_transcription_job_enqueues_and_persists_bytes(store, storage, queue, monkeypatch):
    monkeypatch.setattr(service, "_JOB_STORE", store)
    monkeypatch.setattr(service, "_JOB_QUEUE", queue)
    monkeypatch.setattr(service, "_OBJECT_STORAGE", storage)

    job = service.create_transcription_job(b"audio-bytes", "song.mp3", upload_id=None)

    assert job.status == QUEUED
    assert storage.read_bytes(job.storage_key) == b"audio-bytes"
    assert store.get_job(job.id).status == QUEUED
    assert queue.dequeue(timeout=0.1) == job.id


def test_create_transcription_job_is_idempotent_for_same_upload_id(store, storage, queue, monkeypatch):
    monkeypatch.setattr(service, "_JOB_STORE", store)
    monkeypatch.setattr(service, "_JOB_QUEUE", queue)
    monkeypatch.setattr(service, "_OBJECT_STORAGE", storage)

    job1 = service.create_transcription_job(b"audio-bytes", "song.mp3", upload_id="up-123")
    queue.dequeue(timeout=0.1)  # drain the first enqueue
    job2 = service.create_transcription_job(b"audio-bytes", "song.mp3", upload_id="up-123")

    assert job1.id == job2.id
    # A retried, deduped creation must not enqueue a second, redundant job.
    assert queue.dequeue(timeout=0.1) is None


def test_create_transcription_job_allows_retry_after_permanent_failure(store, storage, queue, monkeypatch):
    monkeypatch.setattr(service, "_JOB_STORE", store)
    monkeypatch.setattr(service, "_JOB_QUEUE", queue)
    monkeypatch.setattr(service, "_OBJECT_STORAGE", storage)

    job1 = service.create_transcription_job(b"audio-bytes", "song.mp3", upload_id="up-456")
    claimed = store.claim_job(job1.id, lease_seconds=LEASE_SECONDS)
    store.mark_failed(job1.id, "Could not decode audio", retry=False, lease_token=claimed.lease_token)

    job2 = service.create_transcription_job(b"audio-bytes", "song.mp3", upload_id="up-456")

    assert job2.id != job1.id
    assert job2.status == QUEUED


def test_reconciliation_recovers_a_creation_that_crashed_after_the_storage_write(
    store, storage, queue, monkeypatch
):
    """Simulates service.create_transcription_job dying between
    storage.write_bytes and mark_creation_ready/enqueue -- run_worker_loop's
    idle tick must find and release it without any web-process involvement,
    which is what makes this work even when the web service and worker are
    separate Render processes."""
    job, created = store.create_job(source_filename="clip.wav", storage_key="jobs/1/input.wav")
    assert created
    storage.write_bytes(job.storage_key, b"fake-audio-bytes")
    # Deliberately do NOT call mark_creation_ready or enqueue -- simulating
    # a crash right here.
    assert store.get_job(job.id).status == CREATING

    monkeypatch.setattr(inference, "run_basic_pitch", lambda raw, name: _fake_transcription_result())

    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_worker_loop,
        kwargs=dict(
            store=store, queue=queue, storage=storage, stop_event=stop_event,
            poll_timeout=0.2, stuck_creating_after_seconds=0,
        ),
        daemon=True,
    )
    thread.start()
    try:
        completed = _run_loop_until(
            stop_event, lambda: (store.get_job(job.id).status == COMPLETED), timeout=10.0
        )
        assert completed
    finally:
        stop_event.set()
        thread.join(timeout=5.0)


def test_reconciliation_permanently_fails_a_creation_that_crashed_before_the_storage_write(
    store, storage, queue,
):
    job, created = store.create_job(source_filename="clip.wav", storage_key="jobs/1/input.wav")
    assert created
    # No storage.write_bytes call at all -- the crash happened before that.

    released = store.reconcile_stuck_creating_jobs(stale_after_seconds=0, exists=storage.exists)

    assert released == []
    refreshed = store.get_job(job.id)
    assert refreshed.status == FAILED
    assert "never stored" in refreshed.error


# --- shared generated-file storage ------------------------------------------


class _RecordingSharedStorage:
    """Minimal in-memory ObjectStorage stand-in with is_shared=True, so
    worker._upload_generated_files' mirroring path actually runs."""

    is_shared = True

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def write_bytes(self, key, data):
        self.objects[key] = data

    def read_bytes(self, key):
        return self.objects[key]

    def exists(self, key):
        return key in self.objects


def test_worker_mirrors_generated_files_into_shared_storage_for_download_fallback(
    store, tmp_path, queue, monkeypatch
):
    """Proves the actual cross-service scenario: a file the worker writes
    (via inference._save_bytes, always local disk) becomes fetchable by
    download_generated_file even when it's NOT on that process's local
    OUTPUT_DIR -- by going through the shared object storage the worker
    mirrors into and the download route falls back to."""
    input_storage = LocalFilesystemStorage(tmp_path / "job-input")
    shared = _RecordingSharedStorage()

    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")
    input_storage.write_bytes(job.storage_key, b"fake-audio-bytes")

    def fake_run_basic_pitch(raw, name):
        # Mirrors what inference.run_basic_pitch really does: writes to
        # local OUTPUT_DIR and returns the bare filename.
        midi_path = inference.OUTPUT_DIR / "worker_output.mid"
        midi_path.write_bytes(b"MThd-fake-midi-bytes")
        return _fake_transcription_result(midi_filename="worker_output.mid", wav_filename="")

    monkeypatch.setattr(inference, "run_basic_pitch", fake_run_basic_pitch)

    try:
        process_one_job(job.id, store, input_storage, queue)
        # process_one_job's _upload_generated_files call used `input_storage`
        # above (is_shared=False) so nothing was mirrored there by design;
        # re-run the mirroring step directly against a shared backend to
        # prove the mechanism, then verify the web-service-side fallback.
        from app.jobs.worker import _upload_generated_files

        result = store.get_job(job.id).result
        _upload_generated_files(result, shared)

        assert shared.exists("generated/worker_output.mid")

        # Simulate the file NOT existing on this (web) process's local
        # disk -- e.g. a different Render service than the worker.
        local_copy = inference.OUTPUT_DIR / "worker_output.mid"
        local_copy.unlink()

        monkeypatch.setattr("app.jobs.service.get_object_storage", lambda: shared)
        response = inference.download_generated_file("worker_output.mid")
        assert response.status_code == 200
        assert response.body == b"MThd-fake-midi-bytes"
    finally:
        for name in ("worker_output.mid",):
            path = inference.OUTPUT_DIR / name
            if path.exists():
                path.unlink()


def test_download_generated_file_404s_when_missing_everywhere(monkeypatch):
    monkeypatch.setattr("app.jobs.service.get_object_storage", lambda: _RecordingSharedStorage())
    with pytest.raises(HTTPException) as exc:
        inference.download_generated_file("does-not-exist.mid")
    assert exc.value.status_code == 404


# --- routes ----------------------------------------------------------------


def test_create_transcribe_job_route_returns_202(store, storage, queue, monkeypatch):
    monkeypatch.setattr(job_routes, "get_job_store", lambda: store)
    monkeypatch.setattr(
        job_routes,
        "create_transcription_job",
        lambda raw, filename, upload_id=None: service.create_transcription_job(
            raw, filename, upload_id
        ),
    )
    monkeypatch.setattr(service, "_JOB_STORE", store)
    monkeypatch.setattr(service, "_JOB_QUEUE", queue)
    monkeypatch.setattr(service, "_OBJECT_STORAGE", storage)

    upload = UploadFile(filename="clip.wav", file=io.BytesIO(b"audio-bytes"))
    response = asyncio.run(
        job_routes.create_transcribe_job(file=upload, upload_id=None, filename=None)
    )

    assert response.status_code == 202
    import json as _json

    body = _json.loads(response.body)
    assert body["status"] == "queued"
    assert store.get_job(body["job_id"]) is not None


def test_get_transcribe_job_route_returns_status_shape(store, monkeypatch):
    monkeypatch.setattr(job_routes, "get_job_store", lambda: store)
    job = _create_queued_job(store, source_filename="clip.wav", storage_key="jobs/1/input.wav")

    payload = asyncio.run(job_routes.get_transcribe_job(job.id))

    assert payload == {
        "job_id": job.id,
        "status": "queued",
        "progress": 0,
        "result": None,
        "error": None,
    }


def test_get_transcribe_job_route_404_for_unknown_job(store, monkeypatch):
    monkeypatch.setattr(job_routes, "get_job_store", lambda: store)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(job_routes.get_transcribe_job("does-not-exist"))

    assert exc.value.status_code == 404


def test_create_transcribe_job_route_404_for_unknown_upload_reference(store, monkeypatch):
    monkeypatch.setattr(job_routes, "get_job_store", lambda: store)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            job_routes.create_transcribe_job(file=None, upload_id="missing", filename=None)
        )

    assert exc.value.status_code == 404


# --- main.py: RUN_WORKER_IN_PROCESS gating warm-up --------------------------


def test_warm_up_is_skipped_when_run_worker_in_process_is_false(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_WORKER_IN_PROCESS", "false")
    # Defense in depth: if anything here ever did fall through to the real
    # env-based store/storage factories despite RUN_WORKER_IN_PROCESS=false
    # skipping that code path entirely, it should land in an isolated
    # tmp_path rather than the real backend/data/ directory.
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.db"))
    import importlib

    from app import main as main_module

    importlib.reload(main_module)
    try:
        assert main_module.RUN_WORKER_IN_PROCESS is False

        warm_up_calls = []
        monkeypatch.setattr(
            main_module.inference, "warm_up_basic_pitch",
            lambda: warm_up_calls.append(1),
        )

        async def _drive_lifespan():
            async with main_module.lifespan(main_module.app):
                pass

        asyncio.run(_drive_lifespan())
        assert warm_up_calls == []
    finally:
        monkeypatch.delenv("RUN_WORKER_IN_PROCESS", raising=False)
        importlib.reload(main_module)


def test_warm_up_runs_when_run_worker_in_process_is_true(store, storage, queue, tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_WORKER_IN_PROCESS", "true")
    # Defense in depth: get_job_store/queue/storage are monkeypatched onto
    # main_module below, but if the real env-based factories were ever hit
    # by some other path, this keeps them pointed at tmp_path instead of
    # the real backend/data/ directory.
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.db"))
    import importlib

    from app import main as main_module

    importlib.reload(main_module)
    try:
        assert main_module.RUN_WORKER_IN_PROCESS is True

        warm_up_calls = []

        async def fake_warm_up():
            warm_up_calls.append(1)

        monkeypatch.setattr(main_module.inference, "warm_up_basic_pitch", fake_warm_up)
        # Avoid touching real infra (default SQLite path, a real worker
        # thread) -- this test is only about the warm-up gating, which the
        # run_worker_loop tests elsewhere already cover independently.
        monkeypatch.setattr(main_module, "get_job_store", lambda: store)
        monkeypatch.setattr(main_module, "get_job_queue", lambda: queue)
        monkeypatch.setattr(main_module, "get_object_storage", lambda: storage)
        monkeypatch.setattr(
            main_module, "run_worker_loop", lambda **kwargs: kwargs["stop_event"].wait()
        )

        async def _drive_lifespan():
            async with main_module.lifespan(main_module.app):
                # Yield control back to the event loop so the warm-up task
                # (scheduled via asyncio.create_task, not awaited directly)
                # actually gets to run before the context manager's
                # `finally` cancels it.
                await asyncio.sleep(0.05)

        asyncio.run(_drive_lifespan())
        assert warm_up_calls == [1]
    finally:
        monkeypatch.delenv("RUN_WORKER_IN_PROCESS", raising=False)
        importlib.reload(main_module)


# --- legacy synchronous endpoint is retired ---------------------------------


def test_legacy_synchronous_transcribe_route_is_removed():
    """The old POST /transcribe (inference.py's transcribe_audio, which ran
    Basic Pitch synchronously inside the request -- the exact thread-outlives-
    the-timeout problem the async job workflow exists to fix) must not be
    registered on inference.router anymore. If it ever came back, it would
    also collide with job_routes.router's POST /transcribe, since both
    mount at the same /api prefix in main.py."""
    paths = {
        (getattr(route, "path", None), tuple(sorted(getattr(route, "methods", []) or [])))
        for route in inference.router.routes
    }
    assert ("/transcribe", ("POST",)) not in paths


def test_async_transcribe_routes_own_the_transcribe_path():
    job_paths = {getattr(route, "path", None) for route in job_routes.router.routes}
    assert "/transcribe" in job_paths
    assert "/transcribe/{job_id}" in job_paths
