"""InProcessWorker (the supervisor behind the in-process transcription thread)
and the worker/queue fields /health reports.

Nothing here starts the app lifespan (no TestClient context manager), so no
startup cleanup runs against the real backend/data directories.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import main  # noqa: E402
from app.jobs.store import create_sqlite_job_store  # noqa: E402


@pytest.fixture
def fake_loop(monkeypatch):
    """Replace run_worker_loop and the adapters it is handed. Each call to the
    loop takes the next behaviour in `script`: "die" returns at once (a thread
    that ended on its own), "run" ticks once and then blocks until stop_event
    (a healthy worker). The last entry repeats."""
    calls: list[int] = []
    script = ["run"]

    def run_worker_loop(*, stop_event, on_tick, **_unused):
        calls.append(1)
        behaviour = script[min(len(calls) - 1, len(script) - 1)]
        if behaviour == "die":
            return
        on_tick()
        stop_event.wait(30)

    monkeypatch.setattr(main, "run_worker_loop", run_worker_loop)
    monkeypatch.setattr(main, "get_job_store", lambda: object())
    monkeypatch.setattr(main, "get_job_queue", lambda: object())
    monkeypatch.setattr(main, "get_object_storage", lambda: object())

    class Handle:
        def set(self, *behaviours):
            script[:] = behaviours

        @property
        def calls(self):
            return len(calls)

    return Handle()


@pytest.fixture
def worker():
    w = main.InProcessWorker()
    yield w
    w.stop()


def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_ensure_running_restarts_a_thread_that_died_and_counts_it(fake_loop, worker):
    fake_loop.set("die", "run")
    worker.start()
    assert _wait_until(lambda: not worker.status()["alive"]), "the first thread should have ended"
    assert worker.status()["restarts"] == 0

    assert worker.ensure_running() is True

    assert _wait_until(lambda: worker.status()["alive"])
    assert worker.status()["restarts"] == 1
    assert fake_loop.calls == 2


def test_ensure_running_leaves_a_healthy_thread_alone(fake_loop, worker):
    fake_loop.set("run")
    worker.start()
    assert _wait_until(lambda: worker.status()["alive"])

    assert worker.ensure_running() is False

    assert worker.status()["restarts"] == 0
    assert fake_loop.calls == 1


def test_watchdog_task_restarts_a_dead_worker(fake_loop, worker, monkeypatch):
    """The real asyncio watchdog, not just ensure_running(): it must notice
    the dead thread on its own interval."""
    fake_loop.set("die", "run")
    monkeypatch.setattr(main, "WORKER_WATCHDOG_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(main, "_in_process_worker", worker)
    worker.start()
    assert _wait_until(lambda: not worker.status()["alive"])

    async def scenario():
        task = asyncio.create_task(main._watch_in_process_worker())
        try:
            for _ in range(200):
                if worker.status()["restarts"] >= 1:
                    return
                await asyncio.sleep(0.02)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(scenario())

    assert worker.status()["restarts"] == 1
    assert _wait_until(lambda: worker.status()["alive"])


def test_a_stopped_worker_stays_stopped(fake_loop, worker):
    fake_loop.set("run")
    worker.start()
    assert _wait_until(lambda: worker.status()["alive"])

    worker.stop()

    assert worker.status()["alive"] is False
    assert worker.ensure_running() is False
    assert worker.status()["restarts"] == 0
    assert fake_loop.calls == 1


def test_a_worker_that_died_after_stop_is_not_resurrected(fake_loop, worker):
    fake_loop.set("die")
    worker.start()
    assert _wait_until(lambda: not worker.status()["alive"])
    worker.stop()

    assert worker.ensure_running() is False
    assert worker.status()["restarts"] == 0


def test_status_has_no_tick_before_the_loop_runs_and_a_fresh_one_after(fake_loop, worker):
    assert worker.status() == {
        "mode": "in_process", "alive": False, "last_tick_age_seconds": None, "restarts": 0,
    }

    fake_loop.set("run")
    worker.start()

    assert _wait_until(lambda: worker.status()["last_tick_age_seconds"] is not None)
    status = worker.status()
    assert status["alive"] is True
    assert 0.0 <= status["last_tick_age_seconds"] < 2.0


# --- /health ----------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = create_sqlite_job_store(str(tmp_path / "jobs.db"))
    monkeypatch.setattr(main, "get_job_store", lambda: s)
    return s


def test_health_reports_worker_and_queue_fields(store, worker, monkeypatch):
    monkeypatch.setattr(main, "RUN_WORKER_IN_PROCESS", True)
    monkeypatch.setattr(main, "_in_process_worker", worker)
    job, _ = store.create_job(source_filename="a.wav", storage_key="k/a")
    store.mark_creation_ready(job.id)

    payload = main.health()

    assert payload["status"] == "ok"
    assert payload["worker"] == {
        "mode": "in_process", "alive": False, "last_tick_age_seconds": None, "restarts": 0,
    }
    assert payload["queue"]["reachable"] is True
    assert payload["queue"]["oldest_queued_job_age_seconds"] is not None
    assert 0.0 <= payload["queue"]["oldest_queued_job_age_seconds"] < 5.0


def test_health_queue_age_is_none_when_nothing_is_queued(store, monkeypatch):
    monkeypatch.setattr(main, "RUN_WORKER_IN_PROCESS", True)

    assert main.health()["queue"] == {"reachable": True, "oldest_queued_job_age_seconds": None}


def test_health_does_not_claim_a_worker_when_it_runs_elsewhere(store, monkeypatch):
    monkeypatch.setattr(main, "RUN_WORKER_IN_PROCESS", False)

    assert main.health()["worker"] == {"mode": "external"}


def test_health_stays_ok_when_the_job_store_is_unreachable(monkeypatch):
    """/health is what keep-warm pings and uptime monitors hit: a database
    hiccup must read as queue.reachable=false, not turn the ping into a 500."""

    class BrokenStore:
        def oldest_queued_job_age_seconds(self):
            raise RuntimeError("database unreachable")

    monkeypatch.setattr(main, "get_job_store", lambda: BrokenStore())
    monkeypatch.setattr(main, "RUN_WORKER_IN_PROCESS", False)

    payload = main.health()

    assert payload["status"] == "ok"
    assert payload["queue"] == {"reachable": False, "oldest_queued_job_age_seconds": None}
