"""Standalone transcription worker process.

Production entrypoint -- run this as its own Render Background Worker
service (NOT inside the web service), so Basic Pitch inference never runs
on a thread the web process spun up as a fire-and-forget workaround. See
CLAUDE.md's "Async Transcription Job Workflow" section for the full picture.

    python -m app.worker_main

Locally, you don't need to run this separately: main.py's lifespan starts
an equivalent worker on a background thread inside the dev server unless
RUN_WORKER_IN_PROCESS=false, so `uvicorn app.main:app --reload` alone is
enough for local development.
"""
from __future__ import annotations

import logging
import signal
import threading
from types import FrameType
from typing import Optional

from app.jobs.service import get_job_queue, get_job_store, get_object_storage
from app.jobs.worker import run_worker_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: Optional[FrameType]) -> None:
        logger.info("Worker received signal %s, shutting down", signal.Signals(signum).name)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Transcription worker starting")
    run_worker_loop(
        store=get_job_store(),
        queue=get_job_queue(),
        storage=get_object_storage(),
        stop_event=stop_event,
    )
    logger.info("Transcription worker stopped")


if __name__ == "__main__":
    main()
