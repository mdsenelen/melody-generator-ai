from __future__ import annotations

import logging
import queue
import threading
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class JobQueue(Protocol):
    def enqueue(self, job_id: str) -> None: ...

    def dequeue(self, timeout: float) -> Optional[str]: ...


class InProcessJobQueue:
    """Thread-safe, in-memory queue for local dev/tests and single-instance
    deployments. A job that's already 'processing' when the process dies is
    recovered by SQLJobStore.reclaim_stale_processing_jobs. A job that's
    still merely 'queued' (created but never claimed) has no such recovery
    path here, since it never had a queue message to lose -- main.py's
    startup re-enqueues any jobs already in 'queued' status for exactly this
    reason.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[str]" = queue.Queue()

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def dequeue(self, timeout: float) -> Optional[str]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class RedisJobQueue:
    """Redis-backed queue for production: BRPUSH/BLPOP give cheap blocking
    pickup across multiple worker processes/instances, and the list survives
    a worker restart (unlike InProcessJobQueue)."""

    def __init__(self, redis_url: str, list_key: str = "transcription_jobs"):
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - exercised only when REDIS_URL is set
            raise RuntimeError(
                "REDIS_URL is set but the redis package is not installed; add redis "
                "to requirements.txt / your production environment"
            ) from exc

        self._redis = redis.Redis.from_url(redis_url)
        self._list_key = list_key

    def enqueue(self, job_id: str) -> None:
        self._redis.lpush(self._list_key, job_id)

    def dequeue(self, timeout: float) -> Optional[str]:
        # brpop's timeout must be a whole number of seconds; round up so a
        # sub-second caller timeout still blocks at least briefly instead of
        # becoming a busy-poll (timeout=0 means "block forever" in Redis).
        whole_seconds = max(1, int(timeout + 0.999))
        item = self._redis.brpop([self._list_key], timeout=whole_seconds)
        if item is None:
            return None
        _, value = item
        return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


_IN_PROCESS_SINGLETON: Optional[InProcessJobQueue] = None
_SINGLETON_LOCK = threading.Lock()


def create_job_queue_from_env(env: Optional[dict[str, str]] = None) -> JobQueue:
    import os

    env = env if env is not None else os.environ
    redis_url = env.get("REDIS_URL")
    if redis_url:
        logger.info("Using Redis job queue")
        return RedisJobQueue(redis_url)

    # The in-process queue must be a single shared instance per process: the
    # web handler's enqueue() and the in-process worker thread's dequeue()
    # need to see the same underlying queue.Queue.
    global _IN_PROCESS_SINGLETON
    with _SINGLETON_LOCK:
        if _IN_PROCESS_SINGLETON is None:
            logger.info("Using in-process job queue (set REDIS_URL for production)")
            _IN_PROCESS_SINGLETON = InProcessJobQueue()
        return _IN_PROCESS_SINGLETON
