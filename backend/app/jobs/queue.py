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


# Client-side read deadline for every Redis call, and the longest a single
# BRPOP may block. The deadline must stay comfortably above the block.
REDIS_SOCKET_TIMEOUT_SECONDS = 30.0
MAX_BLOCK_SECONDS = 20


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

        # The client read deadline must be finite but strictly longer than any
        # BRPOP block. Two failure modes pull in opposite directions:
        #   * too tight (redis-py's default 5s == the worker's 5s poll): the
        #     client's deadline races BRPOP's server-side "timed out, here's
        #     nil" reply and dequeue() intermittently raised TimeoutError.
        #   * unbounded (socket_timeout=None, the previous setting): a
        #     half-open connection (Redis restarted, an idle flow dropped by
        #     the network) blocks BRPOP forever with no exception, so the
        #     worker thread looks alive and consumes nothing. Jobs then sit
        #     in 'queued' indefinitely -- the 2026-11 production incident.
        # dequeue() clamps its BRPOP block to MAX_BLOCK_SECONDS, which is
        # well under REDIS_SOCKET_TIMEOUT_SECONDS, so the two can't race;
        # keepalive + health checks let a dead connection be noticed and
        # replaced between blocks. socket_connect_timeout stays short so an
        # unreachable Redis still fails fast on connect.
        self._redis = redis.Redis.from_url(
            redis_url,
            socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
            socket_keepalive=True,
            health_check_interval=15,
        )
        self._list_key = list_key

    def enqueue(self, job_id: str) -> None:
        self._redis.lpush(self._list_key, job_id)

    def dequeue(self, timeout: float) -> Optional[str]:
        # brpop's timeout must be a whole number of seconds; round up so a
        # sub-second caller timeout still blocks at least briefly instead of
        # becoming a busy-poll (timeout=0 means "block forever" in Redis).
        whole_seconds = min(MAX_BLOCK_SECONDS, max(1, int(timeout + 0.999)))
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
