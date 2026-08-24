from __future__ import annotations

import random
import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request

# Single-process, in-memory limiter. Correct as long as there's one web
# instance and no dedicated worker service (see CLAUDE.md's Production
# Readiness) -- state isn't shared across processes, so this would need to
# move to a shared store (e.g. the Redis already used for the job queue) if
# the app ever scales to multiple web instances.
_LOCK = threading.Lock()
_HITS: dict[tuple[str, str], list[float]] = defaultdict(list)

# Bigger than any configured window, so an opportunistic sweep (below) can
# drop buckets for IPs that never come back without needing a scheduled
# background task -- otherwise a bucket only gets pruned when its own IP
# makes another request, and one that never returns leaks forever.
_STALE_BUCKET_TTL_SECONDS = 3600.0
_SWEEP_PROBABILITY = 1 / 500


def _client_ip(request: Request) -> str:
    # Render (like most PaaS front ends) terminates the connection at a
    # proxy, so request.client.host is the proxy's address, not the real
    # caller -- X-Forwarded-For's left-most entry is the original client.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _sweep_stale_buckets(now: float) -> None:
    stale_keys = [key for key, hits in _HITS.items() if not hits or hits[-1] < now - _STALE_BUCKET_TTL_SECONDS]
    for key in stale_keys:
        del _HITS[key]


def rate_limiter(bucket_name: str, limit: int, window_seconds: float):
    """Returns a FastAPI dependency enforcing `limit` requests per
    `window_seconds` per client IP, scoped to `bucket_name` so different
    routes don't share a budget."""

    async def _check(request: Request) -> None:
        ip = _client_ip(request)
        key = (bucket_name, ip)
        now = time.monotonic()
        cutoff = now - window_seconds
        with _LOCK:
            hits = _HITS[key]
            while hits and hits[0] < cutoff:
                hits.pop(0)

            if len(hits) >= limit:
                retry_after = max(1, int(hits[0] + window_seconds - now) + 1)
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests. Please retry after {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )

            hits.append(now)

            if random.random() < _SWEEP_PROBABILITY:
                _sweep_stale_buckets(now)

    return _check
