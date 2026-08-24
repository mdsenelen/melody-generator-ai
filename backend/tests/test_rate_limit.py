from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference, rate_limit  # noqa: E402
from app.jobs import routes as job_routes  # noqa: E402
from app.jobs import service  # noqa: E402
from app.jobs.queue import InProcessJobQueue  # noqa: E402
from app.jobs.storage import LocalFilesystemStorage  # noqa: E402
from app.jobs.store import create_sqlite_job_store  # noqa: E402
from app.main import app  # noqa: E402


def _make_request(client_ip: str = "1.2.3.4", forwarded_for: Optional[str] = None) -> Request:
    headers = []
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    scope = {
        "type": "http",
        "headers": headers,
        "client": (client_ip, 12345),
    }
    return Request(scope)


class _FakeClock:
    def __init__(self, start: float = 1_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def clean_hits():
    rate_limit._HITS.clear()
    yield
    rate_limit._HITS.clear()


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(rate_limit.time, "monotonic", fake)
    return fake


def test_allows_exactly_limit_requests_then_blocks(clock):
    check = rate_limit.rate_limiter("bucket", limit=3, window_seconds=60)
    request = _make_request()

    for _ in range(3):
        asyncio.run(check(request))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(check(request))

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers
    assert int(exc_info.value.headers["Retry-After"]) > 0


def test_request_succeeds_again_after_window_elapses(clock):
    check = rate_limit.rate_limiter("bucket", limit=1, window_seconds=60)
    request = _make_request()

    asyncio.run(check(request))
    with pytest.raises(HTTPException):
        asyncio.run(check(request))

    clock.advance(61)
    asyncio.run(check(request))  # should not raise


def test_different_ips_get_independent_buckets(clock):
    check = rate_limit.rate_limiter("bucket", limit=1, window_seconds=60)
    request_a = _make_request(client_ip="1.1.1.1")
    request_b = _make_request(client_ip="2.2.2.2")

    asyncio.run(check(request_a))
    asyncio.run(check(request_b))  # different IP, should not raise

    with pytest.raises(HTTPException):
        asyncio.run(check(request_a))


def test_x_forwarded_for_left_most_address_is_used(clock):
    check = rate_limit.rate_limiter("bucket", limit=1, window_seconds=60)
    request_a = _make_request(client_ip="10.0.0.1", forwarded_for="9.9.9.9, 10.0.0.1")
    request_b = _make_request(client_ip="10.0.0.1", forwarded_for="9.9.9.9, 10.0.0.2")

    asyncio.run(check(request_a))
    with pytest.raises(HTTPException):
        # Same forwarded client (9.9.9.9) despite a different immediate peer.
        asyncio.run(check(request_b))


def test_different_bucket_names_do_not_share_budget(clock):
    transcribe_check = rate_limit.rate_limiter("transcribe", limit=1, window_seconds=60)
    variants_check = rate_limit.rate_limiter("generate-variants", limit=1, window_seconds=60)
    request = _make_request()

    asyncio.run(transcribe_check(request))
    asyncio.run(variants_check(request))  # separate bucket, should not raise

    with pytest.raises(HTTPException):
        asyncio.run(transcribe_check(request))


def test_stale_bucket_sweep_removes_old_entries(clock, monkeypatch):
    check = rate_limit.rate_limiter("bucket", limit=5, window_seconds=1)
    old_request = _make_request(client_ip="1.1.1.1")
    asyncio.run(check(old_request))
    assert len(rate_limit._HITS) == 1

    clock.advance(rate_limit._STALE_BUCKET_TTL_SECONDS + 1)

    # Force the opportunistic sweep to fire deterministically, triggered by
    # a different bucket's request.
    monkeypatch.setattr(rate_limit.random, "random", lambda: 0.0)
    new_request = _make_request(client_ip="2.2.2.2")
    asyncio.run(check(new_request))

    assert ("bucket", "1.1.1.1") not in rate_limit._HITS
    assert ("bucket", "2.2.2.2") in rate_limit._HITS


# --- wiring: the dependency is actually attached to the real routes --------
#
# test_jobs.py's route-level tests call job_routes.create_transcribe_job(...)
# as a plain Python function, which never goes through FastAPI's dependency
# injection -- so those tests wouldn't catch a `dependencies=[...]` on the
# decorator being missing or misconfigured. These go through TestClient
# (the real ASGI stack) specifically to catch that class of bug.


@pytest.fixture
def client():
    return TestClient(app)


def test_transcribe_route_returns_429_once_over_its_limit(client, monkeypatch, tmp_path):
    store = create_sqlite_job_store(str(tmp_path / "jobs.db"))
    queue = InProcessJobQueue()
    storage = LocalFilesystemStorage(tmp_path / "storage")
    monkeypatch.setattr(service, "_JOB_STORE", store)
    monkeypatch.setattr(service, "_JOB_QUEUE", queue)
    monkeypatch.setattr(service, "_OBJECT_STORAGE", storage)

    statuses = []
    for i in range(job_routes.TRANSCRIBE_RATE_LIMIT + 1):
        response = client.post(
            "/api/transcribe",
            files={"file": (f"clip{i}.wav", b"audio-bytes", "audio/wav")},
        )
        statuses.append(response.status_code)

    assert statuses[:-1] == [202] * job_routes.TRANSCRIBE_RATE_LIMIT
    assert statuses[-1] == 429


def test_generate_variants_route_returns_429_once_over_its_limit(client, monkeypatch):
    # The rate limiter dependency runs before the route body, so a request
    # that the route itself would reject (no matching upload here) still
    # counts against the limit -- only the final, limit-exceeding call needs
    # to actually be a 429.
    statuses = []
    for i in range(inference.GENERATE_VARIANTS_RATE_LIMIT + 1):
        response = client.post(
            "/api/generate-variants",
            data={"filename": f"does-not-exist-{i}.wav"},
        )
        statuses.append(response.status_code)

    assert statuses[-1] == 429
    assert all(status in (404, 429) for status in statuses)
