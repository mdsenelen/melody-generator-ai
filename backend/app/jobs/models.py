from __future__ import annotations

import dataclasses
from typing import Any, Optional

# CREATING is an internal-only pre-queued state (row exists, input bytes not
# necessarily durable yet) -- never exposed externally. to_status_payload()
# reports it as "queued" so the public API's status enum stays exactly
# queued|processing|completed|failed.
CREATING = "creating"
QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"
# Never stored -- computed at read time (see jobs/routes.py) once a
# terminal job's generated files are old enough that DATA_RETENTION_HOURS
# has almost certainly already deleted them from disk. Reported instead of
# "completed"/"failed" so a stale link reads as gone, not as a fresh error.
EXPIRED = "expired"

# Statuses a job never leaves once reached.
TERMINAL_STATUSES = {COMPLETED, FAILED}


@dataclasses.dataclass
class Job:
    id: str
    status: str
    source_filename: str
    storage_key: str
    created_at: str
    updated_at: str
    upload_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempt_count: int = 0
    max_attempts: int = 3
    progress: int = 0
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    # Fencing token: only the worker holding the lease that matches the
    # job's current lease_token may transition it to completed/failed. Set
    # fresh on every claim (including a stale-job reclaim), so a worker
    # whose job was reclaimed out from under it can't overwrite a newer
    # attempt's result -- its write will match zero rows once the token
    # has moved on.
    lease_token: Optional[str] = None
    lease_expires_at: Optional[str] = None
    # Backoff scheduling for a retried job sitting in 'queued': set by
    # mark_failed(retry=True), cleared once the worker loop's idle-tick
    # sweep re-enqueues it.
    next_attempt_at: Optional[str] = None

    def to_status_payload(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": "queued" if self.status == CREATING else self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }
