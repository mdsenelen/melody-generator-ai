from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from .models import COMPLETED, CREATING, FAILED, PROCESSING, QUEUED, Job

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcription_jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    upload_id TEXT,
    idempotency_key TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    progress INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    result TEXT,
    error TEXT,
    lease_token TEXT,
    lease_expires_at TEXT,
    next_attempt_at TEXT
)
"""
# Additive migration for a local dev DB created before lease_token /
# lease_expires_at / next_attempt_at existed. Nothing is deployed to
# production yet (no live data), so this only needs to handle a developer's
# existing backend/data/jobs.db -- each ADD COLUMN is attempted and a
# "duplicate column" failure is swallowed.
_MIGRATION_COLUMNS = (
    ("lease_token", "TEXT"),
    ("lease_expires_at", "TEXT"),
    ("next_attempt_at", "TEXT"),
)
_STATUS_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_transcription_jobs_status "
    "ON transcription_jobs (status)"
)
# Partial unique index: at most one non-permanently-failed job per
# idempotency key. Enforced by the database itself (not a Python lock), so
# it's correct across any number of web instances/processes sharing the
# same database -- a failed job is excluded so a genuinely fresh retry
# after a permanent failure can still get a new row for the same key.
# Postgres and SQLite (3.24+) both support this exact syntax, including the
# matching `ON CONFLICT (...) WHERE ...` used in create_job below.
_IDEMPOTENCY_UNIQUE_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_transcription_jobs_idempotency "
    "ON transcription_jobs (idempotency_key) WHERE status != 'failed'"
)
# Older SQLJobStore versions created a plain (non-unique) index with this
# name on the same column; drop it if present so it doesn't shadow/conflict
# with the unique one above on an upgraded local dev DB.
_DROP_LEGACY_IDEMPOTENCY_INDEX = "DROP INDEX IF EXISTS ix_transcription_jobs_idempotency"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retry_backoff_seconds(attempt_count: int) -> float:
    # 2s, 4s, 8s, 16s, ... capped at 60s -- comfortably under
    # DEFAULT_STALE_AFTER_SECONDS (180s) so a backoff delay is never
    # mistaken for a dead worker.
    return min(60.0, 2.0 * (2 ** max(0, attempt_count - 1)))


class JobStore(Protocol):
    def create_job(
        self,
        *,
        source_filename: str,
        storage_key: str,
        upload_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        max_attempts: int = 3,
        job_id: Optional[str] = None,
    ) -> tuple[Job, bool]: ...

    def get_job(self, job_id: str) -> Optional[Job]: ...

    def mark_creation_ready(self, job_id: str) -> bool: ...

    def claim_job(self, job_id: str, lease_seconds: float) -> Optional[Job]: ...

    def mark_completed(self, job_id: str, result: dict[str, Any], *, lease_token: str) -> bool: ...

    def mark_failed(
        self, job_id: str, error: str, *, retry: bool, lease_token: Optional[str] = None
    ) -> bool: ...

    def reclaim_stale_processing_jobs(self) -> list[str]: ...

    def get_queued_job_ids(self) -> list[str]: ...

    def get_ready_retry_job_ids(self) -> list[str]: ...

    def reconcile_stuck_creating_jobs(
        self, stale_after_seconds: float, exists: Callable[[str], bool]
    ) -> list[str]: ...


def _row_to_job(row: tuple) -> Job:
    (
        id_, status, source_filename, storage_key, upload_id, idempotency_key,
        attempt_count, max_attempts, progress, created_at, updated_at,
        started_at, completed_at, result_json, error,
        lease_token, lease_expires_at, next_attempt_at,
    ) = row
    return Job(
        id=id_,
        status=status,
        source_filename=source_filename,
        storage_key=storage_key,
        upload_id=upload_id,
        idempotency_key=idempotency_key,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        progress=progress,
        created_at=created_at,
        updated_at=updated_at,
        started_at=started_at,
        completed_at=completed_at,
        result=json.loads(result_json) if result_json else None,
        error=error,
        lease_token=lease_token,
        lease_expires_at=lease_expires_at,
        next_attempt_at=next_attempt_at,
    )


_COLUMNS = (
    "id, status, source_filename, storage_key, upload_id, idempotency_key, "
    "attempt_count, max_attempts, progress, created_at, updated_at, "
    "started_at, completed_at, result, error, "
    "lease_token, lease_expires_at, next_attempt_at"
)


class SQLJobStore:
    """Job metadata store backed by any DB-API 2.0 connection.

    Works unmodified against both sqlite3 (dev/tests) and psycopg (Postgres,
    production) -- the only per-backend difference is the parameter
    placeholder. Every state transition that matters for correctness
    (creation dedup, claim, completion, reclaim) is a single conditional
    SQL statement rather than read-then-write from Python, so it's correct
    under real concurrency (multiple web instances, multiple workers), not
    just under a single process's in-memory lock.
    """

    def __init__(self, connect: Callable[[], Any], placeholder: str = "?"):
        self._connect = connect
        self._ph = placeholder
        self._init_schema()

    def _q(self, sql: str) -> str:
        return sql.replace("?", self._ph) if self._ph != "?" else sql

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(_SCHEMA)
            # Commit before the migration loop below: on Postgres, an
            # ADD COLUMN failure (e.g. the column already exists on a
            # freshly created table, since _SCHEMA already defines it)
            # aborts the whole open transaction, and the except branch's
            # rollback() would otherwise discard this still-uncommitted
            # CREATE TABLE along with it. SQLite doesn't have this failure
            # mode, but committing here is harmless for it too.
            conn.commit()
            for column, coltype in _MIGRATION_COLUMNS:
                try:
                    cur.execute(f"ALTER TABLE transcription_jobs ADD COLUMN {column} {coltype}")
                    conn.commit()
                except Exception:
                    conn.rollback()
            cur.execute(_DROP_LEGACY_IDEMPOTENCY_INDEX)
            cur.execute(_IDEMPOTENCY_UNIQUE_INDEX)
            cur.execute(_STATUS_INDEX)
            conn.commit()
        finally:
            conn.close()

    def create_job(
        self,
        *,
        source_filename: str,
        storage_key: str,
        upload_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        max_attempts: int = 3,
        job_id: Optional[str] = None,
    ) -> tuple[Job, bool]:
        # Atomic upsert: the partial unique index on idempotency_key (see
        # _IDEMPOTENCY_UNIQUE_INDEX) is what makes this safe across
        # multiple web instances -- if a row for this key already exists
        # and isn't 'failed', the INSERT becomes a no-op at the database
        # level (not a Python-level check-then-act), so two processes
        # racing to create the same job can't both succeed.
        job_id = job_id or uuid.uuid4().hex
        now = _now()
        conn = self._connect()
        try:
            cur = conn.cursor()
            if idempotency_key:
                cur.execute(
                    self._q(
                        "INSERT INTO transcription_jobs "
                        "(id, status, source_filename, storage_key, upload_id, "
                        "idempotency_key, attempt_count, max_attempts, progress, "
                        "created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0, ?, 0, ?, ?) "
                        "ON CONFLICT (idempotency_key) WHERE status != 'failed' DO NOTHING"
                    ),
                    (job_id, CREATING, source_filename, storage_key, upload_id,
                     idempotency_key, max_attempts, now, now),
                )
            else:
                cur.execute(
                    self._q(
                        "INSERT INTO transcription_jobs "
                        "(id, status, source_filename, storage_key, upload_id, "
                        "idempotency_key, attempt_count, max_attempts, progress, "
                        "created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0, ?, 0, ?, ?)"
                    ),
                    (job_id, CREATING, source_filename, storage_key, upload_id,
                     idempotency_key, max_attempts, now, now),
                )
            conn.commit()
        finally:
            conn.close()

        # Whichever row now represents this idempotency key is the winner
        # of the race, regardless of which process's INSERT actually took
        # effect -- re-reading it (rather than trusting our own job_id) is
        # what makes the *caller* see a consistent result too.
        winner = self._find_by_idempotency_key(idempotency_key) if idempotency_key else self.get_job(job_id)
        assert winner is not None
        return winner, winner.id == job_id

    def _find_by_idempotency_key(self, idempotency_key: str) -> Optional[Job]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    f"SELECT {_COLUMNS} FROM transcription_jobs "
                    "WHERE idempotency_key = ? AND status != ? "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                (idempotency_key, FAILED),
            )
            row = cur.fetchone()
            return _row_to_job(tuple(row)) if row else None
        finally:
            conn.close()

    def get_job(self, job_id: str) -> Optional[Job]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(f"SELECT {_COLUMNS} FROM transcription_jobs WHERE id = ?"),
                (job_id,),
            )
            row = cur.fetchone()
            return _row_to_job(tuple(row)) if row else None
        finally:
            conn.close()

    def mark_creation_ready(self, job_id: str) -> bool:
        """Transition creating -> queued once input bytes are durably
        stored. Returns False if the job wasn't in 'creating' (already
        handled, e.g. by the stuck-creating reconciler)."""
        now = _now()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "UPDATE transcription_jobs SET status = ?, updated_at = ? "
                    "WHERE id = ? AND status = ?"
                ),
                (QUEUED, now, job_id, CREATING),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()

    def claim_job(self, job_id: str, lease_seconds: float) -> Optional[Job]:
        """Atomically transition queued -> processing and issue a fresh
        lease token. Returns None if the job doesn't exist or isn't queued
        (already claimed by another worker, already terminal, etc) -- the
        caller should treat that as "nothing to do", not an error."""
        now = _now()
        lease_token = uuid.uuid4().hex
        lease_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "UPDATE transcription_jobs "
                    "SET status = ?, started_at = ?, updated_at = ?, progress = 50, "
                    "lease_token = ?, lease_expires_at = ?, next_attempt_at = NULL "
                    "WHERE id = ? AND status = ?"
                ),
                (PROCESSING, now, now, lease_token, lease_expires_at, job_id, QUEUED),
            )
            conn.commit()
            claimed = cur.rowcount == 1
        finally:
            conn.close()
        return self.get_job(job_id) if claimed else None

    def mark_completed(self, job_id: str, result: dict[str, Any], *, lease_token: str) -> bool:
        """Returns False (write discarded) if lease_token no longer matches
        -- the job was reclaimed as stale and a different attempt now owns
        it. The caller must not treat this as its own failure; it just
        means this attempt's result doesn't count anymore."""
        now = _now()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "UPDATE transcription_jobs "
                    "SET status = ?, result = ?, progress = 100, "
                    "updated_at = ?, completed_at = ? WHERE id = ? AND lease_token = ?"
                ),
                (COMPLETED, json.dumps(result), now, now, job_id, lease_token),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()

    def mark_failed(
        self, job_id: str, error: str, *, retry: bool, lease_token: Optional[str] = None
    ) -> bool:
        """lease_token=None is used only by the stale-reclaim path itself
        (which owns the transition unconditionally, since it's the one
        deciding a lease has expired); every other caller must pass the
        lease_token it was issued at claim time, and the write is a no-op
        if that token is no longer current. Returns whether the write took
        effect."""
        now = _now()
        conn = self._connect()
        try:
            cur = conn.cursor()
            lease_clause = "" if lease_token is None else " AND lease_token = ?"
            params_prefix = () if lease_token is None else (lease_token,)

            cur.execute(
                self._q(
                    f"SELECT attempt_count, max_attempts FROM transcription_jobs "
                    f"WHERE id = ?{lease_clause}"
                ),
                (job_id, *params_prefix),
            )
            row = cur.fetchone()
            if row is None:
                return False
            attempt_count, max_attempts = row[0] + 1, row[1]

            if retry and attempt_count < max_attempts:
                next_attempt_at = (
                    datetime.now(timezone.utc)
                    + timedelta(seconds=_retry_backoff_seconds(attempt_count))
                ).isoformat()
                cur.execute(
                    self._q(
                        "UPDATE transcription_jobs "
                        "SET status = ?, attempt_count = ?, updated_at = ?, "
                        "started_at = NULL, error = ?, progress = 0, "
                        "lease_token = NULL, lease_expires_at = NULL, next_attempt_at = ? "
                        f"WHERE id = ?{lease_clause}"
                    ),
                    (QUEUED, attempt_count, now, error, next_attempt_at, job_id, *params_prefix),
                )
            else:
                cur.execute(
                    self._q(
                        "UPDATE transcription_jobs "
                        "SET status = ?, attempt_count = ?, updated_at = ?, "
                        "completed_at = ?, error = ?, lease_token = NULL, "
                        "lease_expires_at = NULL, next_attempt_at = NULL "
                        f"WHERE id = ?{lease_clause}"
                    ),
                    (FAILED, attempt_count, now, now, error, job_id, *params_prefix),
                )
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()

    def reclaim_stale_processing_jobs(self) -> list[str]:
        """Recover jobs whose worker crashed (or is presumed dead) mid-
        processing. Staleness is governed entirely by lease_expires_at,
        set at claim time from claim_job's lease_seconds -- there's no
        separate "how long is too long" threshold here; whoever claims a
        job already decided that when it claimed it. Unlike a select-then-
        update, the lease is cleared in the *same* UPDATE that requeues
        the job, so it's atomic with respect to the original worker's own
        completion write: whichever of the two statements the database
        serializes first wins, and the other one's WHERE clause
        (status='processing' here, or lease_token=<old> there) simply
        matches zero rows."""
        now = _now()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    f"SELECT {_COLUMNS} FROM transcription_jobs "
                    "WHERE status = ? AND lease_expires_at IS NOT NULL AND lease_expires_at < ?"
                ),
                (PROCESSING, now),
            )
            stale = [_row_to_job(tuple(row)) for row in cur.fetchall()]

            requeued: list[str] = []
            for job in stale:
                attempt_count = job.attempt_count + 1
                if attempt_count < job.max_attempts:
                    cur.execute(
                        self._q(
                            "UPDATE transcription_jobs "
                            "SET status = ?, attempt_count = ?, updated_at = ?, "
                            "started_at = NULL, error = ?, progress = 0, "
                            "lease_token = NULL, lease_expires_at = NULL, next_attempt_at = ? "
                            "WHERE id = ? AND status = ? AND lease_expires_at < ?"
                        ),
                        (QUEUED, attempt_count, now,
                         "Worker did not finish processing this job in time",
                         now, job.id, PROCESSING, now),
                    )
                    if cur.rowcount == 1:
                        requeued.append(job.id)
                else:
                    cur.execute(
                        self._q(
                            "UPDATE transcription_jobs "
                            "SET status = ?, attempt_count = ?, updated_at = ?, "
                            "completed_at = ?, error = ?, lease_token = NULL, "
                            "lease_expires_at = NULL, next_attempt_at = NULL "
                            "WHERE id = ? AND status = ? AND lease_expires_at < ?"
                        ),
                        (FAILED, attempt_count, now, now,
                         "Worker did not finish processing this job in time (max attempts reached)",
                         job.id, PROCESSING, now),
                    )
            conn.commit()
        finally:
            conn.close()
        return requeued

    def get_queued_job_ids(self) -> list[str]:
        """Jobs sitting in 'queued' with no corresponding queue message --
        e.g. after an InProcessJobQueue restart, which loses unclaimed
        messages. Used once at worker startup to re-enqueue them. Excludes
        jobs with a future next_attempt_at (backoff delay in progress);
        those are handled by get_ready_retry_job_ids instead."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "SELECT id FROM transcription_jobs "
                    "WHERE status = ? AND next_attempt_at IS NULL ORDER BY created_at"
                ),
                (QUEUED,),
            )
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def get_ready_retry_job_ids(self) -> list[str]:
        """Queued retries whose backoff delay has elapsed. Clears
        next_attempt_at as part of the same call so the next idle tick
        doesn't return the same job again before a worker has claimed it
        (claim_job also clears it, but that's only reached once a worker
        actually dequeues the id this method hands back)."""
        now = _now()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "SELECT id FROM transcription_jobs "
                    "WHERE status = ? AND next_attempt_at IS NOT NULL AND next_attempt_at <= ?"
                ),
                (QUEUED, now),
            )
            ready = [row[0] for row in cur.fetchall()]
            for job_id in ready:
                cur.execute(
                    self._q(
                        "UPDATE transcription_jobs SET next_attempt_at = NULL "
                        "WHERE id = ? AND status = ?"
                    ),
                    (job_id, QUEUED),
                )
            conn.commit()
            return ready
        finally:
            conn.close()

    def reconcile_stuck_creating_jobs(
        self, stale_after_seconds: float, exists: Callable[[str], bool]
    ) -> list[str]:
        """Recovers from a crash between the DB insert and the storage
        write / queued transition in create_transcription_job (see
        service.py). `exists` checks whether the job's input bytes actually
        made it to object storage: if so, the row just missed its queued
        transition and is safe to release; if not, the input was never
        durably stored and the job can't be run. Returns ids that were
        released to 'queued' (the caller enqueues them)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)).isoformat()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "SELECT id, storage_key FROM transcription_jobs "
                    "WHERE status = ? AND created_at < ?"
                ),
                (CREATING, cutoff),
            )
            stuck = cur.fetchall()
        finally:
            conn.close()

        released: list[str] = []
        for job_id, storage_key in stuck:
            if exists(storage_key):
                if self.mark_creation_ready(job_id):
                    released.append(job_id)
            else:
                self.mark_failed(
                    job_id,
                    "Job creation did not complete (input audio was never stored)",
                    retry=False,
                )
        return released


_DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "jobs.db"


def _sqlite_connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_sqlite_job_store(path: Optional[str] = None) -> SQLJobStore:
    db_path = path or str(_DEFAULT_SQLITE_PATH)
    return SQLJobStore(connect=lambda: _sqlite_connect(db_path), placeholder="?")


def create_postgres_job_store(dsn: str) -> SQLJobStore:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised only when DATABASE_URL is set
        raise RuntimeError(
            "DATABASE_URL is set but psycopg is not installed; add psycopg[binary] "
            "to requirements.txt / your production environment"
        ) from exc

    return SQLJobStore(connect=lambda: psycopg.connect(dsn), placeholder="%s")


def create_job_store_from_env(env: Optional[dict[str, str]] = None) -> SQLJobStore:
    import os

    env = env if env is not None else os.environ
    database_url = env.get("DATABASE_URL")
    if database_url:
        logger.info("Using Postgres job store")
        return create_postgres_job_store(database_url)
    logger.info("Using SQLite job store (set DATABASE_URL for Postgres in production)")
    return create_sqlite_job_store(env.get("JOB_DB_PATH"))
