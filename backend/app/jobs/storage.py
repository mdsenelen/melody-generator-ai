from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class ObjectStorage(Protocol):
    # True for a backend actually reachable from more than one process
    # (S3-compatible); False for LocalFilesystemStorage, where "shared" is
    # only true by accident (single dev process). Callers use this to
    # decide whether mirroring a file here is worth doing at all -- see
    # worker.py's _upload_generated_files.
    is_shared: bool

    def write_bytes(self, key: str, data: bytes) -> None: ...

    def read_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class LocalFilesystemStorage:
    """Dev/test default. Also what a single-instance production deployment
    falls back to if no bucket is configured -- fine as long as the web and
    worker processes share a disk, which is NOT Render's default (separate
    services get separate ephemeral disks), so production should configure
    JOB_STORAGE_BUCKET."""

    is_shared = False

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir.resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        candidate = (self._base_dir / key).resolve()
        if self._base_dir != candidate and self._base_dir not in candidate.parents:
            raise ValueError(f"storage key escapes base directory: {key!r}")
        return candidate

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()


class S3CompatibleStorage:
    """Works against AWS S3 or an S3-compatible endpoint (Cloudflare R2,
    etc) by pointing endpoint_url at the provider. Credentials come from
    the standard boto3 resolution chain (explicit args here, then env vars,
    then instance role) -- see create_object_storage_from_env for the env
    vars this app reads explicitly."""

    is_shared = True

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - exercised only when a bucket is configured
            raise RuntimeError(
                "JOB_STORAGE_BUCKET is set but boto3 is not installed; add boto3 "
                "to requirements.txt / your production environment"
            ) from exc

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def write_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def read_bytes(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False


_DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "job_storage"


def create_object_storage_from_env(
    env: Optional[dict[str, str]] = None,
    default_local_dir: Path = _DEFAULT_LOCAL_DIR,
) -> ObjectStorage:
    import os

    env = env if env is not None else os.environ
    bucket = env.get("JOB_STORAGE_BUCKET")
    if bucket:
        logger.info("Using S3-compatible object storage (bucket=%s)", bucket)
        return S3CompatibleStorage(
            bucket=bucket,
            endpoint_url=env.get("JOB_STORAGE_ENDPOINT_URL"),
            region_name=env.get("JOB_STORAGE_REGION", "auto"),
            access_key=env.get("JOB_STORAGE_ACCESS_KEY_ID"),
            secret_key=env.get("JOB_STORAGE_SECRET_ACCESS_KEY"),
        )
    logger.info("Using local filesystem object storage (set JOB_STORAGE_BUCKET for production)")
    return LocalFilesystemStorage(default_local_dir)
