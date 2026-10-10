"""Guards the "zero logs in Render" fix (docs/PROGRESS.md, 2026-10-10).

Every app.* logger call was previously silently dropped in production: no
module here ever called logging.basicConfig/dictConfig, so app.inference /
app.main / app.jobs.* loggers had no handler anywhere up to root, and only
WARNING+ calls ever surfaced (via Python's logging.lastResort fallback,
which only fires at that level). Confirmed directly, not guessed: a real
local server run never printed its own logger.info() lines despite the
code path definitely executing, and the same gap was confirmed in the live
Render log stream (uvicorn's own access-log lines appeared; this project's
"Found joint checkpoint...", "Basic Pitch model warm-up complete", etc.
never did). Importing app.main must now attach a handler so INFO-level app
logging is actually visible.

Run in a fresh interpreter (subprocess), matching test_lazy_imports.py's
existing convention here -- pytest's own log capture, and whatever this
test run has already configured on the root logger, would otherwise make
this untestable in-process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _run_isolated(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )


def test_importing_app_main_makes_info_level_app_logging_visible() -> None:
    result = _run_isolated(
        """
        import logging
        import app.main  # noqa: F401

        logging.getLogger("app.inference").info("PROBE_LOG_LINE_12345")
        """
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    assert "PROBE_LOG_LINE_12345" in result.stdout + result.stderr, (
        "an INFO-level app.* logger call must be visible once app.main is "
        "imported -- it was previously silently dropped (no handler "
        "anywhere up to the root logger)"
    )


def test_logging_writes_to_stdout() -> None:
    """So Render's (or any stdout-oriented) log collector actually sees it."""
    result = _run_isolated(
        """
        import logging
        import app.main  # noqa: F401

        logging.getLogger("app.inference").info("PROBE_STDOUT_LINE_67890")
        """
    )
    assert result.returncode == 0
    assert "PROBE_STDOUT_LINE_67890" in result.stdout


def test_botocore_info_noise_is_silenced() -> None:
    """botocore logs "Found credentials in environment variables" at INFO
    every time a boto3 client is constructed (jobs/storage.py's B2/S3
    client) -- confirmed firing before this fix; must not compete with our
    own log lines in Render's stream."""
    result = _run_isolated(
        """
        import logging
        import os
        import app.main  # noqa: F401

        os.environ["AWS_ACCESS_KEY_ID"] = "test"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
        import boto3
        boto3.client("s3", region_name="us-east-1")
        """
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    assert "Found credentials" not in result.stdout + result.stderr
