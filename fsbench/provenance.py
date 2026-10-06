"""Run provenance: version, git commit, timestamp. Lets any result be reconstructed."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(cwd: str | Path | None = None) -> str | None:
    """HEAD hash, with a ``-dirty`` suffix if the working tree has uncommitted changes."""
    kw = {"capture_output": True, "text": True, "timeout": 3, "cwd": cwd}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], **kw)
        if head.returncode != 0:
            return None
        commit = head.stdout.strip()
        if not commit:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], **kw)
        if dirty.returncode == 0 and dirty.stdout.strip():
            return f"{commit}-dirty"
        return commit
    except (OSError, subprocess.TimeoutExpired):
        return None


def snapshot() -> dict:
    from fsbench import __version__

    return {
        "fsbench_version": __version__,
        "git_commit": git_commit(),
        "timestamp": utc_timestamp(),
        "python_version": platform.python_version(),
    }
