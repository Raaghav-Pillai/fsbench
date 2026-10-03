from __future__ import annotations

import os
from pathlib import Path


def long_path(p: str | os.PathLike) -> Path:
    """Absolute path that bypasses the 260-char MAX_PATH limit on Windows."""
    s = str(Path(p).resolve())
    if os.name != "nt" or s.startswith("\\\\?\\"):
        return Path(s)
    if s.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + s[2:])
    return Path("\\\\?\\" + s)


def display_path(p: str | os.PathLike) -> str:
    s = str(p)
    if s.startswith("\\\\?\\UNC\\"):
        return "\\\\" + s[8:]
    return s[4:] if s.startswith("\\\\?\\") else s
