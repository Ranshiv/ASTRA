"""Filesystem authorization for renderer-originated RPC reads."""
from __future__ import annotations

from pathlib import Path

from . import config


def authorized_path(value: str | Path) -> Path:
    """Accept only existing files below ASTRA's managed data root."""
    candidate = Path(value).expanduser().resolve()
    try:
        candidate.relative_to(config.PATHS.root.resolve())
    except ValueError as exc:
        raise PermissionError("file path is outside the ASTRA data root") from exc
    if not candidate.is_file():
        raise FileNotFoundError(str(candidate))
    return candidate
