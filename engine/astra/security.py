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


def authorized_write_path(value: str | Path, allowed_root: str | Path) -> Path:
    """Accept only paths below `allowed_root`, for renderer-supplied write
    targets (e.g. a checkpoint path) rather than reads under the data root.

    Unlike `authorized_path`, the target need not already exist -- a write
    creates it -- so this only checks containment, never `is_file()`.
    """
    root = Path(allowed_root).expanduser().resolve()
    candidate = Path(value).expanduser().resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"path is outside the allowed directory {root}") from exc
    return candidate


def scoped_id_path(directory: Path, identifier: str, suffix: str = ".json") -> Path:
    """Build `directory/{identifier}{suffix}`, rejecting an `identifier` that
    would resolve outside `directory` (e.g. containing `..` or a path
    separator). `identifier` is a renderer-supplied id such as a
    `dataset_id`, used as a bare filename stem, never as a path itself.
    """
    base = Path(directory).expanduser().resolve()
    candidate = (base / f"{identifier}{suffix}").resolve()
    if candidate.parent != base:
        raise ValueError(f"identifier {identifier!r} escapes the managed directory")
    return candidate
