"""Filesystem authorization boundary for renderer-originated RPC input.

`security.py` is the only gate standing between an untrusted renderer and the
local filesystem: every path a caller supplies (existing-file reads, write
targets, and id-shaped filename components) must be proven to stay inside a
managed directory before the engine touches disk. This is the highest-value
place in the engine to have a test, precisely because it has never had one.
"""

from __future__ import annotations

import pytest

from astra import config, security


def test_authorized_path_accepts_file_under_root(isolated_root):
    target = config.PATHS.root / "inside.txt"
    target.write_text("ok", encoding="utf-8")
    assert security.authorized_path(target) == target.resolve()


def test_authorized_path_rejects_file_outside_root(isolated_root, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    with pytest.raises(PermissionError):
        security.authorized_path(outside)


def test_authorized_path_rejects_traversal_that_escapes_root(isolated_root):
    escaped = config.PATHS.root / ".." / "escaped.txt"
    escaped.resolve().write_text("nope", encoding="utf-8")
    with pytest.raises(PermissionError):
        security.authorized_path(escaped)


def test_authorized_path_rejects_missing_file(isolated_root):
    missing = config.PATHS.root / "does-not-exist.txt"
    with pytest.raises(FileNotFoundError):
        security.authorized_path(missing)


def test_authorized_path_rejects_directory(isolated_root):
    directory = config.PATHS.root / "Datasets"
    with pytest.raises(FileNotFoundError):
        security.authorized_path(directory)


def test_authorized_write_path_accepts_new_file_under_root(isolated_root):
    target = config.PATHS.root / "checkpoints" / "run.json"
    resolved = security.authorized_write_path(target, config.PATHS.root)
    assert resolved == target.resolve()
    assert not resolved.exists()  # write targets need not pre-exist


def test_authorized_write_path_rejects_path_outside_allowed_root(isolated_root, tmp_path):
    outside = tmp_path.parent / "checkpoint.json"
    with pytest.raises(PermissionError):
        security.authorized_write_path(outside, config.PATHS.root)


def test_authorized_write_path_rejects_traversal(isolated_root):
    escaped = config.PATHS.root / ".." / "checkpoint.json"
    with pytest.raises(PermissionError):
        security.authorized_write_path(escaped, config.PATHS.root)


def test_scoped_id_path_builds_expected_filename(tmp_path):
    directory = tmp_path / "manifests"
    directory.mkdir()
    path = security.scoped_id_path(directory, "cone_180.0_22.0_30.0_20260901T000000Z")
    assert path == directory / "cone_180.0_22.0_30.0_20260901T000000Z.json"


def test_scoped_id_path_rejects_traversal_in_identifier(tmp_path):
    directory = tmp_path / "manifests"
    directory.mkdir()
    with pytest.raises(ValueError):
        security.scoped_id_path(directory, "../../escaped")


def test_scoped_id_path_rejects_absolute_path_identifier(tmp_path):
    directory = tmp_path / "manifests"
    directory.mkdir()
    # An absolute path as the "identifier" would otherwise be joined and
    # silently win over `directory` (Path(".") / "/etc/passwd" == "/etc/passwd"
    # on POSIX); the containment check must still catch it.
    with pytest.raises(ValueError):
        security.scoped_id_path(directory, str(tmp_path / "outside" / "escaped"))


def test_scoped_id_path_allows_dots_and_underscores(tmp_path):
    directory = tmp_path / "manifests"
    directory.mkdir()
    # Real dataset ids (see acquire.default_dataset_id) contain dots and
    # underscores, not just the restricted charset project ids use.
    path = security.scoped_id_path(directory, "cone_12.345_-6.789_30.000_20260901T010203Z")
    assert path.parent == directory.resolve()
