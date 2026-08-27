"""Leakage-safe splits for benchmark evaluation.

`tensors.train_test_split` splits by *row index* with a random permutation
-- correct for a training loop over a batch of already-independent
sequences, but wrong for a benchmark: if an object contributes more than
one row (multiple bands, multiple epochs cut into windows), a random row
split can place the same object's rows in both train and test, and the
"held-out" metric partly measures memorization of that object rather than
generalization. `tensors.train_test_split` is left unchanged and keeps
being used for unit tests and non-benchmark training; benchmark paths use
the object-grouped splits here instead.

A `Split` is stored as a JSON file naming which object IDs (or field/season
IDs, for a sky/time split) fall in each fold. It does not store the row
data itself, matching `manifest.Manifest`'s "manifest, not a snapshot"
principle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

SPLIT_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Split:
    split_id: str
    kind: str  # "object_grouped" | "sky_time"
    grouping_key: str
    seed: int
    folds: dict[str, list[str]]  # fold name -> sorted group IDs
    created_utc: str = field(default_factory=_utc_now)
    schema_version: int = SPLIT_SCHEMA_VERSION

    def fold_counts(self) -> dict[str, int]:
        return {name: len(ids) for name, ids in self.folds.items()}

    def content_hash(self) -> str:
        payload = {"kind": self.kind, "grouping_key": self.grouping_key,
                  "seed": self.seed, "folds": self.folds}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


def object_grouped_split(
    object_ids: Sequence[str], *, split_id: str, fractions: dict[str, float] | None = None,
    seed: int = 42,
) -> Split:
    """Assign each distinct object ID to exactly one fold.

    `fractions` defaults to a 60/20/20 train/val/test split. Object IDs are
    deduplicated and sorted before shuffling, so the same input always
    produces the same folds for a given seed regardless of input order.
    """
    fractions = fractions or {"train": 0.6, "val": 0.2, "test": 0.2}
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    unique_ids = sorted(set(object_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_ids))
    shuffled = [unique_ids[i] for i in order]

    folds: dict[str, list[str]] = {}
    cursor = 0
    fold_names = list(fractions)
    for i, name in enumerate(fold_names):
        if i == len(fold_names) - 1:
            chunk = shuffled[cursor:]
        else:
            size = int(round(len(shuffled) * fractions[name]))
            chunk = shuffled[cursor:cursor + size]
            cursor += size
        folds[name] = sorted(chunk)

    return Split(split_id=split_id, kind="object_grouped", grouping_key="object_id",
                seed=seed, folds=folds)


def sky_time_split(
    records: Sequence[dict], *, split_id: str, healpix_nside: int = 8,
    fractions: dict[str, float] | None = None, seed: int = 42,
) -> Split:
    """Group by (sky cell, observing season) so a field or season cannot
    straddle folds even when individual objects within it could otherwise
    pass an object-grouped split.

    `records` is a sequence of `{"object_id", "ra_deg", "dec_deg", "mjd"}`.
    The sky cell uses a coarse HEALPix-like binning (ring index by
    declination band, RA sector count scaled to preserve roughly equal
    area) rather than requiring `astropy_healpix` here, since this module
    must stay import-light; callers needing exact HEALPix indices should
    precompute them and pass a `"cell"` key instead of `ra_deg`/`dec_deg`.
    """
    fractions = fractions or {"train": 0.6, "val": 0.2, "test": 0.2}
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    group_of: dict[str, str] = {}
    for record in records:
        object_id = str(record["object_id"])
        if "cell" in record:
            cell = str(record["cell"])
        else:
            dec = float(record["dec_deg"])
            ra = float(record["ra_deg"])
            dec_band = int((dec + 90.0) / 180.0 * healpix_nside)
            ra_sector = int(ra / 360.0 * healpix_nside * 2)
            cell = f"cell_{dec_band}_{ra_sector}"
        season = int(float(record["mjd"]) // 90)  # ~quarterly season blocks
        group_of[object_id] = f"{cell}:season_{season}"

    groups = sorted(set(group_of.values()))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    shuffled = [groups[i] for i in order]

    group_folds: dict[str, str] = {}
    cursor = 0
    fold_names = list(fractions)
    for i, name in enumerate(fold_names):
        if i == len(fold_names) - 1:
            chunk = shuffled[cursor:]
        else:
            size = int(round(len(shuffled) * fractions[name]))
            chunk = shuffled[cursor:cursor + size]
            cursor += size
        for group in chunk:
            group_folds[group] = name

    folds: dict[str, list[str]] = {name: [] for name in fold_names}
    for object_id, group in group_of.items():
        folds[group_folds[group]].append(object_id)
    for name in folds:
        folds[name].sort()

    return Split(split_id=split_id, kind="sky_time", grouping_key="cell:season",
                seed=seed, folds=folds)


def detect_leakage(split: Split) -> dict:
    """Report any object ID appearing in more than one fold.

    Backs acceptance gate 3 ("no train/validation/test object or field
    overlap is detected"). A clean split returns `{"leaked": [], "clean": True}`.
    """
    seen: dict[str, list[str]] = {}
    for fold_name, ids in split.folds.items():
        for object_id in ids:
            seen.setdefault(object_id, []).append(fold_name)

    leaked = [{"object_id": object_id, "folds": folds}
              for object_id, folds in sorted(seen.items()) if len(folds) > 1]
    return {"leaked": leaked, "clean": not leaked, "n_leaked": len(leaked)}


def save(split: Split, root: Path) -> Path:
    path = root / "splits" / f"{split.split_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    return path


def load(split_id: str, root: Path) -> Split:
    path = root / "splits" / f"{split_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Split(**payload)


__all__ = [
    "Split", "object_grouped_split", "sky_time_split", "detect_leakage",
    "save", "load",
]
