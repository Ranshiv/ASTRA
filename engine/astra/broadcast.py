"""A local, shareable feed of ASTRA's own high-confidence candidates.

ASTRA is a local desktop app (Tauri) with no user accounts, hosting, or
existing outbound network infrastructure -- there is no watch/serve
mechanism anywhere in this codebase (Rust or Python), and picking/
authenticating against a real third-party publish target (a hosted API, a
webhook service) is a much bigger commitment than "let a researcher share
results outside the app" actually requires. This module writes a FILE, not
a network push: a stable-path (not timestamped, unlike `exports.py`'s
snapshot exports -- a feed is one resource that updates in place) JSON
document a researcher can point external tools at or send manually. UI
copy calls this a "local feed file", never "publish"/"broadcast to the
internet", so it never implies network behaviour that does not exist.

The `>= threshold` cutoff mirrors `review.py`'s own precedent (`prediction
= scores >= 0.5`) for a fixed, out-of-band, not-tuned-on-this-data
threshold. The envelope shape (`schema_version` + flat array) mirrors
`alerts.py`'s inbound packet envelope, the one other alert-shaped format
already in this codebase, for consistency rather than inventing a third.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import candidates as candidates_mod, config

SCHEMA_VERSION = 1
DEFAULT_THRESHOLD = 0.5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_feed(name: str = "default", threshold: float = DEFAULT_THRESHOLD,
                  root: Path | None = None) -> dict:
    """Write a local feed file of candidates scoring at/above ``threshold``.

    Overwrites the same stable path on every call -- a feed is one resource
    that updates, not a growing pile of timestamped snapshots. Returns
    ``{"path", "count", "threshold"}``.
    """
    root = root or config.PATHS.projects
    built = candidates_mod.load(name, root)
    generated_utc = _now()

    qualifying = [item for item in built if item.score.get("total", 0.0) >= threshold]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "threshold": threshold,
        "count": len(qualifying),
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "survey": item.survey,
                "object_id": item.object_id,
                "band": item.band,
                "ra_deg": item.ra_deg,
                "dec_deg": item.dec_deg,
                "score_total": item.score.get("total"),
                "artifact_likelihood": item.artifact.get("likelihood"),
                "published_utc": generated_utc,
            }
            for item in qualifying
        ],
    }

    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}_feed.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {"path": str(path), "count": len(qualifying), "threshold": threshold}


__all__ = ["generate_feed", "SCHEMA_VERSION", "DEFAULT_THRESHOLD"]
