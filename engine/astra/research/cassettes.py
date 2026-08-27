"""Content-addressed HTTP record/replay for connector evidence.

Before this module, every connector test monkeypatched `astra.netclient.get`
directly (confirmed by reading `tests/conftest.py` and the `test_surveys_*`
suite), so no connector's *actual* request/response shape was ever exercised
against a fixture -- only against a hand-written stand-in. That leaves every
"this connector works" claim undemonstrated: default CI never speaks HTTP,
by design (`tests/conftest.py`'s `live` marker), but nothing captured what a
real response looked like either.

Cassettes close that gap without turning CI into a network client. A
cassette is a single recorded (request identity -> response) entry, keyed
the same way `tap.py:_identity` keys its offline cache: provider, method,
URL, and sorted parameters, hashed to a content-addressed filename. Three
modes:

- `"record"` -- make the real request (via `netclient`'s existing session,
  retry and throttle policy, unchanged), then write the cassette.
- `"replay"` (default under pytest) -- never touch the network; raise
  `CassetteMissError` if no cassette matches.
- `"off"` -- bypass entirely, i.e. today's behaviour.

Credentials are redacted from recorded headers/params before writing, since
a cassette is meant to be committed to `research/fixtures/cassettes/`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REDACT_KEYS = re.compile(r"(key|token|secret|password|auth|credential|cookie)", re.IGNORECASE)
_MODE_ENV = "ASTRA_CASSETTE_MODE"


class CassetteMissError(RuntimeError):
    """Replay mode requested a request with no recorded cassette."""


class CassetteChecksumError(RuntimeError):
    """A cassette file's stored checksum did not match its own content."""


def mode() -> str:
    """`"record" | "replay" | "off"`, defaulting to `"off"` -- a normal
    interactive run behaves exactly as before this module existed.
    `tests/conftest.py` sets `ASTRA_CASSETTE_MODE=replay` for the pytest
    process, so the default *test* behaviour is replay-only, never live."""
    return os.environ.get(_MODE_ENV, "off")


def _redact(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    return {k: ("<redacted>" if _REDACT_KEYS.search(k) else v)
            for k, v in mapping.items()}


def identity(provider: str, method: str, url: str,
            params: dict[str, Any] | None = None) -> str:
    """Content-addressed cassette key, mirroring `tap.py:_identity`."""
    payload = {"provider": provider, "method": method.upper(), "url": url,
              "params": {k: str(v) for k, v in sorted((params or {}).items())}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cassette_dir(root: Path | None = None) -> Path:
    if root is not None:
        return root
    from .store import research_root
    return research_root() / "fixtures" / "cassettes"


@dataclass
class RecordedResponse:
    status_code: int
    headers: dict[str, str]
    content: bytes
    url: str


def _cassette_path(key: str, root: Path | None = None) -> Path:
    return _cassette_dir(root) / f"{key}.json"


def load(key: str, root: Path | None = None) -> RecordedResponse:
    path = _cassette_path(key, root)
    if not path.exists():
        raise CassetteMissError(f"no cassette recorded for request {key}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = bytes.fromhex(payload["content_hex"])
    if hashlib.sha256(body).hexdigest() != payload["checksum"]:
        raise CassetteChecksumError(f"cassette {key} failed checksum verification")
    return RecordedResponse(status_code=payload["status_code"],
                            headers=payload["headers"], content=body,
                            url=payload["url"])


def save(key: str, response: RecordedResponse, root: Path | None = None) -> Path:
    path = _cassette_path(key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status_code": response.status_code,
        "headers": _redact(dict(response.headers)),
        "url": response.url,
        "content_hex": response.content.hex(),
        "checksum": hashlib.sha256(response.content).hexdigest(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "CassetteMissError", "CassetteChecksumError", "RecordedResponse",
    "mode", "identity", "load", "save",
]
