"""Resilient HTTP for archive queries.

A campaign makes one request per object, serially, for hours. The original
code called `requests.get` directly with no session, no retry and no
throttle, then `raise_for_status()`. Under sustained load an archive answers
429 or 503, the exception propagated to the per-object handler, and the object
was dropped — so throttling produced *silent large-scale data loss* rather
than a visible failure.

Three things fix that:

* a shared `Session`, so connections are reused instead of paying a fresh TLS
  handshake per object;
* `urllib3.Retry` with exponential backoff on exactly the statuses that mean
  "come back later", honouring a `Retry-After` header when the server sends
  one;
* a minimum interval between requests to the same provider, so ASTRA is a
  well-behaved client rather than relying on being slow by accident.

The throttle follows the pattern already proven in `astra.catalogs`, but is
lock-protected: jobs run on a thread pool, and the original bare dict could be
read and written concurrently.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Statuses worth retrying: rate limiting and transient server faults. A 404 or
# 400 is a fact about the request and must not be retried.
RETRY_STATUSES = (429, 500, 502, 503, 504)

MAX_RETRIES = 4
BACKOFF_FACTOR = 1.0

# Minimum seconds between requests to one provider.
REQUEST_INTERVAL_SECONDS: dict[str, float] = {
    "irsa": 0.2,
    "mast": 0.2,
    "gaia": 0.2,
}
DEFAULT_INTERVAL_SECONDS = 0.2

# Product transfers have a hard, caller-configurable ceiling.  The archive's
# Content-Length is only a hint, so the same ceiling is enforced while bytes
# are streamed too.
DEFAULT_MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024


class DownloadTooLargeError(RuntimeError):
    """A streamed response exceeded its explicitly approved byte budget."""

    def __init__(self, received_bytes: int, max_bytes: int):
        super().__init__(
            f"download exceeds the {max_bytes / 1024**2:.1f} MiB limit "
            f"(received at least {received_bytes / 1024**2:.1f} MiB)"
        )
        self.received_bytes = received_bytes
        self.max_bytes = max_bytes


class DownloadIncompleteError(RuntimeError):
    """The server declared more bytes than it delivered."""

    def __init__(self, received_bytes: int, declared_bytes: int):
        super().__init__(
            f"download ended after {received_bytes} bytes; "
            f"server declared {declared_bytes} bytes"
        )
        self.received_bytes = received_bytes
        self.declared_bytes = declared_bytes


@dataclass(frozen=True)
class DownloadResult:
    """Integrity data for an atomically published archive product."""

    path: Path
    bytes_written: int
    sha256: str
    content_length: int | None

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "bytes": self.bytes_written,
            "sha256": self.sha256,
            "content_length": self.content_length,
        }

_lock = threading.Lock()
_last_request_at: dict[str, float] = {}
_session: requests.Session | None = None


def session() -> requests.Session:
    """The shared session, created once."""
    global _session
    with _lock:
        if _session is None:
            _session = requests.Session()
            retry = Retry(
                total=MAX_RETRIES,
                connect=MAX_RETRIES,
                read=MAX_RETRIES,
                status=MAX_RETRIES,
                backoff_factor=BACKOFF_FACTOR,
                status_forcelist=RETRY_STATUSES,
                allowed_methods=frozenset(["GET", "HEAD"]),
                respect_retry_after_header=True,
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry, pool_maxsize=8)
            _session.mount("https://", adapter)
            _session.mount("http://", adapter)
        return _session


def throttle(provider: str) -> None:
    """Space requests to one provider, safely across threads."""
    interval = REQUEST_INTERVAL_SECONDS.get(provider, DEFAULT_INTERVAL_SECONDS)
    with _lock:
        previous = _last_request_at.get(provider)
        now = time.monotonic()
        remaining = 0.0 if previous is None else interval - (now - previous)
        # Reserve the slot before releasing the lock so two threads cannot
        # both decide they may go now.
        _last_request_at[provider] = now + max(remaining, 0.0)

    if remaining > 0:
        time.sleep(remaining)


def get(url: str, params: dict, timeout: float,
        provider: str = "irsa") -> requests.Response:
    """Throttled, retrying GET. Raises only once the retries are exhausted."""
    throttle(provider)
    response = session().get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def _response_content_length(response: requests.Response) -> int | None:
    """Extract a valid Content-Length without trusting malformed headers."""
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Content-Length", headers.get("content-length"))
    try:
        size = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return size if size is None or size >= 0 else None


def _iter_response_chunks(response: requests.Response,
                          chunk_size: int) -> Iterator[bytes]:
    """Support requests responses and deliberately small test doubles."""
    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        yield from iterator(chunk_size=chunk_size)
        return
    content = getattr(response, "content", b"")
    for offset in range(0, len(content), chunk_size):
        yield content[offset:offset + chunk_size]


def download(url: str, destination: str | Path, *, params: dict | None = None,
             timeout: float = 180.0, provider: str = "irsa",
             max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
             chunk_size: int = 1024 * 1024,
             headers: dict[str, str] | None = None,
             progress: Callable[[int, int | None], None] | None = None,
             overwrite: bool = False) -> DownloadResult:
    """Stream an archive product, enforcing size and publishing atomically.

    A failed or interrupted transfer remains only as a temporary sibling file,
    which is always removed.  Consumers therefore never see a partial FITS
    file at its canonical path.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(str(target))

    kwargs = {"params": params or {}, "timeout": timeout, "stream": True}
    if headers:
        kwargs["headers"] = headers
    throttle(provider)
    response = session().get(url, **kwargs)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        response.raise_for_status()
        declared = _response_content_length(response)
        if declared is not None and declared > max_bytes:
            raise DownloadTooLargeError(declared, max_bytes)

        digest = hashlib.sha256()
        received = 0
        try:
            with temporary.open("wb") as handle:
                for chunk in _iter_response_chunks(response, chunk_size):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > max_bytes:
                        raise DownloadTooLargeError(received, max_bytes)
                    handle.write(chunk)
                    digest.update(chunk)
                    if progress is not None:
                        progress(received, declared)
                response_headers = getattr(response, "headers", {}) or {}
                encoding = str(response_headers.get("Content-Encoding", "") or "").lower()
                if declared is not None and not encoding and received != declared:
                    raise DownloadIncompleteError(received, declared)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()

    return DownloadResult(path=target, bytes_written=received,
                          sha256=digest.hexdigest(), content_length=declared)


def reset() -> None:
    """Drop the shared session and throttle state — for tests."""
    global _session
    with _lock:
        if _session is not None:
            _session.close()
        _session = None
        _last_request_at.clear()
