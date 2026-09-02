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
    # A TAP sync query is a database job, not a REST lookup; keep more room
    # between them than the default interval allows.
    "datalab": 1.0,
    "gwosc": 0.5,
    "cadc": 0.5,
    "ads": 1.0,
    "arxiv": 3.0,
    # ALeRCE is a shared community alert broker (NOIRLab), not an
    # ASTRA-dedicated allocation like MAST/IRSA — keep it out of the generic
    # 0.2s default and give it its own bucket, matching gwosc/cadc.
    "alerce": 0.5,
    # Rubin/LSST direct TAP (data.lsst.cloud/api/tap) is a sync database job
    # like datalab, and a credential-gated one at that -- same 1.0s bucket.
    "rubin": 1.0,
    # moving_objects.py: MPC's public search_orbits web service (confirmed
    # reachable with documented shared credentials, not a per-user API key).
    "mpc": 1.0,
    # SDSS SkyServer SQL search and the SAS spec-lite FITS download -- both
    # per-object product/lookup transfers, same bucket as irsa/mast/gaia.
    "sdss": 0.2,
    # OGLE EWS is a university web server with no published rate limit and
    # no API key -- deliberately the slowest bucket here, since a season
    # index plus per-event photometry is a lot of requests to a host that
    # never asked to be a data API.
    "ogle": 1.0,
    # exoplanet_archive.py: NASA Exoplanet Archive TAP sync query -- a
    # database job like datalab, same 1.0s bucket rationale.
    "exoplanetarchive": 1.0,
    # vlass.py: VizieR (CDS)'s Simple Cone Search service -- a shared,
    # widely-used community catalogue host (like ALeRCE/gwosc), not an
    # ASTRA-dedicated allocation; kept out of the generic 0.2s default.
    "vizier": 0.5,
    # dust_3d.py: cdsarc.cds.unistra.fr plain-file downloads (the same CDS
    # organisation as vizier, a different service) -- one large product
    # transfer per cube, same bucket rationale as vizier.
    "cdsarc": 0.5,
    # strong_lens_imaging.py: ps1images.stsci.edu image-cutout service --
    # a different host from mast.stsci.edu (the existing "mast" bucket),
    # so it gets its own bucket rather than sharing MAST's allocation.
    "ps1images": 0.5,
    # catalogs.py: TNS's transient-name search API -- a credential-gated bot
    # endpoint, same 1.0s bucket as the other database-job-shaped services
    # (datalab/rubin/exoplanetarchive), matching catalogs.py's own prior
    # REQUEST_INTERVAL_SECONDS["tns"] before that throttle moved here.
    "tns": 1.0,
    # surveys/asassn.py: ASAS-SN Sky Patrol's own cone-search server
    # (asassn-lb01.ifa.hawaii.edu) -- a single research group's own
    # infrastructure, not a large data-consortium archive like VizieR/MAST,
    # so it gets the same conservative 1.0s bucket as the other
    # database-job-shaped or single-institution services rather than the
    # generic 0.2s default.
    "asassn": 1.0,
    # surveys/antares.py: ANTARES's own alert-broker API (NOIRLab) -- a
    # community broker like alerce, same 0.5s bucket as that connector's
    # "alerce" entry above.
    "antares": 0.5,
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
        provider: str = "irsa", headers: dict[str, str] | None = None) -> requests.Response:
    """Throttled, retrying GET. Raises only once the retries are exhausted.

    A cassette record/replay layer (`astra.research.cassettes`) wraps this
    at request granularity when `ASTRA_CASSETTE_MODE` is set to `"record"`
    or `"replay"` -- see that module's docstring. Default is `"off"`
    everywhere, tests included, i.e. unchanged live-request behaviour; a
    caller that wants replay (e.g. a connector fixture test) sets
    `ASTRA_CASSETTE_MODE=replay` itself.
    """
    from .research import cassettes
    request_mode = cassettes.mode()
    if request_mode == "off":
        return _get_live(url, params, timeout, provider, headers)

    key = cassettes.identity(provider, "GET", url, params)
    if request_mode == "replay":
        recorded = cassettes.load(key)
        return _response_from_cassette(recorded)

    # record mode: make the real request, then persist it.
    response = _get_live(url, params, timeout, provider, headers)
    cassettes.save(key, cassettes.RecordedResponse(
        status_code=response.status_code, headers=dict(response.headers),
        content=response.content, url=response.url))
    return response


def _get_live(url: str, params: dict, timeout: float, provider: str,
              headers: dict[str, str] | None) -> requests.Response:
    throttle(provider)
    kwargs = {"params": params, "timeout": timeout}
    if headers:
        kwargs["headers"] = headers
    response = session().get(url, **kwargs)
    response.raise_for_status()
    return response


def _response_from_cassette(recorded) -> requests.Response:
    """Build a real `requests.Response` from a recorded cassette so callers
    downstream (`.json()`, `.text`, `.content`, `.status_code`) see no
    difference from a live call."""
    response = requests.Response()
    response.status_code = recorded.status_code
    response.headers.update(recorded.headers)
    response._content = recorded.content
    response.url = recorded.url
    response.raise_for_status()
    return response


def post(url: str, data: dict | None = None, timeout: float = 60.0,
         provider: str = "irsa", headers: dict[str, str] | None = None,
         json: dict | None = None) -> requests.Response:
    """Throttled POST, for the few contracts (e.g. TAP async job submission
    in `tap.py`, or ASAS-SN's cone-search endpoint in `surveys/asassn.py`)
    that require one. Unlike `get`, this is NOT auto-retried by the shared
    session's `Retry` policy (`allowed_methods` there is deliberately
    `{"GET", "HEAD"}` only) -- retrying a POST that creates a resource, like
    a TAP async job, risks silently double-submitting it, so that judgment
    call is left to the caller rather than attempted here.
    `requests` follows a redirect response (e.g. a TAP job's `303 See
    Other` to its own status URL) by default; the caller reads the
    resulting `response.url`/`response.text` rather than a raw `Location`
    header.

    `data` form-encodes (the original, still-default contract every
    existing caller uses); `json` JSON-encodes the body and sets the
    matching content type instead, for a contract (like ASAS-SN's) that
    requires an actual JSON body rather than form fields. Passing both is a
    caller error `requests` itself would reject; callers pass exactly one.
    """
    throttle(provider)
    kwargs: dict = {"timeout": timeout}
    if json is not None:
        kwargs["json"] = json
    else:
        kwargs["data"] = data
    if headers:
        kwargs["headers"] = headers
    response = session().post(url, **kwargs)
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

    A cassette record/replay layer (`astra.research.cassettes`) wraps this
    the same way `get()` is wrapped -- see that function's docstring. Before
    this, only `get`'s metadata/query-shaped requests were cassette-
    verifiable; a connector whose evidence depends on a downloaded product
    (e.g. a TESS TPF) had no offline-replayable fixture at all (see
    docs/LIMITATIONS.md's "Cassette layer covers netclient.get, not
    download" gap). Default mode is `"off"`, i.e. today's unmodified
    live-streaming behaviour.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(str(target))

    from .research import cassettes
    request_mode = cassettes.mode()
    cassette_key = (cassettes.identity(provider, "DOWNLOAD", url, params)
                    if request_mode != "off" else None)
    if request_mode == "replay":
        recorded = cassettes.load(cassette_key)
        return _download_result_from_cassette(recorded, target, max_bytes)

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

    result = DownloadResult(path=target, bytes_written=received,
                            sha256=digest.hexdigest(), content_length=declared)

    if request_mode == "record":
        # Read the file back rather than buffering the whole transfer in
        # memory during the stream above -- it already passed the same
        # `max_bytes` check and atomic-publish path a replay will reuse.
        cassettes.save(cassette_key, cassettes.RecordedResponse(
            status_code=response.status_code, headers=dict(response.headers),
            content=target.read_bytes(), url=response.url))

    return result


def _download_result_from_cassette(recorded, target: Path, max_bytes: int) -> DownloadResult:
    """Replay a recorded download: write the cassette's checksummed body to
    `target` through the same atomic temp-file-then-replace path the live
    download uses, so a consumer never sees a partial file either way."""
    content = recorded.content
    if len(content) > max_bytes:
        raise DownloadTooLargeError(len(content), max_bytes)

    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    return DownloadResult(path=target, bytes_written=len(content),
                          sha256=hashlib.sha256(content).hexdigest(),
                          content_length=len(content))


def reset() -> None:
    """Drop the shared session and throttle state — for tests."""
    global _session
    with _lock:
        if _session is not None:
            _session.close()
        _session = None
        _last_request_at.clear()
