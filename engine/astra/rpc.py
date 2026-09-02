"""JSON-lines RPC bridge between the Rust core and the Python engine.

Rust spawns this process and speaks one JSON object per line over stdin;
the engine answers with one JSON object per line on stdout. Line-delimited
JSON is used rather than a socket so the transport has no port to collide
with and dies automatically with the parent process.

Every response carries the request `id` so the Rust side can correlate
replies, and errors are returned as values rather than raised, so a failing
handler never kills the engine.

The ~150 request handlers themselves live in `rpc_handlers/`, grouped by
domain (candidates, projects, curves/FITS, research, ...); this module is
now only the protocol (`dispatch`/`serve`) plus the composed `HANDLERS`
table. Every domain module and this module's own imports below still
reference the same underlying `astra.*` modules directly (not through each
other), so `rpc.<module>` keeps working exactly as before for anything
(chiefly tests) that reaches into it -- e.g. `rpc.candidates_mod`,
`rpc.tess_pixels` -- since Python modules are cached singletons and both
this file and the relevant domain module import the identical object.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from . import (ablation, acquire, alerts, anomaly, artifact, association, attribution,
               broadcast, cache,
               candidates as candidates_mod,
               catalogs, config, credentials, crossmatch, evaluate, evidence,
               events, frb, gw, literature, multiband, significance, tap,
               experiment, exports, featurematrix, features, fitsio, followup, hardware,
               image_features, modalitymatrix, readiness, spectral_features,
               jobs, manifest as manifest_mod, metadata, pipeline, products,
               project as project_mod, review, ranker, security, sed, store, surveys,
               stageb, tensors, tess_pixels, timeframe, viz,
               reproducibility_bundle as bundle_mod,
               exoplanet_archive, habitability, neo_hazard, asteroseismology,
               biosignature, biosignature_fit, technosignature)
from .surveys import gaia_epoch
from .surveys.base import ConeQuery

from .rpc_handlers.common import DEEP_UNAVAILABLE, Handler, PROTOCOL_VERSION, _workspace_root
from .rpc_handlers.candidates import GAIA_PARALLAX_SNR_THRESHOLD
from .rpc_handlers import (
    candidates as _rh_candidates,
    catalog_gw_frb as _rh_catalog_gw_frb,
    core as _rh_core,
    credentials_ranker as _rh_credentials_ranker,
    crossmatch_deep as _rh_crossmatch_deep,
    curves_fits as _rh_curves_fits,
    events as _rh_events,
    experiments_research as _rh_experiments_research,
    followup_schedule as _rh_followup_schedule,
    frontier as _rh_frontier,
    projects as _rh_projects,
    tap_review as _rh_tap_review,
    ztf_tess as _rh_ztf_tess,
)

HANDLERS: dict[str, Handler] = {
    **_rh_core.HANDLERS,
    **_rh_events.HANDLERS,
    **_rh_tap_review.HANDLERS,
    **_rh_followup_schedule.HANDLERS,
    **_rh_projects.HANDLERS,
    **_rh_curves_fits.HANDLERS,
    **_rh_ztf_tess.HANDLERS,
    **_rh_experiments_research.HANDLERS,
    **_rh_candidates.HANDLERS,
    **_rh_catalog_gw_frb.HANDLERS,
    **_rh_credentials_ranker.HANDLERS,
    **_rh_crossmatch_deep.HANDLERS,
    **_rh_frontier.HANDLERS,
}


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    """Route one request to its handler, converting failures into error replies."""
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    handler = HANDLERS.get(method)
    if handler is None:
        return {"id": request_id, "ok": False,
                "error": f"unknown method: {method!r}"}

    try:
        return {"id": request_id, "ok": True, "result": handler(params)}
    except Exception as exc:  # noqa: BLE001 - a bad request must not kill the engine
        return {"id": request_id, "ok": False, "error": str(exc),
                "traceback": traceback.format_exc()}


# Read-only queries over the job registry -- jobs.py already guards its
# state with its own lock (job worker threads mutate it while the main
# thread polls it today), so these are safe to run without the dispatch
# lock below. That matters because these four are exactly what a UI polls
# to show progress on a long-running job; if they waited behind whatever
# slow synchronous handler the main loop happened to be running, the
# progress bar for an UNRELATED job would freeze until that handler
# finished -- the actual bug this dispatch model closes.
_UNLOCKED_METHODS = frozenset({"job.status", "job.cancel", "job.retry", "job.list"})

# Every other handler's thread-safety across shared state (SQLite metadata,
# on-disk caches, project files) has never been audited for concurrent
# access, so they stay mutually exclusive -- this lock reproduces today's
# serial-execution behavior for them exactly. What changes is only that the
# stdin-reading loop itself is never the thing blocked: it hands each
# request to its own thread and immediately goes back to reading, so a slow
# handler in flight no longer prevents the NEXT line (e.g. a job.status
# poll) from even being read, which serial synchronous dispatch could not
# avoid no matter how that next request was handled once read.
_DISPATCH_LOCK = threading.Lock()


def _dispatch_gated(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("method") in _UNLOCKED_METHODS:
        return dispatch(request)
    with _DISPATCH_LOCK:
        return dispatch(request)


def serve(stdin=None, stdout=None) -> None:
    """Read requests until stdin closes, which happens when Rust exits.

    The reader (Rust) can also disappear WITHOUT stdin closing first -- a dev
    hot-reload, a force-quit, or the window being killed while a response is
    in flight all close the pipe from the other end. On Windows, writing to a
    pipe whose reader is already gone raises `OSError: [Errno 22] Invalid
    argument` instead of the `BrokenPipeError` Unix would give (documented
    CPython behaviour on Windows, not specific to this file). That write used
    to be unguarded here, so it took down the whole engine process with an
    unhandled exception every time it happened. There is nothing to retry a
    failed write to a broken pipe against, so the correct response is to log
    it as an expected shutdown and exit, not let it become an unhandled crash.

    Each request is dispatched on its own thread (see `_dispatch_gated`)
    rather than inline, so this loop is never blocked reading the next line
    by whatever the previous request is doing. Responses are still written
    one at a time (`write_lock`), since two threads finishing at once must
    not interleave their bytes into a single broken line on the wire.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    logger = logging.getLogger(__name__)
    write_lock = threading.Lock()
    stopped = threading.Event()

    def _write_response(response: dict[str, Any]) -> None:
        if stopped.is_set():
            return
        try:
            with write_lock:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()  # Rust blocks on a line; buffering would deadlock it
        except OSError:
            logger.info("stdout closed while writing a response; the reader "
                       "(Rust) is gone, so this is a normal shutdown.")
            stopped.set()

    def _handle(request: dict[str, Any]) -> None:
        _write_response(_dispatch_gated(request))

    # A Tauri restart can interrupt a request after discovery but before the
    # final response. Recover it before accepting new UI commands.
    jobs.recover(HANDLERS)

    in_flight: list[threading.Thread] = []
    for line in stdin:
        if stopped.is_set():
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_response({"id": None, "ok": False, "error": f"invalid JSON: {exc}"})
            continue

        thread = threading.Thread(target=_handle, args=(request,), daemon=True)
        in_flight.append(thread)
        thread.start()
        # A finished thread doesn't need tracking any more; an unbounded
        # backlog across a long session is the only failure mode worth
        # guarding, not the common case of one still running.
        in_flight = [t for t in in_flight if t.is_alive()]

    # stdin closed (Rust exited): let whatever was already dispatched finish
    # and write its response rather than abandoning it mid-flight.
    for thread in in_flight:
        thread.join(timeout=30)
