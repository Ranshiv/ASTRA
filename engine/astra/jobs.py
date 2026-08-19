"""Persistent, resumable background jobs for archive and research work.

The engine process is intentionally short-lived (it is supervised by Tauri),
so an in-memory Future is not enough: requests, progress, checkpoints, and
cancellation state are persisted in the metadata index.  A new engine can
recover queued/running work without duplicating a completed request.
"""

from __future__ import annotations

import inspect
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from . import config, metadata

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="astra-job")
_FUTURES: dict[str, Future] = {}
_CANCEL_EVENTS: dict[str, threading.Event] = {}
_HANDLERS: dict[str, Callable[..., Any]] = {}
_LOCK = threading.RLock()

TERMINAL = {"completed", "failed", "cancelled"}


class JobCancelled(RuntimeError):
    """Raised internally when a cooperative job observes cancellation."""


@dataclass
class JobContext:
    job_id: str
    method: str
    project_id: str | None
    _cancel_event: threading.Event

    @property
    def root(self):
        return config.PATHS.projects

    def cancelled(self) -> bool:
        if self._cancel_event.is_set():
            return True
        record = metadata.get_job(self.root, self.job_id)
        return bool(record and record.get("cancel_requested"))

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise JobCancelled("job cancelled by user")

    def update(self, *, fraction: float | None = None, phase: str | None = None,
               message: str | None = None, items_done: int | None = None,
               items_total: int | None = None, bytes_downloaded: int | None = None,
               bytes_total: int | None = None) -> None:
        """Publish a monotonic-ish progress snapshot without changing params."""
        record = metadata.get_job(self.root, self.job_id)
        if record is None:
            return
        previous = record.get("progress") or {}
        progress = dict(previous)
        if fraction is not None:
            progress["fraction"] = max(0.0, min(1.0, float(fraction)))
        if phase is not None:
            progress["phase"] = str(phase)
        if message is not None:
            progress["message"] = str(message)
        if items_done is not None:
            progress["items_done"] = max(0, int(items_done))
        if items_total is not None:
            progress["items_total"] = max(0, int(items_total))
        if bytes_downloaded is not None:
            progress["bytes_downloaded"] = max(0, int(bytes_downloaded))
        if bytes_total is not None:
            progress["bytes_total"] = max(0, int(bytes_total))
        metadata.put_job(
            self.root, self.job_id, self.method, "running",
            progress=progress, retry_count=int(record.get("retry_count", 0)),
            byte_count=int(progress.get("bytes_downloaded", record.get("byte_count", 0))),
            project_id=self.project_id, params=record.get("params", {}),
            checkpoint=record.get("checkpoint"),
            idempotency_key=record.get("idempotency_key"),
        )

    def checkpoint(self, value: object) -> None:
        record = metadata.get_job(self.root, self.job_id)
        if record is None:
            return
        metadata.put_job(
            self.root, self.job_id, self.method, "running",
            progress=record.get("progress"), checkpoint=value,
            retry_count=int(record.get("retry_count", 0)),
            byte_count=int(record.get("byte_count", 0)),
            project_id=self.project_id, params=record.get("params", {}),
            idempotency_key=record.get("idempotency_key"),
        )


def _call_handler(handler: Callable[..., Any], params: dict[str, Any], context: JobContext) -> Any:
    """Support old one-argument handlers and new progress-aware handlers."""
    try:
        signature = inspect.signature(handler)
        parameters = signature.parameters
        if "progress" in parameters:
            return handler(params, progress=context)
        if "context" in parameters:
            return handler(params, context=context)
    except (TypeError, ValueError):
        # Some extension callables do not expose a Python signature. Keep the
        # original one-argument contract for those handlers.
        pass
    return handler(params)


def _start(job_id: str, method: str, params: dict[str, Any],
           handler: Callable[..., Any], *, project_id: str | None = None,
           retry_count: int = 0, idempotency_key: str | None = None) -> None:
    event = _CANCEL_EVENTS.setdefault(job_id, threading.Event())
    _HANDLERS[job_id] = handler
    root = config.PATHS.projects
    metadata.clear_job_cancel(root, job_id)
    metadata.put_job(
        root, job_id, method, "queued", params=params, project_id=project_id,
        progress={"fraction": 0.0, "phase": "queued", "items_done": 0},
        retry_count=retry_count, idempotency_key=idempotency_key,
    )

    def run() -> None:
        context = JobContext(job_id, method, project_id, event)
        metadata.put_job(
            root, job_id, method, "running", params=params, project_id=project_id,
            progress={"fraction": 0.0, "phase": "starting", "items_done": 0},
            retry_count=retry_count, idempotency_key=idempotency_key,
        )
        try:
            result = _call_handler(handler, params, context)
            current = metadata.get_job(root, job_id) or {}
            if context.cancelled():
                metadata.put_job(
                    root, job_id, method, "cancelled", params=params,
                    project_id=project_id, progress={"fraction": 0.0, "phase": "cancelled"},
                    checkpoint=current.get("checkpoint"), byte_count=current.get("byte_count", 0),
                    retry_count=retry_count, idempotency_key=idempotency_key,
                    cancel_requested=True,
                )
            else:
                metadata.put_job(
                    root, job_id, method, "completed", result=result, params=params,
                    project_id=project_id,
                    progress={"fraction": 1.0, "phase": "completed", "items_done": 1},
                    checkpoint=current.get("checkpoint"), byte_count=current.get("byte_count", 0),
                    retry_count=retry_count, idempotency_key=idempotency_key,
                    cancel_requested=False,
                )
        except JobCancelled as exc:
            current = metadata.get_job(root, job_id) or {}
            metadata.put_job(
                root, job_id, method, "cancelled", error=str(exc), params=params,
                project_id=project_id, progress={"fraction": 0.0, "phase": "cancelled"},
                checkpoint=current.get("checkpoint"), byte_count=current.get("byte_count", 0),
                retry_count=retry_count, idempotency_key=idempotency_key,
                cancel_requested=True,
            )
        except Exception as exc:  # noqa: BLE001 - status is the public API
            current = metadata.get_job(root, job_id) or {}
            metadata.put_job(
                root, job_id, method, "failed", error=str(exc), params=params,
                project_id=project_id, error_kind=type(exc).__name__,
                progress={"phase": "failed"}, checkpoint=current.get("checkpoint"),
                byte_count=current.get("byte_count", 0), retry_count=retry_count,
                idempotency_key=idempotency_key, cancel_requested=False,
            )
            # Keep the traceback in the process log, never in the UI payload.
            traceback.print_exc()
        finally:
            with _LOCK:
                _FUTURES.pop(job_id, None)

    with _LOCK:
        _FUTURES[job_id] = _EXECUTOR.submit(run)


def submit(method: str, params: dict[str, Any], handler: Callable[..., Any], *,
           project_id: str | None = None,
           idempotency_key: str | None = None) -> dict:
    root = config.PATHS.projects
    project_id = project_id or params.get("project_id")
    idempotency_key = idempotency_key or params.get("idempotency_key")
    if idempotency_key:
        existing = metadata.find_job_by_idempotency(root, str(idempotency_key))
        if existing and existing["status"] not in {"failed", "cancelled"}:
            return {"job_id": existing["job_id"], "status": existing["status"],
                    "method": existing["method"], "existing": True}

    job_id = f"job_{uuid.uuid4().hex}"
    metadata.put_job(
        root, job_id, method, "queued", params=params, project_id=project_id,
        progress={"fraction": 0.0, "phase": "queued", "items_done": 0},
        idempotency_key=str(idempotency_key) if idempotency_key else None,
    )
    _start(job_id, method, params, handler, project_id=project_id,
           idempotency_key=str(idempotency_key) if idempotency_key else None)
    return {"job_id": job_id, "status": "queued", "method": method}


def status(job_id: str) -> dict:
    record = metadata.get_job(config.PATHS.projects, job_id)
    if record is None:
        raise KeyError(f"job not found: {job_id}")
    return record


def list_all(statuses: tuple[str, ...] | None = None) -> list[dict]:
    return metadata.list_jobs(config.PATHS.projects, statuses=statuses)


def cancel(job_id: str) -> dict:
    root = config.PATHS.projects
    record = status(job_id)
    if record["status"] in TERMINAL:
        return record
    event = _CANCEL_EVENTS.setdefault(job_id, threading.Event())
    event.set()
    metadata.request_job_cancel(root, job_id)
    future = _FUTURES.get(job_id)
    if future is not None and future.cancel():
        metadata.put_job(root, job_id, record["method"], "cancelled",
                         params=record.get("params", {}), project_id=record.get("project_id"),
                         retry_count=record.get("retry_count", 0),
                         idempotency_key=record.get("idempotency_key"),
                         progress={"fraction": 0.0, "phase": "cancelled"})
    return status(job_id)


def retry(job_id: str, handlers: dict[str, Callable[..., Any]] | None = None) -> dict:
    root = config.PATHS.projects
    record = status(job_id)
    if record["status"] not in {"failed", "cancelled", "paused", "partial"}:
        return record
    handler = _HANDLERS.get(job_id) or (handlers or {}).get(record["method"])
    if handler is None:
        raise KeyError(f"no handler registered for {record['method']}")
    _CANCEL_EVENTS[job_id] = threading.Event()
    _start(job_id, record["method"], record.get("params", {}), handler,
           project_id=record.get("project_id"),
           retry_count=int(record.get("retry_count", 0)) + 1,
           idempotency_key=record.get("idempotency_key"))
    # ``ThreadPoolExecutor.submit`` may start the worker before this function
    # returns.  The public retry contract is the same as ``submit``: the
    # caller receives an acknowledgement that the retry was queued, then
    # observes the authoritative running/completed state through ``status``.
    # Returning the transient ``running`` state here made the API racy and
    # forced clients to special-case an otherwise valid retry acknowledgement.
    acknowledged = status(job_id)
    acknowledged["status"] = "queued"
    return acknowledged


def recover(handlers: dict[str, Callable[..., Any]]) -> list[str]:
    """Resume jobs left by an engine restart and return their IDs."""
    resumed: list[str] = []
    root = config.PATHS.projects
    for record in metadata.list_jobs(root, statuses=("queued", "running", "paused")):
        method = record["method"]
        handler = handlers.get(method)
        if handler is None:
            continue
        if record["status"] == "running":
            metadata.put_job(
                root, record["job_id"], method, "paused", error="engine restarted; job resumed",
                params=record.get("params", {}), project_id=record.get("project_id"),
                progress=record.get("progress"), checkpoint=record.get("checkpoint"),
                retry_count=record.get("retry_count", 0), byte_count=record.get("byte_count", 0),
                idempotency_key=record.get("idempotency_key"),
            )
        _CANCEL_EVENTS[record["job_id"]] = threading.Event()
        _start(record["job_id"], method, record.get("params", {}), handler,
               project_id=record.get("project_id"),
               retry_count=record.get("retry_count", 0),
               idempotency_key=record.get("idempotency_key"))
        resumed.append(record["job_id"])
    return resumed
