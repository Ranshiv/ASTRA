"""Durable job lifecycle tests."""

from __future__ import annotations

import time

from astra import jobs, metadata


def _wait(job_id: str, *statuses: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = jobs.status(job_id)
        if record["status"] in statuses:
            return record
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {statuses}: {jobs.status(job_id)}")


def test_job_persists_progress_and_result(isolated_root):
    def work(params, progress):
        progress.update(phase="fetch", message="fixture", fraction=0.5,
                        items_done=1, items_total=2, bytes_downloaded=12)
        return {"answer": params["answer"]}

    submitted = jobs.submit("fixture.work", {"answer": 42}, work,
                            idempotency_key="fixture-42")
    record = _wait(submitted["job_id"], "completed")

    assert record["result"] == {"answer": 42}
    assert record["progress"]["fraction"] == 1.0
    assert record["retry_count"] == 0
    assert record["byte_count"] == 12
    assert metadata.find_job_by_idempotency(isolated_root.projects, "fixture-42")["job_id"] == submitted["job_id"]

    duplicate = jobs.submit("fixture.work", {"answer": 42}, work,
                            idempotency_key="fixture-42")
    assert duplicate["job_id"] == submitted["job_id"]
    assert duplicate["existing"] is True


def test_job_cancellation_is_cooperative(isolated_root):
    def work(_params, progress):
        for index in range(100):
            progress.raise_if_cancelled()
            progress.update(phase="loop", fraction=index / 100, items_done=index, items_total=100)
            time.sleep(0.01)
        return {"finished": True}

    submitted = jobs.submit("fixture.cancel", {}, work)
    _wait(submitted["job_id"], "running")
    jobs.cancel(submitted["job_id"])
    record = _wait(submitted["job_id"], "cancelled")
    assert record["cancel_requested"] is True


def test_recover_resumes_persisted_queued_job(isolated_root):
    metadata.put_job(
        isolated_root.projects, "job_recover", "fixture.recover", "queued",
        params={"value": "ok"}, progress={"fraction": 0.2},
    )

    def work(params):
        return params

    assert jobs.recover({"fixture.recover": work}) == ["job_recover"]
    assert _wait("job_recover", "completed")["result"] == {"value": "ok"}


def test_job_failure_can_be_retried(isolated_root):
    attempts = {"count": 0}

    def work(_params):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary")
        return {"ok": True}

    submitted = jobs.submit("fixture.retry", {}, work)
    assert _wait(submitted["job_id"], "failed")["error_kind"] == "RuntimeError"
    retried = jobs.retry(submitted["job_id"], {"fixture.retry": work})
    assert retried["status"] == "queued"
    record = _wait(submitted["job_id"], "completed")
    assert record["result"] == {"ok": True}
    assert record["retry_count"] == 1
