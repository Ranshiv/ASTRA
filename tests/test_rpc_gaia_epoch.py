"""gaia.epoch_ingest / gaia.epoch_status: RPC surface for chunked DR4 ingestion.

Exercises the RPC layer against a real fixture file and a real checkpoint on
disk (small, fast) rather than monkeypatching gaia_epoch's internals -- the
chunking/checkpoint mechanics themselves are covered in depth by
tests/test_surveys_gaia_epoch.py; these tests only need to confirm parameters
are wired through correctly and the handler shape (progress=None default,
job.submit-compatible) works.
"""

from __future__ import annotations

import json

from astra import rpc


def _write_fixture(path, chunks):
    path.write_text(json.dumps({"chunks": chunks}), encoding="utf-8")


class TestGaiaEpochIngest:
    def test_ingests_a_fixture_and_reports_counts(self, tmp_path):
        fixture = tmp_path / "fixture.json"
        _write_fixture(fixture, [
            [{"source_id": "1", "time": 2459000.0, "g_flux": 100.0, "g_flux_error": 2.0}],
            [{"source_id": "2", "time": 2459001.0, "g_flux": "bad", "g_flux_error": 2.0}],
        ])
        checkpoint = tmp_path / "checkpoint.json"

        response = rpc.dispatch({"id": 1, "method": "gaia.epoch_ingest", "params": {
            "fixture_path": str(fixture), "checkpoint": str(checkpoint), "batch_size": 10,
        }})

        assert response["ok"] is True
        result = response["result"]
        assert result["chunks_total"] == 2
        assert result["rows_accepted"] == 1
        assert result["rows_rejected"] == 1
        assert result["rejection_histogram"] == {"invalid_value": 1}
        assert checkpoint.exists()

    def test_works_with_no_progress_context_like_a_direct_dispatch_call(self, tmp_path):
        """dispatch() calls handler(params) only -- progress must default safely."""
        fixture = tmp_path / "fixture.json"
        _write_fixture(fixture, [[{"source_id": "1", "time": 1.0, "g_flux": 1.0, "g_flux_error": 1.0}]])
        checkpoint = tmp_path / "checkpoint.json"

        response = rpc.dispatch({"id": 1, "method": "gaia.epoch_ingest", "params": {
            "fixture_path": str(fixture), "checkpoint": str(checkpoint),
        }})

        assert response["ok"] is True


class TestGaiaEpochStatus:
    def test_reports_not_exists_for_a_missing_checkpoint(self, tmp_path):
        response = rpc.dispatch({"id": 1, "method": "gaia.epoch_status", "params": {
            "checkpoint": str(tmp_path / "missing.json"),
        }})

        assert response["ok"] is True
        assert response["result"] == {"exists": False}

    def test_reports_state_after_an_ingest(self, tmp_path):
        fixture = tmp_path / "fixture.json"
        _write_fixture(fixture, [
            [{"source_id": "1", "time": 2459000.0, "g_flux": 100.0, "g_flux_error": 2.0}],
        ])
        checkpoint = tmp_path / "checkpoint.json"
        rpc.dispatch({"id": 1, "method": "gaia.epoch_ingest", "params": {
            "fixture_path": str(fixture), "checkpoint": str(checkpoint),
        }})

        response = rpc.dispatch({"id": 2, "method": "gaia.epoch_status", "params": {
            "checkpoint": str(checkpoint),
        }})

        assert response["ok"] is True
        result = response["result"]
        assert result["exists"] is True
        assert result["chunks_completed"] == 1
        assert result["rows_accepted"] == 1
        assert result["rows_available"] == 1
