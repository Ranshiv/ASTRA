"""Chunked, checkpointed Gaia DR4 epoch ingestion (surveys/gaia_epoch.py).

No network anywhere in this suite -- there is no live DR4 endpoint yet (see
the module docstring). Chunks are synthetic in-memory lists; validation of
`GaiaEpochAdapter.validate_chunk` itself is covered in tests/test_connectors.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from astra import metadata
from astra.surveys import gaia, gaia_epoch


def _row(source_id, time, g_flux=100.0, g_flux_error=2.0):
    return {"source_id": source_id, "time": time, "g_flux": g_flux, "g_flux_error": g_flux_error}


class TestIngestResumable:
    def test_multi_chunk_ingest_reports_rejection_histogram_by_reason(self, tmp_path):
        chunks = [
            [_row("1", 2459000.0), _row("2", 2459000.5)],
            [{"source_id": "3", "time": 2459001.0, "g_flux_error": 2.0}],  # missing g_flux
            [_row("4", 2459002.0, g_flux_error=float("nan"))],             # non-finite
        ]
        checkpoint = tmp_path / "checkpoint.json"

        report = gaia_epoch.ingest_resumable(chunks, checkpoint=checkpoint, batch_size=10)

        assert report.chunks_total == 3
        assert report.chunks_completed == 3
        assert report.rows_accepted == 2
        assert report.rows_rejected == 2
        assert report.rejection_histogram == {"missing_column": 1, "invalid_value": 1}
        assert report.resumed is False

        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert len(state["parts"]) >= 1
        for part in state["parts"]:
            assert Path(part).exists()

    def test_batch_size_controls_how_often_a_part_is_flushed(self, tmp_path):
        chunks = [[_row(str(i), 2459000.0 + i)] for i in range(5)]
        checkpoint = tmp_path / "checkpoint.json"

        report = gaia_epoch.ingest_resumable(chunks, checkpoint=checkpoint, batch_size=2)

        assert report.rows_accepted == 5
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        # 5 rows at batch_size=2: two full parts flushed mid-run, one on completion.
        assert len(state["parts"]) == 3

    def test_batch_size_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError):
            gaia_epoch.ingest_resumable([], checkpoint=tmp_path / "c.json", batch_size=0)

    def test_resuming_skips_already_completed_chunks_without_reprocessing(self, tmp_path, monkeypatch):
        checkpoint = tmp_path / "checkpoint.json"
        calls: list[list[dict]] = []
        real_validate = gaia.GaiaEpochAdapter.validate_chunk

        def counting_validate(rows):
            calls.append(rows)
            return real_validate(rows)

        monkeypatch.setattr(gaia.GaiaEpochAdapter, "validate_chunk", counting_validate)

        good_chunks = [[_row("1", 2459000.0)], [_row("2", 2459001.0)], [_row("3", 2459002.0)]]

        def interrupting_source():
            for index, chunk in enumerate(good_chunks):
                if index == 2:
                    raise RuntimeError("simulated interruption")
                yield chunk

        with pytest.raises(RuntimeError, match="simulated interruption"):
            gaia_epoch.ingest_resumable(interrupting_source(), checkpoint=checkpoint, batch_size=10)

        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert state["completed_chunk_ids"] == [0, 1]
        assert len(calls) == 2

        report = gaia_epoch.ingest_resumable(good_chunks, checkpoint=checkpoint, batch_size=10)

        assert report.resumed is True
        assert report.chunks_completed == 3
        assert report.rows_accepted == 3
        # Chunks 0 and 1 were skipped by index on the resumed run, not
        # revalidated -- only chunk 2 (the one that never completed) adds a
        # third call.
        assert len(calls) == 3

    def test_a_failed_chunk_is_recorded_but_does_not_stop_the_run(self, tmp_path, monkeypatch):
        checkpoint = tmp_path / "checkpoint.json"

        def raising_validate(rows):
            raise ValueError("boom")

        chunks = [[_row("1", 2459000.0)]]
        monkeypatch.setattr(gaia.GaiaEpochAdapter, "validate_chunk", raising_validate)

        report = gaia_epoch.ingest_resumable(chunks, checkpoint=checkpoint, batch_size=10)

        assert report.chunks_failed == 1
        assert report.rows_accepted == 0


class TestReadIngestedRows:
    def test_round_trips_accepted_rows(self, tmp_path):
        chunks = [[_row("1", 2459000.0, g_flux=42.0, g_flux_error=1.5)]]
        checkpoint = tmp_path / "checkpoint.json"
        gaia_epoch.ingest_resumable(chunks, checkpoint=checkpoint, batch_size=10)

        rows = gaia_epoch.read_ingested_rows(checkpoint)

        assert len(rows) == 1
        assert rows[0]["source_id"] == "1"
        assert rows[0]["g_flux"] == pytest.approx(42.0)
        assert rows[0]["g_flux_error"] == pytest.approx(1.5)

    def test_missing_checkpoint_yields_no_rows(self, tmp_path):
        assert gaia_epoch.read_ingested_rows(tmp_path / "missing.json") == []

    def test_a_stale_schema_version_part_is_skipped(self, tmp_path):
        chunks = [[_row("1", 2459000.0)]]
        checkpoint = tmp_path / "checkpoint.json"
        gaia_epoch.ingest_resumable(chunks, checkpoint=checkpoint, batch_size=10)

        # Inject a second part written under a different schema version.
        state_root = checkpoint.parent / checkpoint.stem
        stale_part = state_root / "part-999999.parquet"
        table = pa.table({
            "source_id": pa.array(["stale"]), "time": pa.array([1.0]),
            "g_flux": pa.array([1.0]), "g_flux_error": pa.array([1.0]),
        }, metadata={b"gaia_epoch_schema_version": b"999",
                    b"gaia_epoch_schema_hash": b"not-the-real-hash"})
        pq.write_table(table, stale_part)
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        state["parts"].append(str(stale_part))
        checkpoint.write_text(json.dumps(state), encoding="utf-8")

        rows = gaia_epoch.read_ingested_rows(checkpoint)

        assert len(rows) == 1
        assert rows[0]["source_id"] == "1"


class TestCrossMatchRecall:
    def test_recall_reflects_known_dr3_coverage(self, isolated_root):
        metadata.upsert_sources(isolated_root.projects, [
            {"source_key": "Gaia/dr3/1", "survey": "Gaia", "release": "dr3",
             "object_id": "1", "ra_deg": 180.0, "dec_deg": 10.0, "extra": {}},
            {"source_key": "Gaia/dr3/2", "survey": "Gaia", "release": "dr3",
             "object_id": "2", "ra_deg": 181.0, "dec_deg": 11.0, "extra": {}},
        ])
        ingested_rows = [_row("1", 2459000.0), _row("2", 2459001.0), _row("999", 2459002.0)]

        result = gaia_epoch.cross_match_recall(ingested_rows, projects_root=isolated_root.projects)

        assert result["checked"] == 3
        assert result["matched"] == 2
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["unmatched_source_ids"] == ["999"]

    def test_empty_input_reports_nan_recall_not_a_crash(self, isolated_root):
        result = gaia_epoch.cross_match_recall([], projects_root=isolated_root.projects)
        assert result["checked"] == 0
        assert result["recall"] != result["recall"]  # NaN


class TestPositionalResidualSelfConsistency:
    def test_round_trip_residual_is_small_for_a_real_proper_motion(self, isolated_root):
        metadata.upsert_sources(isolated_root.projects, [
            {"source_key": "Gaia/dr3/1", "survey": "Gaia", "release": "dr3",
             "object_id": "1", "ra_deg": 180.0, "dec_deg": 10.0,
             "extra": {"pmra": 50.0, "pmdec": -30.0}},
        ])
        # JD for ~2020.0 and ~2022.0 (roughly two years apart).
        ingested_rows = [_row("1", 2458849.5), _row("1", 2459580.0)]

        result = gaia_epoch.positional_residual_self_consistency(
            ingested_rows, projects_root=isolated_root.projects)

        assert result["checked_sources"] == 1
        assert result["median_residual_arcsec"] is not None
        # The forward/backward cos(dec) asymmetry is a real but second-order
        # effect; it must be far smaller than the multi-arcsecond proper
        # motion itself, not merely finite.
        assert result["median_residual_arcsec"] < 0.5

    def test_a_source_with_only_one_epoch_row_is_not_checked(self, isolated_root):
        metadata.upsert_sources(isolated_root.projects, [
            {"source_key": "Gaia/dr3/1", "survey": "Gaia", "release": "dr3",
             "object_id": "1", "ra_deg": 180.0, "dec_deg": 10.0, "extra": {}},
        ])
        result = gaia_epoch.positional_residual_self_consistency(
            [_row("1", 2459000.0)], projects_root=isolated_root.projects)

        assert result["checked_sources"] == 0
        assert result["median_residual_arcsec"] is None


class TestThroughput:
    def test_rows_per_second_is_computed_and_positive(self, tmp_path):
        """No hard floor asserted -- there is no real DR4 payload to benchmark
        against yet (see the approved plan's explicit open item); this only
        proves the measurement itself works."""
        rng = np.random.default_rng(0)
        chunks = [
            [_row(str(i * 50 + j), 2459000.0 + j) for j in range(50)]
            for i in range(40)
        ]
        checkpoint = tmp_path / "checkpoint.json"

        report = gaia_epoch.ingest_resumable(chunks, checkpoint=checkpoint, batch_size=256)

        assert report.rows_accepted == 2000
        assert report.rows_per_second > 0
        assert np.isfinite(report.rows_per_second)
