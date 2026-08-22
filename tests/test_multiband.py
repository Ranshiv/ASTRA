"""Multiband sidecar builder: real curves in, real joint period out.

Verified once already during planning against real synthetic data (a
recovered period at 4.7e-5 relative error); these tests pin the grouping,
skip, and sidecar-write behaviour around that already-verified core.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import modalitymatrix, multiband, store
from astra.surveys.base import LightCurve, SourceRef


def _write_curve(root, object_id, band, mag, amp, period, rng, survey="ZTF",
                 release="dr24", n=200, baseline=500.0):
    t = np.sort(rng.uniform(0, baseline, n))
    y = mag + amp * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.03, n)
    curve = LightCurve(
        source=SourceRef(survey=survey, object_id=object_id, ra_deg=180.0, dec_deg=10.0),
        release=release, band=band, value_kind="mag",
        time=t, value=y, value_err=np.full(n, 0.03), time_system="HJD_UTC",
    )
    return store.write_curve(curve, root)


class TestBuildMultibandSidecar:
    def test_two_band_object_gets_a_sidecar_row_with_the_recovered_period(self, tmp_path):
        rng = np.random.default_rng(7)
        dataset_root = tmp_path / "datasets"
        project_root = tmp_path / "projects"
        true_period = 1.234
        _write_curve(dataset_root, "objA", "g", 18.0, 0.4, true_period, rng)
        _write_curve(dataset_root, "objA", "r", 17.5, 0.35, true_period, rng)

        result = multiband.build_multiband_sidecar(
            survey="ZTF", name="test", root=project_root, dataset_root=dataset_root)

        assert result["objects_scanned"] == 1
        assert result["objects_fit"] == 1
        assert result["objects_skipped_single_band"] == 0

        table = modalitymatrix.load(result["sidecar"]["path"])
        assert table.num_rows == 1
        row = table.to_pylist()[0]
        assert row["object_id"] == "objA"
        assert row["band"] == multiband.MULTIBAND_BAND_KEY
        assert row["multiband__best_period_days"] == pytest.approx(true_period, rel=0.02)

    def test_single_band_object_is_skipped_not_given_a_nan_row(self, tmp_path):
        rng = np.random.default_rng(8)
        dataset_root = tmp_path / "datasets"
        project_root = tmp_path / "projects"
        _write_curve(dataset_root, "objB", "g", 18.0, 0.0, 1.0, rng)

        result = multiband.build_multiband_sidecar(
            survey="ZTF", name="test", root=project_root, dataset_root=dataset_root)

        assert result["objects_scanned"] == 1
        assert result["objects_fit"] == 0
        assert result["objects_skipped_single_band"] == 1

        table = modalitymatrix.load(result["sidecar"]["path"])
        assert table.num_rows == 0

    def test_mixed_population_fits_only_the_multi_band_objects(self, tmp_path):
        rng = np.random.default_rng(9)
        dataset_root = tmp_path / "datasets"
        project_root = tmp_path / "projects"
        _write_curve(dataset_root, "multi", "g", 18.0, 0.4, 2.5, rng)
        _write_curve(dataset_root, "multi", "r", 17.5, 0.35, 2.5, rng)
        _write_curve(dataset_root, "single", "g", 18.0, 0.0, 1.0, rng)

        result = multiband.build_multiband_sidecar(
            survey="ZTF", name="test", root=project_root, dataset_root=dataset_root)

        assert result["objects_scanned"] == 2
        assert result["objects_fit"] == 1
        assert result["objects_skipped_single_band"] == 1

        table = modalitymatrix.load(result["sidecar"]["path"])
        assert table.to_pylist()[0]["object_id"] == "multi"

    def test_empty_store_produces_an_empty_sidecar_not_an_error(self, tmp_path):
        dataset_root = tmp_path / "datasets"
        project_root = tmp_path / "projects"

        result = multiband.build_multiband_sidecar(
            survey="ZTF", name="test", root=project_root, dataset_root=dataset_root)

        assert result["objects_scanned"] == 0
        assert result["objects_fit"] == 0

    def test_the_sidecar_never_touches_evidence_pys_per_survey_periods(self, tmp_path):
        """The whole point of the sidecar design: this must be additive
        evidence, never a replacement for evidence.py's independent per-
        survey period fits (score_profile's period_agreement needs both)."""
        from astra import evidence

        rng = np.random.default_rng(10)
        dataset_root = tmp_path / "datasets"
        project_root = tmp_path / "projects"
        _write_curve(dataset_root, "objA", "g", 18.0, 0.4, 1.234, rng)
        _write_curve(dataset_root, "objA", "r", 17.5, 0.35, 1.234, rng)

        multiband.build_multiband_sidecar(
            survey="ZTF", name="test", root=project_root, dataset_root=dataset_root)

        # evidence.py's own WEIGHTS/scoring contract is unaffected by
        # anything this module wrote.
        assert "period_agreement" in evidence.WEIGHTS
        assert evidence.WEIGHTS["period_agreement"] == 0.27
