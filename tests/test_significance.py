"""Calibration and selection-function diagnostics."""

from __future__ import annotations

from astra import significance


def test_calibration_reports_reference_fdr_without_changing_scores():
    report = significance.calibrate(
        [0.1, 0.2, 0.9, 0.95],
        reference_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        threshold=0.8,
        strata={"survey": "ztf", "band": "g"},
    )
    assert report["ready"] is True
    assert report["reference_kind"] == "external_reference"
    assert report["selected"] == 2
    assert 0 <= report["estimated_fdr"] <= 1
    assert report["strata"]["survey"] == "ztf"


def test_calibration_empty_population_is_explicitly_unready():
    report = significance.calibrate([])
    assert report == {
        "schema_version": significance.SCHEMA_VERSION,
        "method": "empirical_tail",
        "ready": False,
        "reason": "no finite scores",
        "n_observed": 0,
        "n_reference": 0,
    }


def test_selection_keeps_unknown_dimensions_and_reports_interval():
    report = significance.evaluate_selection(
        [
            {"amplitude": 0.1, "duration_days": 1, "magnitude": 18, "detected": True},
            {"amplitude": 0.1, "duration_days": 1, "magnitude": 18, "detected": False},
            {"detected": True},
        ],
        edges={"amplitude": [0, 1], "duration_days": [0, 2], "magnitude": [15, 20]},
    )
    assert report["injected"] == 3
    assert any(cell["bins"]["amplitude"] == "unknown" for cell in report["cells"])
    known = next(cell for cell in report["cells"] if cell["bins"]["amplitude"] != "unknown")
    assert known["completeness"] == 0.5
    assert len(known["ci95"]) == 2


def test_selection_model_reports_interpretable_coefficients_and_bootstrap_interval():
    rows = [
        {"amplitude": 0.1, "duration_days": 1, "magnitude": 18, "detected": False},
        {"amplitude": 0.2, "duration_days": 2, "magnitude": 17, "detected": False},
        {"amplitude": 1.0, "duration_days": 2, "magnitude": 16, "detected": True},
        {"amplitude": 1.5, "duration_days": 4, "magnitude": 15, "detected": True},
    ]
    result = significance.evaluate_selection(
        rows, dimensions=("amplitude", "duration_days", "magnitude"),
        fit_model=True, bootstrap_samples=5, seed=7,
    )
    assert result["model"]["ready"]
    assert set(result["model"]["coefficients"]) == {"amplitude", "duration_days", "magnitude"}
    assert result["model"]["bootstrap"]["samples"] == 5
