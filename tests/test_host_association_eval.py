"""Synthetic-population generation and the top-k recall / calibration
metric arithmetic for `host_association_eval.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import host_association_eval as hae


# ---------------------------------------------------------------------------
# synthesize_host_population
# ---------------------------------------------------------------------------

def test_synthesize_host_population_produces_a_true_host_and_contaminants():
    rng = np.random.default_rng(1)
    field = hae.synthesize_host_population(rng, n_contaminants=3)
    assert field.true_host_id == "host_true"
    assert len(field.candidates) == 4
    assert field.true_host_id in field.redshifts
    assert field.true_host_id in field.r_e_arcsec
    assert "contaminant_0" in field.foreground_flags  # default veto flag


def test_synthesize_host_population_rejects_bad_inputs():
    rng = np.random.default_rng(1)
    with pytest.raises(hae.HostAssociationError):
        hae.synthesize_host_population(rng, true_r_e_arcsec=0.0)
    with pytest.raises(hae.HostAssociationError):
        hae.synthesize_host_population(rng, n_contaminants=-1)


# ---------------------------------------------------------------------------
# evaluate_top_k_recall
# ---------------------------------------------------------------------------

def test_evaluate_top_k_recall_is_high_for_a_well_separated_population():
    # A true host offset scale much smaller than the contaminant offset
    # scale, and a much brighter fiducial magnitude, should be an easy
    # recovery case.
    result = hae.evaluate_top_k_recall(
        n_trials=40, k_values=(1, 2, 3), seed=7,
        true_r_e_arcsec=1.0, contaminant_offset_scale_arcsec=15.0, n_contaminants=3)
    assert result.recall[1] >= 0.5
    assert result.recall[3] >= result.recall[1]  # recall is non-decreasing in k


def test_evaluate_top_k_recall_rejects_bad_inputs():
    with pytest.raises(hae.HostAssociationError):
        hae.evaluate_top_k_recall(n_trials=0)
    with pytest.raises(hae.HostAssociationError):
        hae.evaluate_top_k_recall(k_values=())


def test_top_k_recall_result_to_dict_shape():
    result = hae.TopKRecallResult(k_values=(1, 2), n_trials=10, recall={1: 0.5, 2: 0.8})
    payload = result.to_dict()
    assert payload["recall"]["1"] == 0.5
    assert payload["n_trials"] == 10


# ---------------------------------------------------------------------------
# evaluate_probability_calibration
# ---------------------------------------------------------------------------

def test_evaluate_probability_calibration_reports_a_finite_ece_and_full_bin_coverage():
    result = hae.evaluate_probability_calibration(
        n_trials=40, n_bins=5, seed=11, n_contaminants=3)
    assert 0.0 <= result.expected_calibration_error <= 1.0
    assert sum(result.bin_counts) == result.n_candidates
    assert result.n_candidates == 40 * 4  # 1 true host + 3 contaminants per trial


def test_evaluate_probability_calibration_rejects_bad_inputs():
    with pytest.raises(hae.HostAssociationError):
        hae.evaluate_probability_calibration(n_trials=0)
    with pytest.raises(hae.HostAssociationError):
        hae.evaluate_probability_calibration(n_bins=0)


def test_calibration_result_to_dict_shape():
    result = hae.CalibrationResult(
        n_bins=2, n_trials=5, n_candidates=10, bin_edges=[0.0, 0.5, 1.0],
        bin_predicted=[0.1, 0.9], bin_empirical=[0.2, 0.8], bin_counts=[6, 4],
        expected_calibration_error=0.06)
    payload = result.to_dict()
    assert payload["expected_calibration_error"] == 0.06
    assert len(payload["bin_edges"]) == 3


def test_host_association_eval_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "host_association" not in rpc_source
