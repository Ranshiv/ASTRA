"""summary/bootstrap/BH/calibration correctness for research/stats.py."""

from __future__ import annotations

import numpy as np

from astra.research import stats


def test_summary_matches_known_values():
    result = stats.summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["mean"] == 3.0
    assert result["n"] == 5


def test_summary_empty_returns_none():
    assert stats.summary([]) is None
    assert stats.summary([float("nan"), float("inf")]) is None


def test_paired_bootstrap_ci_recovers_known_mean():
    rng = np.random.default_rng(0)
    groups = {f"obj{i}": [float(rng.normal(0.5, 0.01))] for i in range(200)}
    result = stats.paired_bootstrap_ci(groups, np.mean, n_resamples=500, seed=1)
    assert abs(result["point"] - 0.5) < 0.05
    assert result["ci"][0] <= result["point"] <= result["ci"][1]
    assert result["n_groups"] == 200


def test_paired_bootstrap_ci_empty_groups():
    result = stats.paired_bootstrap_ci({}, np.mean)
    assert result["n_groups"] == 0


def test_paired_bootstrap_ci_correlated_rows_widen_interval():
    """Resampling by group (not row) should give a wider interval than
    resampling the same values as independent rows would, when rows within
    a group are correlated -- the whole reason for grouping.

    Uses 3 groups with *zero within-group noise* (40 identical rows each,
    at three distinct group means) so the comparison is not a statistical
    coin flip: group-level bootstrap can only ever draw from 3 distinct
    group means (a wide, discrete resampling distribution), while
    row-level bootstrap draws from 120 rows that -- despite being only 3
    distinct values repeated 40x each -- averages via the CLT into a much
    tighter concentration around the mean. A random-margin version of this
    test (noisy groups, seed-dependent effect size) was flaky; this
    deterministic version is not.
    """
    groups = {"g0": [0.0] * 40, "g1": [3.0] * 40, "g2": [-3.0] * 40}
    grouped = stats.paired_bootstrap_ci(groups, np.mean, n_resamples=2000, seed=2)

    flat_values = [v for vs in groups.values() for v in vs]
    row_groups = {f"row{i}": [v] for i, v in enumerate(flat_values)}
    row_level = stats.paired_bootstrap_ci(row_groups, np.mean, n_resamples=2000, seed=2)

    grouped_width = grouped["ci"][1] - grouped["ci"][0]
    row_width = row_level["ci"][1] - row_level["ci"][0]
    assert grouped_width > row_width * 3


def test_benjamini_hochberg_rejects_small_p_values():
    p_values = [0.001, 0.002, 0.9, 0.95, 0.5]
    result = stats.benjamini_hochberg(p_values, alpha=0.05)
    assert result["rejected"][0] is True
    assert result["rejected"][1] is True
    assert result["rejected"][3] is False


def test_benjamini_hochberg_empty():
    result = stats.benjamini_hochberg([])
    assert result["rejected"] == []


def test_reliability_table_and_ece_perfect_calibration():
    probs = [0.1] * 10 + [0.9] * 10
    outcomes = [0] * 9 + [1] + [1] * 9 + [0]
    ece = stats.expected_calibration_error(probs, outcomes, n_bins=2)
    assert ece < 0.2


def test_brier_score_perfect_predictions():
    assert stats.brier_score([1.0, 0.0], [1, 0]) == 0.0
