"""xray_hardness.py: hardness-ratio computation and discrete-state model."""

from __future__ import annotations

import numpy as np
import pytest

from astra import xray_hardness as xh


class TestHardnessRatio:
    def test_equal_soft_and_hard_gives_zero(self):
        ratio = xh.hardness_ratio(np.array([5.0]), np.array([5.0]))
        assert ratio[0] == pytest.approx(0.0)

    def test_pure_hard_gives_plus_one(self):
        ratio = xh.hardness_ratio(np.array([0.0]), np.array([10.0]))
        assert ratio[0] == pytest.approx(1.0)

    def test_pure_soft_gives_minus_one(self):
        ratio = xh.hardness_ratio(np.array([10.0]), np.array([0.0]))
        assert ratio[0] == pytest.approx(-1.0)

    def test_zero_total_gives_nan_not_a_fabricated_value(self):
        ratio = xh.hardness_ratio(np.array([0.0]), np.array([0.0]))
        assert np.isnan(ratio[0])

    def test_rejects_negative_flux(self):
        with pytest.raises(xh.XrayHardnessError):
            xh.hardness_ratio(np.array([-1.0]), np.array([1.0]))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(xh.XrayHardnessError):
            xh.hardness_ratio(np.array([1.0, 2.0]), np.array([1.0]))


class TestFitHardnessStates:
    def test_rejects_too_few_points(self):
        with pytest.raises(xh.XrayHardnessError):
            xh.fit_hardness_states(np.array([0.1, 0.2, 0.3]), n_states=2)

    def test_rejects_non_positive_n_states(self):
        with pytest.raises(xh.XrayHardnessError):
            xh.fit_hardness_states(np.array([0.1, 0.2, 0.3, 0.4]), n_states=0)

    def test_recovers_two_well_separated_clusters(self):
        rng = np.random.default_rng(1)
        soft_state = rng.normal(-0.6, 0.03, 20)
        hard_state = rng.normal(0.6, 0.03, 20)
        series = np.concatenate([soft_state, hard_state])
        fit = xh.fit_hardness_states(series, n_states=2, seed=1)
        assert fit["n_points_fit"] == 40
        # State 0 must be the SOFTER state (ascending-mean relabelling).
        assert fit["state_means"][0] < fit["state_means"][1]
        labels = np.array(fit["labels"])
        assert set(labels[:20].tolist()) == {0}
        assert set(labels[20:].tolist()) == {1}

    def test_non_finite_points_are_labelled_minus_one(self):
        series = np.array([0.1, 0.2, np.nan, 0.3, -0.1, -0.2])
        fit = xh.fit_hardness_states(series, n_states=2, seed=2)
        assert fit["labels"][2] == -1
        assert fit["n_points_fit"] == 5

    def test_reports_convergence(self):
        rng = np.random.default_rng(3)
        series = rng.normal(0.0, 0.05, 20)
        fit = xh.fit_hardness_states(series, n_states=1, seed=3)
        assert isinstance(fit["converged"], bool)


class TestDetectStateTransitions:
    def test_finds_the_single_transition_point(self):
        labels = [0, 0, 0, 1, 1, 1]
        assert xh.detect_state_transitions(labels) == [3]

    def test_no_transitions_in_a_constant_series(self):
        assert xh.detect_state_transitions([0, 0, 0, 0]) == []

    def test_missing_data_gap_is_not_counted_as_a_transition(self):
        # -1 (missing) sits between two SAME-state real points; the
        # transition check should skip over the gap, not fire on it.
        labels = [0, 0, -1, 0, 0]
        assert xh.detect_state_transitions(labels) == []

    def test_a_real_transition_across_a_gap_is_still_found(self):
        labels = [0, 0, -1, 1, 1]
        assert xh.detect_state_transitions(labels) == [3]

    def test_multiple_transitions(self):
        labels = [0, 1, 0, 1]
        assert xh.detect_state_transitions(labels) == [1, 2, 3]


def test_not_referenced_by_rpc():
    """Diagnostic-only discipline: matching every prior roadmap module."""
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "xray_hardness" not in source
