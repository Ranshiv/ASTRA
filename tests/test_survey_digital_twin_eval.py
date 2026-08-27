"""survey_digital_twin_eval.py: distance and transfer-performance success
criteria for backlog item 42.

Deep-model training tests skip cleanly without PyTorch, the same convention
`test_open_world_eval.py` uses for the same reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import survey_digital_twin_eval as sdte
from astra import tensors

torch = pytest.importorskip("torch", reason="PyTorch not installed")


def _sequence_batch(n, length=32, seed=0, scale=1.0, coverage=1.0) -> tensors.SequenceBatch:
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 1, length)
    values = np.stack([
        scale * np.sin(time * rng.uniform(2, 6)) + rng.normal(0, 0.05, length)
        for _ in range(n)
    ]).astype(np.float32)
    mask = (rng.random((n, length)) < coverage).astype(np.float32)
    stacked = np.stack([values * mask, mask], axis=1)
    identities = [{"object_id": f"obj{i}"} for i in range(n)]
    return tensors.SequenceBatch(values=stacked, identities=identities, length=length)


class TestSummaryStatisticDistance:
    def test_identical_populations_have_near_zero_distance(self):
        batch = _sequence_batch(n=80, seed=1)

        result = sdte.summary_statistic_distance(batch.values, batch.values)

        assert result["mean_ks_statistic"] == pytest.approx(0.0, abs=1e-9)
        assert all(value == pytest.approx(0.0, abs=1e-9)
                  for value in result["per_feature"].values())

    def test_very_different_populations_have_large_distance(self):
        real = _sequence_batch(n=80, seed=1, scale=1.0)
        synthetic = _sequence_batch(n=80, seed=2, scale=50.0)

        result = sdte.summary_statistic_distance(real.values, synthetic.values)

        assert result["mean_ks_statistic"] > 0.5

    def test_too_few_rows_reports_a_note_instead_of_a_number(self):
        real = _sequence_batch(n=1, seed=1)
        synthetic = _sequence_batch(n=1, seed=2)

        result = sdte.summary_statistic_distance(real.values, synthetic.values)

        assert np.isnan(result["mean_ks_statistic"])
        assert "note" in result

    def test_reports_every_named_summary_feature(self):
        real = _sequence_batch(n=80, seed=1)
        synthetic = _sequence_batch(n=80, seed=2)

        result = sdte.summary_statistic_distance(real.values, synthetic.values)

        assert set(result["per_feature"]) == set(sdte._SUMMARY_FEATURE_NAMES)


class TestEvaluateTransferPerformance:
    def test_requires_at_least_two_seeds(self):
        real = _sequence_batch(n=60, seed=1)
        synthetic = _sequence_batch(n=60, seed=2)

        with pytest.raises(ValueError, match="at least two seeds"):
            sdte.evaluate_transfer_performance(real, synthetic, seeds=(1,))

    def test_requires_enough_real_rows_for_a_test_split(self):
        real = _sequence_batch(n=5, seed=1)
        synthetic = _sequence_batch(n=60, seed=2)

        with pytest.raises(ValueError, match="real_batch needs at least"):
            sdte.evaluate_transfer_performance(real, synthetic)

    def test_requires_enough_synthetic_rows_to_train_on(self):
        real = _sequence_batch(n=60, seed=1)
        synthetic = _sequence_batch(n=5, seed=2)

        with pytest.raises(ValueError, match="synthetic_batch needs at least"):
            sdte.evaluate_transfer_performance(real, synthetic)

    def test_runs_both_arms_and_returns_well_formed_summaries(self):
        real = _sequence_batch(n=60, seed=1, length=32)
        synthetic = _sequence_batch(n=60, seed=2, length=32)

        result = sdte.evaluate_transfer_performance(
            real, synthetic, fraction=0.15, seeds=(1, 2), epochs=3,
            model_config=None)

        assert set(result.keys()) == {
            "trained_on_real", "trained_on_synthetic", "held_out_test_injection"}
        # Not asserting which arm wins -- only that both ran and produced a
        # well-formed summary shape, the same restraint
        # `test_open_world_eval.py` applies to its own closed/open arms.
        for arm in ("trained_on_real", "trained_on_synthetic"):
            if result[arm] is not None:
                assert {"mean", "std", "ci95", "n"} <= result[arm].keys()
