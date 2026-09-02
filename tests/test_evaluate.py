"""`evaluate.py`'s injection-recovery harness -- previously untested despite
producing the numbers behind every "method A beats method B" claim in
docs/DEFERRED.txt (e.g. the n=54 PCA-vs-deep-learning comparison). See the
P0 research plan's Tier 0.12 and Step 1d.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import evaluate


def _flat_sequence(length: int = 200) -> np.ndarray:
    """(2, length): row 0 values (all zero, MAD-normalised baseline), row 1
    the observed mask (all ones -- fully observed)."""
    return np.stack([np.zeros(length), np.ones(length)])


@pytest.mark.parametrize("kind", evaluate.ANOMALY_KINDS)
def test_inject_changes_only_masked_region_and_is_deterministic(kind):
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    seq = _flat_sequence()

    out_a = evaluate.inject(seq, kind, rng_a, strength=6.0)
    out_b = evaluate.inject(seq, kind, rng_b, strength=6.0)

    assert not np.allclose(out_a[0], seq[0]), f"{kind} injection did not perturb the sequence"
    np.testing.assert_array_equal(out_a[0], out_b[0])  # same seed -> same injection


def test_inject_respects_the_observation_mask():
    """Injected signal must vanish where the curve was never observed --
    `out[0] = values * mask` in evaluate.inject."""
    seq = _flat_sequence()
    seq[1, 100:] = 0.0  # unobserved for the back half
    rng = np.random.default_rng(1)

    out = evaluate.inject(seq, "step", rng, strength=6.0)

    assert np.all(out[0, 100:] == 0.0)


def test_inject_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown anomaly kind"):
        evaluate.inject(_flat_sequence(), "not-a-real-kind", np.random.default_rng(0))


def test_build_injected_labels_true_by_construction():
    values = np.stack([_flat_sequence() for _ in range(40)])
    identities = [{"object_id": f"obj{i}"} for i in range(40)]

    result = evaluate.build_injected(values, identities, fraction=0.25, seed=3)

    assert len(result) == 40
    assert result.labels.sum() == 10
    # Every labelled row actually differs from its untouched original.
    for i in np.where(result.labels == 1)[0]:
        assert not np.allclose(result.values[i], values[i])
    for i in np.where(result.labels == 0)[0]:
        np.testing.assert_array_equal(result.values[i], values[i])


def test_build_injected_empty_input():
    result = evaluate.build_injected(np.empty((0, 2, 10)), [], fraction=0.1)
    assert len(result) == 0
    assert result.to_dict()["injected"] == 0


def test_score_method_recovers_a_perfect_ranking():
    labels = np.array([0, 0, 1, 0, 1, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.15, 0.8, 0.05])  # both positives rank highest

    result = evaluate.score_method("perfect", scores, labels)

    assert result.roc_auc == pytest.approx(1.0)
    assert result.precision_at_k == pytest.approx(1.0)
    assert result.recall_at_k == pytest.approx(1.0)


def test_score_method_handles_degenerate_label_sets():
    """All-positive or all-negative labels have no meaningful ROC-AUC;
    the function must report NaN, not raise or silently fabricate a number."""
    all_zero = evaluate.score_method("m", np.array([0.1, 0.2]), np.array([0, 0]))
    all_one = evaluate.score_method("m", np.array([0.1, 0.2]), np.array([1, 1]))
    empty = evaluate.score_method("m", np.array([]), np.array([]))

    for result in (all_zero, all_one, empty):
        assert np.isnan(result.roc_auc)
        assert result.note == "degenerate label set"


def test_score_method_does_not_silently_credit_non_finite_scores():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, np.nan, 0.05, 0.9])

    result = evaluate.score_method("m", scores, labels)

    assert np.isfinite(result.roc_auc)  # must complete, not raise or propagate NaN


def test_comparison_best_ignores_non_finite_methods():
    comparison = evaluate.Comparison(methods=[
        evaluate.MethodScore("nan_method", float("nan"), float("nan"), float("nan"), float("nan")),
        evaluate.MethodScore("real_method", 0.75, 0.6, 0.5, 0.5),
    ])
    assert comparison.best().name == "real_method"


def test_comparison_best_is_none_when_every_method_is_degenerate():
    comparison = evaluate.Comparison(methods=[
        evaluate.MethodScore("a", float("nan"), float("nan"), float("nan"), float("nan")),
    ])
    assert comparison.best() is None
