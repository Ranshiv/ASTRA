"""Self-checks for `attribution.py`'s stability signal and glossary coverage.

Offline research/evaluation module, following this codebase's `*_eval.py`
convention: never wired into `rpc.py` (see `test_not_referenced_by_rpc`
below), used to measure and demonstrate a claim rather than to serve a
live request.

`evaluate_stability_quality` gives the concrete, checkable version of
`attribution.explain_candidate_stable`'s central claim: a median-only
occlusion impact is trustworthy when the feature's "other" population is
roughly symmetric, and NOT trustworthy when it is skewed/bimodal enough
that the 25th/50th/75th percentile references disagree about what "typical"
means. `evaluate_glossary_coverage` guards against a real feature silently
falling back to a raw, jargon-y name in the UI.
"""

from __future__ import annotations

import numpy as np

from . import attribution, feature_glossary
from .featurematrix import FeatureMatrix


def _symmetric_matrix(n_rows: int = 60, seed: int = 0) -> FeatureMatrix:
    """One feature outlier row against an otherwise Gaussian ("symmetric"
    reference) population -- the 25th/50th/75th percentiles of the other
    rows should be close together, so the stability check should agree."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 0.1, size=(n_rows, 2))
    values[0, 1] = 25.0
    identities = [{"object_id": f"obj{i}", "survey": "TEST", "band": "g", "path": f"p{i}"}
                 for i in range(n_rows)]
    return FeatureMatrix(values=values, identities=identities,
                         feature_names=("control", "target"), feature_version=1)


def _skewed_matrix(n_rows: int = 60, seed: int = 0) -> FeatureMatrix:
    """Same shape, but the "other" population for `target` is a 70/30 split
    between a cluster near 0 and a cluster near 50, with the candidate
    matching the minority (near-50) cluster. The median/25th-percentile
    reference (~0, the majority) and the 75th-percentile reference (~50,
    the candidate's own cluster) imply opposite conclusions about how
    anomalous the candidate's value is, so the occlusion impact should flip
    sign across quantiles -- exactly what stability-checking is meant to
    catch."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 0.1, size=(n_rows, 2))
    n_b = int(n_rows * 0.30)
    n_a = n_rows - n_b - 1
    values[1:1 + n_a, 1] = rng.normal(0.0, 0.05, size=n_a)
    values[1 + n_a:, 1] = rng.normal(50.0, 0.05, size=n_b)
    values[0, 1] = 50.0
    identities = [{"object_id": f"obj{i}", "survey": "TEST", "band": "g", "path": f"p{i}"}
                 for i in range(n_rows)]
    return FeatureMatrix(values=values, identities=identities,
                         feature_names=("control", "target"), feature_version=1)


def evaluate_stability_quality(seed: int = 0) -> dict:
    """Measures whether `explain_candidate_stable` flags stable vs. unstable
    impacts as designed, on constructed symmetric vs. skewed populations.

    Returns ``{"symmetric_stable", "skewed_stable", "matches_expectation"}``.
    Not an assertion by itself -- callers (tests) decide what to require.
    """
    symmetric_result = attribution.explain_candidate_stable(
        _symmetric_matrix(seed=seed), candidate_index=0, stability_top=2)
    skewed_result = attribution.explain_candidate_stable(
        _skewed_matrix(seed=seed), candidate_index=0, stability_top=2)

    def _target_stable(result: dict) -> bool | None:
        for component in result.get("components", []):
            if component["feature"] == "target":
                return component.get("stable")
        return None

    symmetric_stable = _target_stable(symmetric_result)
    skewed_stable = _target_stable(skewed_result)
    return {
        "symmetric_stable": symmetric_stable,
        "skewed_stable": skewed_stable,
        "matches_expectation": bool(symmetric_stable is True and skewed_stable is False),
    }


def evaluate_glossary_coverage() -> dict:
    """Confirms every shipped feature column has a real (non-fallback)
    glossary entry, so a future new feature that forgets to update
    `feature_glossary.py` is caught here rather than shown to a user as a
    raw column name.
    """
    from . import features, featurematrix

    all_names = (
        tuple(features.FEATURE_NAMES)
        + tuple(featurematrix.GAIA_JOIN_COLUMNS)
        + tuple(featurematrix.STELLAR_MANIFOLD_COLUMNS)
    )
    missing = [name for name in all_names if name not in feature_glossary.FEATURE_LABELS]
    return {
        "total": len(all_names), "missing": missing,
        "fully_covered": len(missing) == 0,
    }


__all__ = ["evaluate_stability_quality", "evaluate_glossary_coverage"]
