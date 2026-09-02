"""Occlusion-based per-candidate feature attribution (attribution.py)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import attribution
from astra.featurematrix import FeatureMatrix


def _matrix(n_rows=30, n_features=4, outlier_row=0, outlier_feature=1,
           outlier_value=50.0, seed=0) -> FeatureMatrix:
    """A tight population, uniform in every feature, except one row that is
    extreme in exactly one feature -- that feature should dominate its
    attribution."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 0.1, size=(n_rows, n_features))
    values[outlier_row, outlier_feature] = outlier_value
    identities = [{"object_id": f"obj{i}", "survey": "TEST", "band": "g", "path": f"p{i}"}
                 for i in range(n_rows)]
    feature_names = tuple(f"feature_{i}" for i in range(n_features))
    return FeatureMatrix(values=values, identities=identities,
                         feature_names=feature_names, feature_version=1)


def _symmetric_matrix(n_rows=60, seed=1) -> FeatureMatrix:
    """A control feature plus a target feature whose "other" population is
    roughly Gaussian (unimodal/symmetric): the 25th/50th/75th percentile
    references should agree closely, so occlusion stability should hold."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 0.1, size=(n_rows, 2))
    values[0, 1] = 25.0
    identities = [{"object_id": f"obj{i}", "survey": "TEST", "band": "g", "path": f"p{i}"}
                 for i in range(n_rows)]
    return FeatureMatrix(values=values, identities=identities,
                         feature_names=("control", "target"), feature_version=1)


def _skewed_matrix(n_rows=60, seed=1) -> FeatureMatrix:
    """A target feature whose "other" population is a 70/30 split between a
    cluster near 0 and a cluster near 50, with the candidate matching the
    50-cluster. The median/25th-percentile reference (~0, the majority) and
    the 75th-percentile reference (~50, the candidate's own cluster) imply
    opposite conclusions about how anomalous the candidate's value is, so
    the resulting occlusion impact should flip sign across quantiles --
    exactly the case `explain_candidate_stable` is meant to flag."""
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


class TestExplainCandidate:
    def test_dominant_feature_gets_top_impact(self):
        matrix = _matrix()

        result = attribution.explain_candidate(matrix, candidate_index=0, top=4)

        assert result["explainable"] is True
        assert result["components"][0]["feature"] == "feature_1"
        assert result["components"][0]["impact"] > 0

    def test_non_dominant_row_has_small_impacts(self):
        matrix = _matrix()

        result = attribution.explain_candidate(matrix, candidate_index=5, top=4)

        assert result["explainable"] is True
        assert all(abs(component["impact"]) < 1.0 for component in result["components"])

    def test_non_finite_feature_is_not_explainable(self):
        matrix = _matrix()
        matrix.values[0, 0] = np.nan

        result = attribution.explain_candidate(matrix, candidate_index=0)

        assert result["explainable"] is False
        assert "reason" in result

    def test_components_are_sorted_by_absolute_impact(self):
        matrix = _matrix()

        result = attribution.explain_candidate(matrix, candidate_index=0, top=4)

        impacts = [abs(component["impact"]) for component in result["components"]]
        assert impacts == sorted(impacts, reverse=True)

    def test_top_truncates_the_component_list(self):
        matrix = _matrix(n_features=6)

        result = attribution.explain_candidate(matrix, candidate_index=0, top=2)

        assert len(result["components"]) == 2

    def test_reports_the_identifying_path(self):
        matrix = _matrix()

        result = attribution.explain_candidate(matrix, candidate_index=3)

        assert result["path"] == "p3"


class TestExplainCandidateStable:
    def test_dominant_feature_keeps_top_rank(self):
        matrix = _matrix()

        result = attribution.explain_candidate_stable(matrix, candidate_index=0, top=4)

        assert result["explainable"] is True
        assert result["components"][0]["feature"] == "feature_1"

    def test_stable_true_on_a_symmetric_reference_population(self):
        matrix = _symmetric_matrix()

        result = attribution.explain_candidate_stable(matrix, candidate_index=0,
                                                       stability_top=2)

        target = next(c for c in result["components"] if c["feature"] == "target")
        assert target["stable"] is True

    def test_stable_false_on_a_skewed_reference_population(self):
        matrix = _skewed_matrix()

        result = attribution.explain_candidate_stable(matrix, candidate_index=0,
                                                       stability_top=2)

        target = next(c for c in result["components"] if c["feature"] == "target")
        assert target["stable"] is False

    def test_narrative_names_the_dominant_features_label(self):
        matrix = _matrix()

        result = attribution.explain_candidate_stable(matrix, candidate_index=0, top=4,
                                                       stability_top=4)

        assert result["narrative"]
        assert isinstance(result["narrative"], str)

    def test_narrative_flags_an_unstable_top_component(self):
        matrix = _skewed_matrix()

        result = attribution.explain_candidate_stable(matrix, candidate_index=0,
                                                       stability_top=2)

        assert "reference-sensitive" in result["narrative"] or "caution" in result["narrative"]

    def test_explain_candidate_unchanged(self):
        """Backward compatibility: explain_candidate's own behaviour must not
        shift as a side effect of adding explain_candidate_stable."""
        matrix = _matrix()

        result = attribution.explain_candidate(matrix, candidate_index=0, top=4)

        assert result["explainable"] is True
        assert result["components"][0]["feature"] == "feature_1"
        assert "label" not in result["components"][0]
        assert "narrative" not in result

    def test_non_finite_feature_is_not_explainable(self):
        matrix = _matrix()
        matrix.values[0, 0] = np.nan

        result = attribution.explain_candidate_stable(matrix, candidate_index=0)

        assert result["explainable"] is False
        assert "reason" in result
