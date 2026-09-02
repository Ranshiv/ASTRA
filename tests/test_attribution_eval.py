"""attribution_eval.py: stability-check and glossary-coverage self-checks."""

from __future__ import annotations

from astra import attribution_eval as evaluation


class TestEvaluateStabilityQuality:
    def test_matches_expectation_on_constructed_populations(self):
        result = evaluation.evaluate_stability_quality(seed=1)

        assert result["symmetric_stable"] is True
        assert result["skewed_stable"] is False
        assert result["matches_expectation"] is True


class TestEvaluateGlossaryCoverage:
    def test_fully_covered(self):
        result = evaluation.evaluate_glossary_coverage()

        assert result["fully_covered"] is True
        assert result["missing"] == []


def test_not_referenced_by_rpc():
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "attribution_eval" not in source
