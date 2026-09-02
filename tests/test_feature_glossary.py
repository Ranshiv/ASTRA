"""Human-readable feature labels/units (feature_glossary.py)."""

from __future__ import annotations

from astra import feature_glossary, features, featurematrix


class TestDescribe:
    def test_known_feature_has_a_real_label(self):
        info = feature_glossary.describe("time_span_days")

        assert info["label"] == "Observation baseline"
        assert info["unit"] == "days"

    def test_unknown_feature_falls_back_gracefully(self):
        info = feature_glossary.describe("totally_new_column")

        assert info["label"] == "Totally New Column"
        assert info["unit"] is None
        assert info["description"]


class TestCoverage:
    def test_every_shipped_feature_name_is_covered(self):
        all_names = (
            tuple(features.FEATURE_NAMES)
            + tuple(featurematrix.GAIA_JOIN_COLUMNS)
            + tuple(featurematrix.STELLAR_MANIFOLD_COLUMNS)
        )
        missing = [name for name in all_names if name not in feature_glossary.FEATURE_LABELS]
        assert missing == []


class TestFormatValue:
    def test_formats_a_unit_bearing_feature(self):
        text = feature_glossary.format_value("time_span_days", 14.2)
        assert text == "14.2 days"

    def test_formats_a_unitless_feature(self):
        text = feature_glossary.format_value("reduced_chi2", 0.6234)
        assert text == "0.623"

    def test_formats_nan_as_not_available(self):
        text = feature_glossary.format_value("time_span_days", float("nan"))
        assert text == "n/a"
