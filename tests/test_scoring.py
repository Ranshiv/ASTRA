"""Composite scoring and artifact assessment."""

from __future__ import annotations

import numpy as np
import pytest

from astra import artifact, scoring


def features(**overrides) -> dict[str, float]:
    base = {
        "n_points": 500.0, "robust_amplitude": 0.5, "median_err": 0.02,
        "reduced_chi2": 150.0, "period_snr": 30.0, "eta": 0.5,
        "change_point_score": 4.0, "best_period_days": 0.5668,
        "kurtosis": 0.1, "time_span_days": 1500.0,
    }
    base.update(overrides)
    return base


class TestWeights:
    def test_weights_match_plan_section_16(self):
        assert scoring.WEIGHTS["statistical_rarity"] == 0.25
        assert scoring.WEIGHTS["cross_survey_consistency"] == 0.20
        assert scoring.WEIGHTS["temporal_uniqueness"] == 0.15
        assert scoring.WEIGHTS["physical_inconsistency"] == 0.15
        assert scoring.WEIGHTS["catalog_novelty"] == 0.10
        assert scoring.WEIGHTS["model_agreement"] == 0.10
        assert scoring.WEIGHTS["data_quality"] == 0.05

    def test_weights_sum_to_one(self):
        assert sum(scoring.WEIGHTS.values()) == pytest.approx(1.0)


class TestCombine:
    def test_all_components_present(self):
        result = scoring.combine({k: 1.0 for k in scoring.WEIGHTS})
        assert result.total == pytest.approx(1.0)
        assert result.weight_used == pytest.approx(1.0)

    def test_missing_components_are_renormalised_not_zeroed(self):
        """None means 'not checked'; a candidate must not be punished for it."""
        partial = {k: None for k in scoring.WEIGHTS}
        partial["statistical_rarity"] = 1.0

        result = scoring.combine(partial)

        assert result.total == pytest.approx(1.0)
        assert result.weight_used == pytest.approx(0.25)

    def test_zero_differs_from_missing(self):
        checked = dict.fromkeys(scoring.WEIGHTS, None)
        checked["statistical_rarity"] = 1.0
        checked["catalog_novelty"] = 0.0

        unchecked = dict.fromkeys(scoring.WEIGHTS, None)
        unchecked["statistical_rarity"] = 1.0

        assert scoring.combine(checked).total < scoring.combine(unchecked).total

    def test_no_components_scores_zero(self):
        result = scoring.combine(dict.fromkeys(scoring.WEIGHTS, None))
        assert result.total == 0.0

    def test_top_drivers_are_ordered_by_contribution(self):
        result = scoring.combine({
            **dict.fromkeys(scoring.WEIGHTS, 0.1),
            "statistical_rarity": 1.0,
        })
        assert result.top_drivers(1)[0][0] == "statistical_rarity"


class TestTemporalUniqueness:
    def test_strong_periodicity_scores_high(self):
        value = scoring.temporal_uniqueness(features(period_snr=50.0, eta=0.1))
        assert value > 0.6

    def test_white_noise_scores_low(self):
        value = scoring.temporal_uniqueness(
            features(period_snr=1.0, eta=2.0, change_point_score=3.0))
        assert value < 0.1

    def test_missing_inputs_return_none(self):
        assert scoring.temporal_uniqueness({}) is None


class TestDataQuality:
    def test_well_sampled_high_signal_scores_high(self):
        assert scoring.data_quality(features()) > 0.9

    def test_sparse_noisy_data_scores_low(self):
        value = scoring.data_quality(
            features(n_points=20.0, robust_amplitude=0.01, median_err=0.05))
        assert value < 0.2


class TestPhysicalInconsistency:
    def test_rr_lyrae_at_the_right_luminosity_is_consistent(self):
        score, reason = scoring.physical_inconsistency(
            0.5668, {"abs_g_mag": 0.6, "parallax_snr": 20.0})
        assert score == 0.0
        assert "consistent with rr_lyrae" in reason

    def test_rr_lyrae_period_but_far_too_faint_is_inconsistent(self):
        """The interesting case: pulsates like one, but is nowhere near one."""
        score, reason = scoring.physical_inconsistency(
            0.5668, {"abs_g_mag": 8.0, "parallax_snr": 20.0})
        assert score > 0.5
        assert "from the rr_lyrae locus" in reason

    def test_no_gaia_record_returns_none(self):
        score, reason = scoring.physical_inconsistency(0.5668, {})
        assert score is None
        assert "no Gaia astrometry" in reason

    def test_gaia_record_without_a_usable_parallax_returns_none(self):
        """Negative parallaxes are common in Gaia and yield no distance."""
        score, reason = scoring.physical_inconsistency(
            0.5668, {"parallax": -0.3, "abs_g_mag": None})
        assert score is None
        assert "no absolute magnitude" in reason

    def test_noisy_parallax_is_refused(self):
        score, reason = scoring.physical_inconsistency(
            0.5668, {"abs_g_mag": 0.6, "parallax_snr": 1.0})
        assert score is None
        assert "too noisy" in reason

    def test_extinction_correction_is_used_when_available(self):
        score, reason = scoring.physical_inconsistency(
            0.5668, {"abs_g_mag": 1.6, "parallax_snr": 20.0, "a_g": 1.0}
        )
        assert score == pytest.approx(0.0)
        assert "extinction-corrected" in reason

    def test_period_outside_every_class_returns_none(self):
        score, _ = scoring.physical_inconsistency(
            500.0, {"abs_g_mag": 0.6, "parallax_snr": 20.0})
        assert score is None


class TestCatalogNovelty:
    def test_known_variable_has_no_novelty(self):
        score, _ = scoring.catalog_novelty(
            {"phot_variable_flag": "VARIABLE"})
        assert score == 0.0

    def test_unflagged_source_scores_partial_novelty(self):
        score, reason = scoring.catalog_novelty(
            {"phot_variable_flag": "NOT_AVAILABLE"})
        assert 0.0 < score < 1.0
        assert "weak evidence" in reason

    def test_no_catalogue_information_returns_none(self):
        score, reason = scoring.catalog_novelty(None)
        assert score is None
        assert "no catalogue cross-reference" in reason


class TestArtifactAssessment:
    def test_clean_multi_survey_variable_is_not_an_artifact(self):
        result = artifact.assess(features(), resolved_surveys=2,
                                 period_agrees_across_surveys=True)
        assert result.likelihood < 0.3
        assert result.verdict == "no strong artifact indicators"

    def test_one_day_period_is_flagged_as_sampling(self):
        result = artifact.assess(features(best_period_days=1.0))
        names = [i.name for i in result.indicators]
        assert "sampling_period" in names

    def test_sidereal_day_is_flagged(self):
        found, name = artifact.matches_sampling_period(0.99727)
        assert found and "sidereal" in name

    def test_genuine_period_is_not_flagged(self):
        found, _ = artifact.matches_sampling_period(0.5668)
        assert not found

    def test_low_significance_variation_is_suspicious(self):
        result = artifact.assess(
            features(robust_amplitude=0.01, median_err=0.02, reduced_chi2=1.1))
        names = [i.name for i in result.indicators]
        assert "low_significance" in names
        assert "consistent_with_constant" in names
        assert result.likelihood > 0.4

    def test_single_instrument_raises_suspicion(self):
        alone = artifact.assess(features(), resolved_surveys=1)
        together = artifact.assess(features(), resolved_surveys=3)
        assert alone.likelihood > together.likelihood

    def test_extreme_kurtosis_suggests_cosmic_rays(self):
        result = artifact.assess(features(kurtosis=100.0))
        assert "extreme_outliers" in [i.name for i in result.indicators]

    def test_sparse_curve_is_suspicious(self):
        result = artifact.assess(features(n_points=20.0))
        assert "sparse_sampling" in [i.name for i in result.indicators]

    def test_blending_is_recorded(self):
        result = artifact.assess(features(), resolved_surveys=1,
                                 blended=["TESS"])
        assert "blended_photometry" in [i.name for i in result.indicators]

    def test_agreement_from_a_blended_survey_is_not_clearing_evidence(self):
        """Regression: the report must not say 'only one survey resolves this'
        and 'independent instruments agree' at the same time."""
        result = artifact.assess(features(), resolved_surveys=1,
                                 blended=["TESS"],
                                 period_agrees_across_surveys=True)

        assert not any("Independent instruments" in c
                       for c in result.clearing_evidence)
        assert "agreement_not_independent" in [i.name for i in result.indicators]

    def test_agreement_from_a_resolving_survey_is_clearing_evidence(self):
        result = artifact.assess(features(), resolved_surveys=2,
                                 period_agrees_across_surveys=True)
        assert any("Independent instruments" in c
                   for c in result.clearing_evidence)

    def test_period_disagreement_is_suspicious(self):
        result = artifact.assess(features(), resolved_surveys=2,
                                 period_agrees_across_surveys=False)
        assert "period_disagreement" in [i.name for i in result.indicators]

    def test_likelihood_stays_in_range(self):
        worst = artifact.assess(
            features(best_period_days=1.0, robust_amplitude=0.001,
                     median_err=0.05, reduced_chi2=1.0, n_points=10.0,
                     kurtosis=200.0, change_point_score=100.0),
            resolved_surveys=1, blended=["TESS"],
            period_agrees_across_surveys=False)
        assert 0.0 <= worst.likelihood <= 1.0
        assert worst.verdict == "probably an artifact"

    def test_clearing_evidence_lowers_the_likelihood(self):
        without = artifact.assess(features(reduced_chi2=1.5),
                                  resolved_surveys=1)
        with_evidence = artifact.assess(features(reduced_chi2=1.5),
                                        resolved_surveys=3,
                                        period_agrees_across_surveys=True)
        assert with_evidence.likelihood < without.likelihood
