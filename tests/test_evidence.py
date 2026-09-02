"""Cross-survey evidence scoring: period agreement (with harmonics/aliases
and its false-alarm probability), amplitude/colour consistency, and the
overall weighted consistency score. No test file existed for this despite it
being the module the whole project's cross-survey corroboration argument
rests on."""

from __future__ import annotations

import math

import numpy as np
import pytest

from astra import evidence
from astra.crossmatch import MatchGroup
from astra.surveys.base import SourceRef


def _view(survey="ZTF", band="g", period=1.0, snr=10.0, chi2=5.0,
         amplitude=0.1, start=0.0, end=100.0, value_kind="mag",
         median=18.0) -> evidence.SurveyView:
    return evidence.SurveyView(
        survey=survey, object_id="obj-1", band=band, points=200,
        reduced_chi2=chi2, best_period_days=period, period_snr=snr,
        robust_amplitude=amplitude, time_start=start, time_end=end,
        value_kind=value_kind, median_value=median,
    )


class TestPeriodsAgree:
    def test_identical_periods_agree_directly(self):
        agree, kind = evidence.periods_agree(1.0, 1.0)
        assert agree is True
        assert kind == "direct"

    def test_double_period_agrees_as_a_harmonic(self):
        agree, kind = evidence.periods_agree(1.0, 2.0)
        assert agree is True
        assert "harmonic" in kind

    def test_half_period_agrees_as_a_harmonic(self):
        agree, kind = evidence.periods_agree(2.0, 1.0)
        assert agree is True

    def test_one_day_alias_agrees(self):
        # 1/P_true = 1/P_obs - 1 for a real signal aliased by nightly sampling.
        true_period = 1.0 / (1.0 / 0.9091 - 1)
        agree, kind = evidence.periods_agree(0.9091, true_period)
        assert agree is True
        assert "alias" in kind

    def test_unrelated_periods_disagree(self):
        agree, kind = evidence.periods_agree(1.0, 7.3)
        assert agree is False
        assert kind == "disagree"

    def test_non_finite_period_is_unavailable(self):
        agree, kind = evidence.periods_agree(float("nan"), 1.0)
        assert agree is False
        assert kind == "unavailable"

    def test_zero_or_negative_period_is_unavailable(self):
        assert evidence.periods_agree(0.0, 1.0) == (False, "unavailable")
        assert evidence.periods_agree(1.0, -1.0) == (False, "unavailable")


class TestPeriodAgreementFap:
    def test_non_finite_or_non_positive_period_is_fully_uninformative(self):
        assert evidence.period_agreement_fap(float("nan")) == 1.0
        assert evidence.period_agreement_fap(0.0) == 1.0
        assert evidence.period_agreement_fap(-1.0) == 1.0

    def test_degenerate_search_band_is_fully_uninformative(self):
        assert evidence.period_agreement_fap(
            1.0, min_period_days=10.0, max_period_days=5.0) == 1.0

    def test_a_normal_period_yields_a_fap_between_zero_and_one(self):
        fap = evidence.period_agreement_fap(1.0, min_period_days=0.1,
                                            max_period_days=100.0)
        assert 0.0 <= fap <= 1.0

    def test_a_wider_search_band_yields_a_smaller_fap(self):
        narrow = evidence.period_agreement_fap(1.0, min_period_days=0.5,
                                               max_period_days=2.0)
        wide = evidence.period_agreement_fap(1.0, min_period_days=0.01,
                                             max_period_days=1000.0)
        assert wide < narrow


class TestFractionalAmplitude:
    def test_mag_amplitude_is_divided_by_the_linearisation_constant(self):
        view = _view(amplitude=1.0857, value_kind="mag")
        assert evidence.fractional_amplitude(view) == pytest.approx(1.0)

    def test_flux_amplitude_is_divided_by_the_median(self):
        view = _view(amplitude=10.0, value_kind="flux", median=100.0)
        assert evidence.fractional_amplitude(view) == pytest.approx(0.1)

    def test_non_finite_amplitude_yields_nan(self):
        view = _view(amplitude=float("nan"))
        assert math.isnan(evidence.fractional_amplitude(view))

    def test_zero_amplitude_yields_nan(self):
        view = _view(amplitude=0.0)
        assert math.isnan(evidence.fractional_amplitude(view))

    def test_flux_with_zero_median_yields_nan(self):
        view = _view(amplitude=10.0, value_kind="flux", median=0.0)
        assert math.isnan(evidence.fractional_amplitude(view))


class TestAmplitudeAgreement:
    def test_identical_amplitudes_in_the_same_band_score_near_one(self):
        first = _view(band="g", amplitude=1.0857)
        second = _view(band="g", amplitude=1.0857)
        score, note = evidence.amplitude_agreement(first, second)
        assert score == pytest.approx(1.0)
        assert "amplitude ratio" in note

    def test_wildly_different_amplitudes_score_low(self):
        first = _view(band="g", amplitude=1.0857)
        second = _view(band="g", amplitude=1.0857 * 100)
        score, _ = evidence.amplitude_agreement(first, second)
        assert score < 0.3

    def test_incomparable_pair_returns_none(self):
        first = _view(amplitude=float("nan"))
        second = _view(amplitude=1.0)
        assert evidence.amplitude_agreement(first, second) is None

    def test_redder_band_varying_more_is_penalised(self):
        # BAND_WAVELENGTH_NM: g (bluer, 484nm) vs z (redder, 900nm).
        bluer = _view(band="g", amplitude=1.0857 * 0.5)
        redder = _view(band="z", amplitude=1.0857 * 2.0)
        score, note = evidence.amplitude_agreement(bluer, redder)
        assert "redder band varies more" in note


class TestScoreProfile:
    def test_two_agreeing_surveys_score_well(self):
        profile = evidence.CrossSurveyProfile(
            views=[_view(survey="ZTF", period=1.0), _view(survey="TESS", period=1.0)],
            separations_arcsec={"ZTF": 0.1, "TESS": 0.2},
        )
        scored = evidence.score_profile(profile)
        assert scored.components["period_agreement"] > 0
        assert scored.components["independent_detection"] == pytest.approx(2 / 3)
        assert 0.0 <= scored.consistency <= 1.0
        assert scored.weight_used > 0

    def test_disagreeing_periods_score_zero_for_that_component(self):
        profile = evidence.CrossSurveyProfile(
            views=[_view(survey="ZTF", period=1.0, snr=10),
                   _view(survey="TESS", period=7.3, snr=10)],
        )
        scored = evidence.score_profile(profile)
        assert scored.components["period_agreement"] == 0.0
        assert "incompatible periods" in " ".join(scored.notes)

    def test_a_single_survey_cannot_confirm_itself(self):
        profile = evidence.CrossSurveyProfile(views=[_view(survey="ZTF")])
        scored = evidence.score_profile(profile)
        assert "Fewer than two surveys resolve" in " ".join(scored.notes)

    def test_blended_surveys_are_excluded_from_independent_detection(self):
        profile = evidence.CrossSurveyProfile(
            views=[_view(survey="ZTF"), _view(survey="TESS")],
            blended=["TESS"],
        )
        scored = evidence.score_profile(profile)
        assert scored.resolved_surveys == 1
        assert any("Blended in TESS" in note for note in scored.notes)

    def test_weight_used_excludes_amplitude_agreement_with_only_one_view(self):
        # Every component except amplitude_agreement always gets a real value
        # (falling back to 0.0 when there's no evidence); amplitude_agreement
        # is the one component actually OMITTED from `components` when no
        # pair exists to compare (a single view), so it alone drops out of
        # weight_used rather than being renormalised in as a 0.0.
        profile = evidence.CrossSurveyProfile(views=[_view()])
        scored = evidence.score_profile(profile)
        assert "amplitude_agreement" not in scored.components
        expected = sum(v for k, v in evidence.WEIGHTS.items() if k != "amplitude_agreement")
        assert scored.weight_used == pytest.approx(expected)

    def test_to_dict_reports_weights_and_weighted_contributions(self):
        profile = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[_view(survey="ZTF"), _view(survey="TESS")]))
        payload = profile.to_dict()
        assert payload["weight_version"] == evidence.WEIGHT_VERSION
        assert set(payload["weights"]) == set(payload["components"])
        for key, weight in payload["weights"].items():
            assert payload["weighted"][key] == pytest.approx(
                round(weight * payload["components"][key], 4))


class TestProfileGroup:
    def test_assembles_and_scores_views_from_matched_curves(self, curve):
        source_a = SourceRef(survey="ZTF", object_id="obj-1", ra_deg=180.1, dec_deg=22.4)
        source_b = SourceRef(survey="TESS", object_id="obj-1", ra_deg=180.1, dec_deg=22.4)
        group = MatchGroup(members={"ZTF": source_a, "TESS": source_b},
                           separations={"ZTF": 0.1, "TESS": 0.2})
        curves_by_key = {
            ("ZTF", "obj-1"): [curve],
            ("TESS", "obj-1"): [curve],
        }
        profile = evidence.profile_group(group, curves_by_key)
        assert len(profile.views) == 2
        assert profile.consistency >= 0.0

    def test_empty_curves_yield_no_views(self):
        source_a = SourceRef(survey="ZTF", object_id="obj-1", ra_deg=180.1, dec_deg=22.4)
        group = MatchGroup(members={"ZTF": source_a})
        profile = evidence.profile_group(group, {})
        assert profile.views == []


class TestLoadCurveIndex:
    def test_missing_root_returns_an_empty_index(self, tmp_path):
        index = evidence.load_curve_index(tmp_path / "does-not-exist")
        assert index.by_key == {}
        assert index.positions_by_path == {}

    def test_indexes_a_stored_curve_by_survey_and_object_id(self, tmp_path, curve):
        from astra import store

        result = store.write_curve(curve, tmp_path)
        index = evidence.load_curve_index(tmp_path)
        key = (curve.source.survey, curve.source.object_id)
        assert key in index.by_key
        assert str(result.path) in index.positions_by_path
        assert index.positions_by_path[str(result.path)]["ra_deg"] == curve.source.ra_deg

    def test_load_curves_by_key_is_the_same_index(self, tmp_path, curve):
        from astra import store

        store.write_curve(curve, tmp_path)
        by_key = evidence.load_curves_by_key(tmp_path)
        key = (curve.source.survey, curve.source.object_id)
        assert key in by_key

    def test_a_corrupt_file_does_not_stop_indexing(self, tmp_path, curve):
        from astra import store

        store.write_curve(curve, tmp_path)
        survey_dir = next(tmp_path.iterdir())
        bad = survey_dir / "corrupt.parquet"
        bad.write_bytes(b"not a real parquet file")
        index = evidence.load_curve_index(tmp_path)
        key = (curve.source.survey, curve.source.object_id)
        assert key in index.by_key
        assert len(index.by_key[key]) == 1
