"""Positional matching, proper motion and cross-survey evidence."""

from __future__ import annotations

import numpy as np
import pytest

from astra import crossmatch, evidence
from astra.crossmatch import MatchGroup
from astra.surveys.base import LightCurve, SourceRef


def src(survey, oid, ra, dec, **extra):
    return SourceRef(survey=survey, object_id=oid, ra_deg=ra, dec_deg=dec,
                     extra=extra)


class TestSeparation:
    def test_identical_positions_are_zero(self):
        assert crossmatch.angular_separation_arcsec(10, 20, 10, 20) == \
            pytest.approx(0.0)

    def test_one_arcsec_in_declination(self):
        sep = crossmatch.angular_separation_arcsec(10.0, 20.0, 10.0,
                                                   20.0 + 1 / 3600)
        assert sep == pytest.approx(1.0, rel=1e-4)

    def test_ra_separation_shrinks_with_declination(self):
        """One degree of RA is a smaller arc near the pole."""
        equator = crossmatch.angular_separation_arcsec(10.0, 0.0, 11.0, 0.0)
        high = crossmatch.angular_separation_arcsec(10.0, 60.0, 11.0, 60.0)
        assert high == pytest.approx(equator * 0.5, rel=0.01)


class TestProperMotion:
    def test_no_motion_leaves_the_position_alone(self):
        ra, dec = crossmatch.propagate_position(10.0, 20.0, None, None,
                                                2016.0, 2019.0)
        assert (ra, dec) == (10.0, 20.0)

    def test_declination_motion_accumulates(self):
        _, dec = crossmatch.propagate_position(10.0, 20.0, 0.0, 1000.0,
                                               2016.0, 2019.0)
        assert (dec - 20.0) * 3600 == pytest.approx(3.0, rel=1e-6)

    def test_ra_motion_is_divided_by_cos_dec(self):
        """pmRA is sky-projected, so RA itself changes faster near the pole."""
        ra_eq, _ = crossmatch.propagate_position(10.0, 0.0, 1000.0, 0.0,
                                                 2016.0, 2019.0)
        ra_hi, _ = crossmatch.propagate_position(10.0, 60.0, 1000.0, 0.0,
                                                 2016.0, 2019.0)
        assert (ra_hi - 10.0) == pytest.approx((ra_eq - 10.0) * 2.0, rel=1e-6)

    def test_pole_does_not_divide_by_zero(self):
        ra, dec = crossmatch.propagate_position(10.0, 90.0, 1000.0, 0.0,
                                                2016.0, 2019.0)
        assert np.isfinite(ra) and np.isfinite(dec)

    def test_gaia_source_is_propagated_from_its_own_epoch(self):
        source = src("Gaia", "g1", 10.0, 20.0, pmra=0.0, pmdec=1000.0)
        _, dec, moved = crossmatch.epoch_corrected(source, to_epoch=2019.0)
        assert moved is True
        assert (dec - 20.0) * 3600 == pytest.approx(3.0, rel=1e-6)


class TestMatching:
    def test_close_sources_match(self):
        a = [src("ZTF", "z1", 10.0, 20.0)]
        b = [src("Gaia", "g1", 10.0, 20.0 + 0.5 / 3600)]

        matches = crossmatch.match_catalogs(a, b, radius_arcsec=2.0)

        assert len(matches) == 1
        assert matches[0].separation_arcsec == pytest.approx(0.5, rel=0.01)

    def test_distant_sources_do_not_match(self):
        a = [src("ZTF", "z1", 10.0, 20.0)]
        b = [src("Gaia", "g1", 10.0, 20.0 + 10.0 / 3600)]
        assert crossmatch.match_catalogs(a, b, radius_arcsec=2.0) == []

    def test_nearest_counterpart_wins_and_competitors_are_counted(self):
        a = [src("ZTF", "z1", 10.0, 20.0)]
        b = [
            src("Gaia", "far", 10.0, 20.0 + 1.5 / 3600),
            src("Gaia", "near", 10.0, 20.0 + 0.3 / 3600),
        ]

        match = crossmatch.match_catalogs(a, b, radius_arcsec=2.0)[0]

        assert match.counterpart.object_id == "near"
        assert match.competitors == 1

    def test_high_proper_motion_star_still_matches(self):
        """The failure mode this exists to prevent: 1000 mas/yr over 3 years
        moves 3 arcsec, well outside a 2 arcsec radius."""
        ztf = [src("ZTF", "z1", 10.0, 20.0 + 3.0 / 3600)]
        gaia = [src("Gaia", "g1", 10.0, 20.0, pmra=0.0, pmdec=1000.0)]

        without = crossmatch.match_catalogs(ztf, gaia, 2.0, epoch=2016.0)
        with_motion = crossmatch.match_catalogs(ztf, gaia, 2.0, epoch=2019.0)

        assert without == []
        assert len(with_motion) == 1
        assert with_motion[0].proper_motion_applied is True

    def test_empty_inputs_are_safe(self):
        assert crossmatch.match_catalogs([], [src("Gaia", "g", 1, 1)]) == []
        assert crossmatch.match_catalogs([src("ZTF", "z", 1, 1)], []) == []


class TestGrouping:
    def test_group_collects_all_surveys(self):
        groups = crossmatch.group_sources({
            "ZTF": [src("ZTF", "z1", 10.0, 20.0), src("ZTF", "z2", 11.0, 21.0)],
            "Gaia": [src("Gaia", "g1", 10.0, 20.0 + 0.4 / 3600)],
        })

        first = next(g for g in groups if g.members["ZTF"].object_id == "z1")
        assert first.independent_surveys == 2
        assert first.members["Gaia"].object_id == "g1"

    def test_unmatched_object_is_still_a_group(self):
        """Being seen by only one instrument is information, not nothing."""
        groups = crossmatch.group_sources({
            "ZTF": [src("ZTF", "z2", 11.0, 21.0)],
            "Gaia": [src("Gaia", "g1", 10.0, 20.0)],
        })
        assert all(g.independent_surveys >= 1 for g in groups)

    def test_summary_counts_multi_survey_groups(self):
        groups = crossmatch.group_sources({
            "ZTF": [src("ZTF", "z1", 10.0, 20.0), src("ZTF", "z2", 50.0, 20.0)],
            "Gaia": [src("Gaia", "g1", 10.0, 20.0)],
        })
        summary = crossmatch.summarise(groups)

        assert summary["groups"] == 2
        assert summary["multi_survey"] == 1

    def test_empty_input(self):
        assert crossmatch.group_sources({}) == []
        assert crossmatch.summarise([])["groups"] == 0

    def test_grouping_bias_report_exposes_anchor_selection(self):
        sources = {
            "ZTF": [src("ZTF", "z1", 10.0, 20.0), src("ZTF", "z2", 11.0, 21.0)],
            "Gaia": [src("Gaia", "g1", 10.0, 20.0)],
        }
        report = crossmatch.grouping_bias_report(sources)
        assert report["anchor_survey"] == "ZTF"
        assert report["survey_counts"] == {"ZTF": 2, "Gaia": 1}
        assert "selection function" in report["warning"]


class TestBlending:
    def test_shared_counterpart_is_flagged_as_blended(self):
        """TESS pixels are 21 arcsec, so one TESS source covers many stars.
        Counting it once per neighbour would manufacture confirmation."""
        groups = crossmatch.group_sources({
            "ZTF": [src("ZTF", "z1", 10.0, 20.0),
                    src("ZTF", "z2", 10.0, 20.0 + 5.0 / 3600),
                    src("ZTF", "z3", 10.0, 20.0 + 9.0 / 3600)],
            "TESS": [src("TESS", "TIC 1", 10.0, 20.0 + 4.0 / 3600)],
        }, radius_arcsec=15.0)

        assert all("TESS" in g.blended for g in groups)
        assert all(g.resolved_surveys == 1 for g in groups)

    def test_unique_well_separated_match_is_not_blended(self):
        groups = crossmatch.group_sources({
            "ZTF": [src("ZTF", "z1", 10.0, 20.0)],
            "Gaia": [src("Gaia", "g1", 10.0, 20.0 + 0.05 / 3600)],
        }, radius_arcsec=2.0)

        assert groups[0].blended == set()
        assert groups[0].resolved_surveys == 2

    def test_match_inside_the_survey_beam_is_blended(self):
        """A 10 arcsec separation is well inside a TESS pixel."""
        groups = crossmatch.group_sources({
            "ZTF": [src("ZTF", "z1", 10.0, 20.0)],
            "TESS": [src("TESS", "TIC 1", 10.0, 20.0 + 10.0 / 3600)],
        }, radius_arcsec=15.0)

        assert "TESS" in groups[0].blended

    def test_blended_evidence_does_not_count_as_independent(self):
        blended = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[
                evidence.SurveyView("ZTF", "z", "g", 500, 100.0, 0.5668,
                                    20.0, 0.5, 2458000.0, 2458400.0),
                evidence.SurveyView("TESS", "t", "TESS", 18000, 200.0, 0.5665,
                                    27.0, 0.5, 2458000.0, 2458400.0),
            ],
            separations_arcsec={"ZTF": 0.0, "TESS": 8.0},
            blended=["TESS"], match_radius_arcsec=15.0))

        assert blended.resolved_surveys == 1
        assert blended.components["independent_detection"] < 0.5
        assert any("Blended" in note for note in blended.notes)

    def test_positional_quality_uses_the_actual_match_radius(self):
        """Grading a 15 arcsec match against a fixed 2 arcsec scale would
        saturate every group at zero."""
        profile = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[evidence.SurveyView("ZTF", "z", "g", 500, 100.0, 0.5,
                                       20.0, 0.5, 2458000.0, 2458400.0)],
            separations_arcsec={"ZTF": 0.0, "TESS": 3.0},
            match_radius_arcsec=15.0))

        assert profile.components["positional_quality"] == pytest.approx(0.8)


class TestPeriodAgreement:
    def test_identical_periods_agree(self):
        agreed, kind = evidence.periods_agree(0.5668, 0.5668)
        assert agreed and kind == "direct"

    def test_close_periods_agree_within_tolerance(self):
        """The real measured case: ZTF 0.5668 d against TESS 0.5657 d."""
        agreed, _ = evidence.periods_agree(0.5668, 0.5657)
        assert agreed

    def test_double_period_counts_as_agreement(self):
        agreed, kind = evidence.periods_agree(0.5668, 1.1336)
        assert agreed and "harmonic" in kind

    def test_half_period_counts_as_agreement(self):
        agreed, kind = evidence.periods_agree(1.1336, 0.5668)
        assert agreed and "harmonic" in kind

    def test_unrelated_periods_disagree(self):
        agreed, kind = evidence.periods_agree(0.5668, 3.7)
        assert not agreed and kind == "disagree"

    def test_missing_period_is_not_agreement(self):
        agreed, kind = evidence.periods_agree(float("nan"), 0.5)
        assert not agreed and kind == "unavailable"

    def test_one_day_alias_is_recognised(self):
        true_period = 0.5668
        alias = 1.0 / (1.0 / true_period - 1.0)
        agreed, kind = evidence.periods_agree(true_period, alias)
        assert agreed and "alias" in kind


class TestEvidenceScoring:
    def _view(self, survey, period, chi2=100.0, snr=20.0, start=2458000.0,
              end=2458400.0):
        return evidence.SurveyView(
            survey=survey, object_id="x", band="g", points=500,
            reduced_chi2=chi2, best_period_days=period, period_snr=snr,
            robust_amplitude=0.5, time_start=start, time_end=end,
        )

    def test_agreeing_surveys_score_higher_than_one_survey(self):
        alone = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", 0.5668)],
            separations_arcsec={"ZTF": 0.0}))
        together = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", 0.5668), self._view("TESS", 0.5657)],
            separations_arcsec={"ZTF": 0.0, "TESS": 0.3}))

        assert together.consistency > alone.consistency

    def test_disagreeing_periods_score_lower_than_agreeing_ones(self):
        agree = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", 0.5668), self._view("TESS", 0.5657)],
            separations_arcsec={"ZTF": 0.0, "TESS": 0.3}))
        disagree = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", 0.5668), self._view("TESS", 3.9)],
            separations_arcsec={"ZTF": 0.0, "TESS": 0.3}))

        assert agree.consistency > disagree.consistency
        assert disagree.components["period_agreement"] == 0.0

        # Not 1.0: weight version 2 discounts agreement by the probability
        # that two unrelated periods pass the alias-tolerant test anyway.
        assert 0.85 < agree.components["period_agreement"] < 1.0
        assert agree.components["period_agreement"] == pytest.approx(
            1.0 - agree.period_fap)

    def test_single_survey_is_flagged_as_unverifiable(self):
        profile = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", 0.5668)]))
        assert any("Fewer than two surveys resolve" in note
                   for note in profile.notes)

    def test_disjoint_epochs_are_noted(self):
        profile = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", 0.5, start=2458000.0, end=2458100.0),
                   self._view("TESS", 0.5, start=2459000.0, end=2459100.0)],
            separations_arcsec={"ZTF": 0.0, "TESS": 0.2}))

        assert profile.components["temporal_overlap"] == 0.0
        assert any("simultaneously" in note for note in profile.notes)

    def test_crowded_field_lowers_positional_quality(self):
        clean = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", 0.5)],
            separations_arcsec={"ZTF": 0.0, "TESS": 0.1}))
        crowded = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", 0.5)],
            separations_arcsec={"ZTF": 0.0, "TESS": 1.9},
            ambiguous=["TESS"]))

        assert clean.components["positional_quality"] > \
            crowded.components["positional_quality"]

    def test_consistency_stays_in_range(self):
        profile = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", 0.5668), self._view("TESS", 0.5657),
                   self._view("Gaia", 0.5668)],
            separations_arcsec={"ZTF": 0.0, "TESS": 0.1, "Gaia": 0.05}))

        assert 0.0 <= profile.consistency <= 1.0

    def test_empty_profile_scores_zero(self):
        profile = evidence.score_profile(evidence.CrossSurveyProfile())
        assert profile.consistency == pytest.approx(0.0)


class TestPeriodFalseAlarm:
    """A boolean "they agree" hides how easy agreeing is (plan section 15)."""

    def test_fap_is_a_few_percent_for_a_typical_period(self):
        fap = evidence.period_agreement_fap(0.5668, max_period_days=1370.0)
        assert 0.01 < fap < 0.20

    def test_short_periods_are_easier_to_match_by_chance(self):
        """Near the 0.05 d search floor the acceptance windows are wide in
        frequency and the aliases crowd together, so coincidence is likelier."""
        short = evidence.period_agreement_fap(0.06, max_period_days=1370.0)
        typical = evidence.period_agreement_fap(0.5668, max_period_days=1370.0)
        assert short > typical

    def test_a_looser_tolerance_raises_the_false_alarm_rate(self):
        tight = evidence.period_agreement_fap(0.5668, tolerance=0.01,
                                              max_period_days=1370.0)
        loose = evidence.period_agreement_fap(0.5668, tolerance=0.08,
                                              max_period_days=1370.0)
        assert loose > tight * 3

    def test_unusable_input_reports_no_information_not_certainty(self):
        assert evidence.period_agreement_fap(float("nan")) == 1.0
        assert evidence.period_agreement_fap(0.5, max_period_days=0.01) == 1.0

    def test_overlapping_windows_are_not_counted_twice(self):
        """A FAP is a probability; merging is what keeps it one."""
        for period in (0.051, 0.06, 0.5, 5.0, 50.0):
            assert 0.0 <= evidence.period_agreement_fap(
                period, max_period_days=1370.0) <= 1.0


class TestAmplitudeAgreement:
    """Plan section 16's amplitude/colour consistency."""

    def _view(self, survey, band, amplitude, value_kind="mag", median=float("nan")):
        return evidence.SurveyView(
            survey=survey, object_id="x", band=band, points=500,
            reduced_chi2=100.0, best_period_days=0.5, period_snr=20.0,
            robust_amplitude=amplitude, time_start=2458000.0,
            time_end=2458400.0, value_kind=value_kind, median_value=median,
        )

    def test_matching_amplitudes_agree(self):
        score, _ = evidence.amplitude_agreement(
            self._view("ZTF", "g", 0.40), self._view("PTF", "g", 0.40))
        assert score == pytest.approx(1.0)

    def test_wildly_different_amplitudes_disagree(self):
        score, _ = evidence.amplitude_agreement(
            self._view("ZTF", "g", 0.40), self._view("PTF", "g", 4.0))
        assert score == 0.0

    def test_magnitude_and_flux_are_compared_after_conversion(self):
        """0.1 mag is about a 9.2% flux variation; the raw numbers 0.1 and
        920 would be meaningless to compare directly."""
        ztf = self._view("ZTF", "g", 0.10)
        tess = self._view("TESS", "TESS", 920.0, value_kind="flux", median=10000.0)
        score, _ = evidence.amplitude_agreement(ztf, tess)
        assert score > 0.95

    def test_flux_without_a_median_is_incomparable_not_disagreement(self):
        assert evidence.amplitude_agreement(
            self._view("ZTF", "g", 0.10),
            self._view("TESS", "TESS", 920.0, value_kind="flux")) is None

    def test_missing_amplitude_is_incomparable(self):
        assert evidence.amplitude_agreement(
            self._view("ZTF", "g", float("nan")),
            self._view("PTF", "g", 0.4)) is None

    def test_redder_band_varying_more_is_penalised(self):
        """The blend signature: the two detections are not the same star."""
        normal = evidence.amplitude_agreement(
            self._view("ZTF", "g", 0.40), self._view("ZTF", "r", 0.30))
        inverted = evidence.amplitude_agreement(
            self._view("ZTF", "g", 0.30), self._view("ZTF", "r", 0.40 * 2))
        assert normal[0] > inverted[0]
        assert "redder band" in inverted[1]

    def test_incomparable_amplitudes_are_omitted_not_zeroed(self):
        """Punishing an object for a measurement nobody made would be wrong."""
        profile = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[
                self._view("ZTF", "g", 0.4),
                self._view("TESS", "TESS", 900.0, value_kind="flux"),
            ],
            separations_arcsec={"ZTF": 0.0, "TESS": 0.3}))

        assert "amplitude_agreement" not in profile.components
        assert profile.weight_used == pytest.approx(
            1.0 - evidence.WEIGHTS["amplitude_agreement"])

    def test_weight_used_is_full_when_every_component_is_available(self):
        profile = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", "g", 0.4), self._view("PTF", "g", 0.4)],
            separations_arcsec={"ZTF": 0.0, "PTF": 0.3}))

        assert profile.weight_used == pytest.approx(1.0)
        assert "amplitude_agreement" in profile.components

    def test_disagreeing_amplitudes_lower_the_consistency(self):
        agree = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", "g", 0.4), self._view("PTF", "g", 0.4)],
            separations_arcsec={"ZTF": 0.0, "PTF": 0.3}))
        disagree = evidence.score_profile(evidence.CrossSurveyProfile(
            views=[self._view("ZTF", "g", 0.4), self._view("PTF", "g", 6.0)],
            separations_arcsec={"ZTF": 0.0, "PTF": 0.3}))

        assert agree.consistency > disagree.consistency
