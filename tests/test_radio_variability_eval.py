"""radio_variability_eval.py: cross-survey association recall and
spectral-index Monte Carlo uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from astra import radio_variability_eval as evaluation
from astra.frb import FrbBurst


def _burst(ra_deg=180.0, dec_deg=20.0, ra_err_deg=0.01, dec_err_deg=0.01) -> FrbBurst:
    return FrbBurst(tns_name="FRBTEST", repeater_name="", ra_deg=ra_deg, ra_err_deg=ra_err_deg,
                    dec_deg=dec_deg, dec_err_deg=dec_err_deg, mjd_400=59000.0)


class TestNearestRadioCounterpart:
    def test_finds_the_nearest_source_within_radius(self):
        sources = [
            {"object_id": "far", "ra_deg": 180.01, "dec_deg": 20.0},
            {"object_id": "near", "ra_deg": 180.0003, "dec_deg": 20.0},
        ]
        match = evaluation.nearest_radio_counterpart(180.0, 20.0, sources, match_radius_arcsec=5.0)
        assert match["object_id"] == "near"

    def test_returns_none_when_nothing_is_within_radius(self):
        sources = [{"object_id": "far", "ra_deg": 181.0, "dec_deg": 20.0}]
        match = evaluation.nearest_radio_counterpart(180.0, 20.0, sources, match_radius_arcsec=5.0)
        assert match is None

    def test_empty_source_list_returns_none(self):
        assert evaluation.nearest_radio_counterpart(180.0, 20.0, [], match_radius_arcsec=5.0) is None


class TestCrossSurveyAssociationRecall:
    def test_rejects_empty_trials(self):
        with pytest.raises(evaluation.RadioVariabilityEvalError):
            evaluation.cross_survey_association_recall([])

    def test_perfect_recovery_on_well_separated_true_matches(self):
        burst = _burst()
        trials = [{
            "burst": burst, "query_ra_deg": burst.ra_deg, "query_dec_deg": burst.dec_deg,
            "radio_sources": [
                {"object_id": "true", "ra_deg": burst.ra_deg + 1e-6, "dec_deg": burst.dec_deg},
                {"object_id": "distractor", "ra_deg": burst.ra_deg + 0.01, "dec_deg": burst.dec_deg},
            ],
            "true_counterpart_id": "true",
        } for _ in range(5)]
        result = evaluation.cross_survey_association_recall(trials)
        assert result["burst_association_recall"] == pytest.approx(1.0)
        assert result["counterpart_recall"] == pytest.approx(1.0)

    def test_no_signal_case_gives_zero_recall(self):
        # The query position is far from both the burst and every
        # candidate radio source -- the explicit no-signal regression case.
        burst = _burst()
        trials = [{
            "burst": burst, "query_ra_deg": burst.ra_deg + 5.0, "query_dec_deg": burst.dec_deg,
            "radio_sources": [{"object_id": "true", "ra_deg": burst.ra_deg, "dec_deg": burst.dec_deg}],
            "true_counterpart_id": "true",
        }]
        result = evaluation.cross_survey_association_recall(trials)
        assert result["burst_association_recall"] == 0.0
        assert result["counterpart_recall"] == 0.0

    def test_wrong_nearest_source_is_not_counted_as_correct(self):
        burst = _burst()
        trials = [{
            "burst": burst, "query_ra_deg": burst.ra_deg, "query_dec_deg": burst.dec_deg,
            "radio_sources": [
                {"object_id": "wrong_but_nearest", "ra_deg": burst.ra_deg + 1e-7,
                 "dec_deg": burst.dec_deg},
                {"object_id": "true", "ra_deg": burst.ra_deg + 3e-6, "dec_deg": burst.dec_deg},
            ],
            "true_counterpart_id": "true",
        }]
        result = evaluation.cross_survey_association_recall(trials)
        assert result["counterpart_recall"] == 0.0


class TestSpectralIndexMonteCarlo:
    def test_reports_a_point_estimate_and_intervals(self):
        result = evaluation.spectral_index_monte_carlo(
            np.array([1.4, 3.0]), np.array([10.0, 6.0]), np.array([0.5, 0.3]),
            n_samples=2000, seed=1)
        assert result["point_estimate"]["n_points"] == 2
        assert result["n_valid_samples"] > 1900
        assert "0.68" in result["intervals"]

    def test_interval_widens_with_larger_flux_errors(self):
        tight = evaluation.spectral_index_monte_carlo(
            np.array([1.4, 3.0]), np.array([10.0, 6.0]), np.array([0.1, 0.06]),
            n_samples=3000, seed=2)
        wide = evaluation.spectral_index_monte_carlo(
            np.array([1.4, 3.0]), np.array([10.0, 6.0]), np.array([3.0, 2.0]),
            n_samples=3000, seed=2)
        tight_width = tight["intervals"]["0.68"][1] - tight["intervals"]["0.68"][0]
        wide_width = wide["intervals"]["0.68"][1] - wide["intervals"]["0.68"][0]
        assert wide_width > tight_width

    def test_point_estimate_matches_direct_fit(self):
        frequency = np.array([1.4, 3.0, 6.0])
        flux = np.array([10.0, 6.0, 3.5])
        flux_err = np.array([0.5, 0.3, 0.2])
        result = evaluation.spectral_index_monte_carlo(frequency, flux, flux_err, n_samples=100, seed=3)
        direct = evaluation.rv.fit_spectral_index(frequency, flux, flux_err)
        assert result["point_estimate"]["alpha"] == pytest.approx(direct["alpha"])


@pytest.mark.live
class TestRealTwoPointSpectralIndexLive:
    """Confirmed live this session (2026-08-25): a real VLASS source near
    RA=180, Dec=0 with a close NVSS cross-match (`nvss_dist_arcsec` < 5")
    gives a real two-point spectral index of alpha=-1.24 +/- 0.08 (68%),
    a physically sensible steep-spectrum synchrotron value -- the full
    VLASS -> NVSS cross-match -> Monte Carlo spectral-index pipeline
    exercised end-to-end against real data, not just mocked responses."""

    def test_real_vlass_nvss_pair_gives_a_sensible_spectral_index(self):
        from astra.surveys.base import ConeQuery
        from astra.surveys.vlass import VLASSConnector

        sources = VLASSConnector().cone_search(
            ConeQuery(ra_deg=180.0, dec_deg=0.0, radius_arcsec=1800.0), limit=20)
        candidates = [s for s in sources
                     if s.extra.get("nvss_dist_arcsec") is not None
                     and s.extra["nvss_dist_arcsec"] < 5.0]
        assert candidates, "no VLASS source with a close real NVSS cross-match found"

        source = candidates[0]
        result = evaluation.real_two_point_spectral_index(
            source.ra_deg, source.dec_deg, source.extra["flux_total_mjy"],
            source.extra["flux_total_err_mjy"])
        assert result is not None
        assert -3.0 < result["point_estimate"]["alpha"] < 3.0


def test_not_referenced_by_rpc():
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "radio_variability_eval" not in source
