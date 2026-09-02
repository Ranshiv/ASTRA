"""Footprint/stratum validation, the closed-form Poisson-Gamma credible
interval, hierarchical-rate recovery on synthetic strata, and the
anchor-survey sweep's reuse of `crossmatch.group_sources`.

The closed-form tests need no `research` extra. The `fit_hierarchical_rate`
tests are gated on `emcee`, matching `test_agn_changepoint.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import gamma

from astra import population_rate as pr
from astra.surveys.base import SourceRef

try:
    import emcee  # noqa: F401
    _HAS_EMCEE = True
except ImportError:
    _HAS_EMCEE = False

requires_emcee = pytest.mark.skipif(
    not _HAS_EMCEE, reason="emcee not installed (opt-in 'research' extra)")


# ---------------------------------------------------------------------------
# SurveyFootprint / Stratum validation
# ---------------------------------------------------------------------------

def test_survey_footprint_computes_exposure():
    footprint = pr.SurveyFootprint(survey="ZTF", area_deg2=1000.0, baseline_days=30.0)
    assert footprint.exposure_deg2_days == pytest.approx(30_000.0)


def test_survey_footprint_rejects_non_positive_inputs():
    with pytest.raises(pr.PopulationRateError):
        pr.SurveyFootprint(survey="ZTF", area_deg2=0.0, baseline_days=30.0)
    with pytest.raises(pr.PopulationRateError):
        pr.SurveyFootprint(survey="ZTF", area_deg2=100.0, baseline_days=-1.0)


class TestHealpixCoverage:
    def test_from_healpix_coverage_derives_area_from_real_pixel_count(self):
        astropy_healpix = pytest.importorskip("astropy_healpix")
        import astropy.units as u

        nside = 32
        healpix = astropy_healpix.HEALPix(nside=nside, order="nested")
        pixel_area_deg2 = float(healpix.pixel_area.to(u.deg ** 2).value)
        pixels = frozenset({0, 1, 2, 3, 4})

        footprint = pr.SurveyFootprint.from_healpix_coverage(
            "ZTF", baseline_days=30.0, covered_pixel_indices=pixels, nside=nside)

        assert footprint.area_deg2 == pytest.approx(pixel_area_deg2 * 5)
        assert footprint.has_coverage_map is True
        assert footprint.nside == nside

    def test_from_healpix_coverage_rejects_empty_pixel_set(self):
        pytest.importorskip("astropy_healpix")
        with pytest.raises(pr.PopulationRateError):
            pr.SurveyFootprint.from_healpix_coverage(
                "ZTF", baseline_days=30.0, covered_pixel_indices=set(), nside=32)

    def test_covers_reports_membership_via_shared_healpix_common_lookup(self):
        astropy_healpix = pytest.importorskip("astropy_healpix")
        import astropy.units as u

        from astra.healpix_common import _target_pixel

        nside = 32
        target_pixel = _target_pixel(180.0, 30.0, nside, "nested")
        footprint = pr.SurveyFootprint.from_healpix_coverage(
            "ZTF", baseline_days=30.0, covered_pixel_indices={target_pixel}, nside=nside)

        assert footprint.covers(180.0, 30.0) is True
        assert footprint.covers(0.0, -80.0) is False

    def test_plain_footprint_has_no_coverage_map(self):
        footprint = pr.SurveyFootprint(survey="ZTF", area_deg2=100.0, baseline_days=30.0)
        assert footprint.has_coverage_map is False
        with pytest.raises(pr.PopulationRateError):
            footprint.covers(180.0, 30.0)

    def test_covered_pixels_without_nside_rejected(self):
        with pytest.raises(pr.PopulationRateError):
            pr.SurveyFootprint(survey="ZTF", area_deg2=100.0, baseline_days=30.0,
                               covered_pixel_indices=frozenset({0, 1}))


def test_stratum_rejects_bad_completeness_and_detected():
    footprint = pr.SurveyFootprint(survey="ZTF", area_deg2=100.0, baseline_days=30.0)
    with pytest.raises(pr.PopulationRateError):
        pr.Stratum(label="s", footprint=footprint, completeness=0.0, detected=1)
    with pytest.raises(pr.PopulationRateError):
        pr.Stratum(label="s", footprint=footprint, completeness=1.5, detected=1)
    with pytest.raises(pr.PopulationRateError):
        pr.Stratum(label="s", footprint=footprint, completeness=0.5, detected=-1)


def test_stratum_exposure_multiplies_footprint_by_completeness():
    footprint = pr.SurveyFootprint(survey="ZTF", area_deg2=100.0, baseline_days=10.0)
    stratum = pr.Stratum(label="s", footprint=footprint, completeness=0.5, detected=3)
    assert stratum.exposure == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# poisson_rate_credible_interval
# ---------------------------------------------------------------------------

def test_poisson_rate_credible_interval_matches_hand_computed_gamma_quantile():
    detected, exposure, level = 12, 400.0, 0.9
    point, low, high = pr.poisson_rate_credible_interval(detected, exposure, level=level)
    assert point == pytest.approx(detected / exposure)
    tail = (1.0 - level) / 2.0
    expected_low = gamma.ppf(tail, a=detected + 0.5, scale=1.0 / exposure)
    expected_high = gamma.ppf(1.0 - tail, a=detected + 0.5, scale=1.0 / exposure)
    assert low == pytest.approx(expected_low)
    assert high == pytest.approx(expected_high)
    assert low < point < high


def test_poisson_rate_credible_interval_handles_zero_detections_sanely():
    point, low, high = pr.poisson_rate_credible_interval(0, 100.0)
    assert point == 0.0
    assert low >= 0.0
    assert high > low


def test_poisson_rate_credible_interval_handles_zero_exposure():
    assert pr.poisson_rate_credible_interval(5, 0.0) == (0.0, 0.0, float("inf"))


def test_poisson_rate_credible_interval_rejects_bad_inputs():
    with pytest.raises(pr.PopulationRateError):
        pr.poisson_rate_credible_interval(-1, 100.0)
    with pytest.raises(pr.PopulationRateError):
        pr.poisson_rate_credible_interval(1, -1.0)
    with pytest.raises(pr.PopulationRateError):
        pr.poisson_rate_credible_interval(1, 100.0, level=1.0)


# ---------------------------------------------------------------------------
# fit_hierarchical_rate
# ---------------------------------------------------------------------------

def _synthetic_strata(rng, n_strata=5, mu_log_rate=-3.0, sigma_log_rate=0.3):
    import math

    strata = []
    for i in range(n_strata):
        z = float(rng.normal())
        rate = math.exp(mu_log_rate + sigma_log_rate * z)
        footprint = pr.SurveyFootprint(survey=f"s{i}", area_deg2=200.0, baseline_days=180.0)
        completeness = 0.7
        expected = rate * footprint.exposure_deg2_days * completeness
        detected = int(rng.poisson(expected))
        strata.append(pr.Stratum(label=f"s{i}", footprint=footprint,
                                 completeness=completeness, detected=detected))
    return strata


@requires_emcee
def test_fit_hierarchical_rate_recovers_injected_mu_log_rate():
    rng = np.random.default_rng(11)
    strata = _synthetic_strata(rng, n_strata=8, mu_log_rate=-3.0, sigma_log_rate=0.2)
    fit = pr.fit_hierarchical_rate(strata, n_walkers=40, n_steps=3000, seed=7)
    assert fit.mu_log_rate_median == pytest.approx(-3.0, abs=0.6)
    assert set(fit.per_stratum_rate_medians) == {s.label for s in strata}


@requires_emcee
def test_fit_hierarchical_rate_rejects_bad_inputs():
    with pytest.raises(pr.PopulationRateError):
        pr.fit_hierarchical_rate([])
    footprint = pr.SurveyFootprint(survey="s", area_deg2=100.0, baseline_days=30.0)
    strata = [pr.Stratum(label="s", footprint=footprint, completeness=0.5, detected=1)]
    with pytest.raises(pr.PopulationRateError):
        pr.fit_hierarchical_rate(strata, n_walkers=2)  # below 2 * (2 + 1) = 6


@requires_emcee
def test_hierarchical_rate_fit_to_dict_shape():
    rng = np.random.default_rng(3)
    strata = _synthetic_strata(rng, n_strata=4)
    fit = pr.fit_hierarchical_rate(strata, n_walkers=16, n_steps=200, seed=3)
    payload = fit.to_dict()
    assert "mu_log_rate_median" in payload
    assert payload["n_walkers"] == 16


# ---------------------------------------------------------------------------
# anchor_survey_rate_sweep
# ---------------------------------------------------------------------------

def _by_survey_fixture():
    wide = [SourceRef(survey="wide", object_id=f"w{i}", ra_deg=180.0 + i * 0.001, dec_deg=0.0)
           for i in range(10)]
    narrow = [SourceRef(survey="narrow", object_id=f"n{i}", ra_deg=180.0 + i * 0.001, dec_deg=0.0)
             for i in range(4)]
    return {"wide": wide, "narrow": narrow}


def test_anchor_survey_rate_sweep_uses_each_anchors_own_group_count():
    by_survey = _by_survey_fixture()
    footprints = {"wide": pr.SurveyFootprint(survey="wide", area_deg2=100.0, baseline_days=100.0),
                 "narrow": pr.SurveyFootprint(survey="narrow", area_deg2=100.0, baseline_days=100.0)}
    completeness = {"wide": 0.9, "narrow": 0.9}
    result = pr.anchor_survey_rate_sweep(by_survey, footprints, completeness, ["wide", "narrow"])
    assert result["anchors"]["wide"]["detected"] == 10
    assert result["anchors"]["narrow"]["detected"] == 4
    assert result["anchors"]["wide"]["rate_point"] > result["anchors"]["narrow"]["rate_point"]


def test_anchor_survey_rate_sweep_rejects_bad_inputs():
    by_survey = _by_survey_fixture()
    footprints = {"wide": pr.SurveyFootprint(survey="wide", area_deg2=100.0, baseline_days=100.0)}
    completeness = {"wide": 0.9}
    with pytest.raises(pr.PopulationRateError):
        pr.anchor_survey_rate_sweep(by_survey, footprints, completeness, [])
    with pytest.raises(pr.PopulationRateError):
        pr.anchor_survey_rate_sweep(by_survey, footprints, completeness, ["narrow"])


def test_population_rate_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "population_rate" not in rpc_source
