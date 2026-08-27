"""SIE lens equation: deflection/potential consistency, image solving,
Fermat time delay, model fitting, and the KiDS catalogue cross-check
(roadmap item 29). All numerical checks are against synthetic ground
truth or internal self-consistency; no network in the offline tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from astra import netclient, rpc
from astra.strong_lens import (
    KIDS_LENS_CATALOG, ExternalShear, SIELens, StrongLensError, deflection, fermat_potential,
    fit_lens_model, lens_equation_residual, lensing_potential, magnification,
    query_kids_strong_lens_catalog, shear_deflection, shear_potential, solve_image_positions,
    time_delay_seconds, total_deflection, total_potential,
)
from astra.tap import parse_votable


class TestSIELens:
    def test_rejects_non_positive_theta_e(self):
        with pytest.raises(StrongLensError):
            SIELens(theta_e=0.0, axis_ratio=0.8)

    def test_rejects_axis_ratio_out_of_range(self):
        with pytest.raises(StrongLensError):
            SIELens(theta_e=1.0, axis_ratio=1.5)
        with pytest.raises(StrongLensError):
            SIELens(theta_e=1.0, axis_ratio=0.0)


class TestDeflectionSymmetry:
    def test_circular_case_points_radially(self):
        lens = SIELens(theta_e=1.0, axis_ratio=1.0)
        theta = np.array([[2.0, 0.0], [0.0, 3.0]])
        alpha = deflection(lens, theta)
        # SIS deflection has constant magnitude theta_e and points along theta.
        assert np.linalg.norm(alpha[0]) == pytest.approx(1.0, rel=1e-6)
        assert alpha[0, 1] == pytest.approx(0.0, abs=1e-9)
        assert alpha[1, 0] == pytest.approx(0.0, abs=1e-9)

    def test_elliptical_case_matches_circular_at_q_near_one(self):
        circular = SIELens(theta_e=1.0, axis_ratio=1.0)
        nearly_circular = SIELens(theta_e=1.0, axis_ratio=0.999999)
        theta = np.array([[1.5, 0.7]])
        assert deflection(circular, theta) == pytest.approx(deflection(nearly_circular, theta), abs=1e-4)


class TestPotentialConsistency:
    def test_gradient_of_potential_equals_deflection(self):
        # psi(theta) = theta . alpha(theta) is claimed exact for a
        # 0-homogeneous deflection field (this module's own docstring
        # derivation) -- verify it against alpha via a numerical gradient,
        # not just by construction.
        lens = SIELens(theta_e=1.2, axis_ratio=0.7, position_angle=0.3)
        theta0 = np.array([1.1, -0.6])
        eps = 1e-6
        grad = np.array([
            (lensing_potential(lens, (theta0 + [eps, 0]).reshape(1, 2))[0]
             - lensing_potential(lens, (theta0 - [eps, 0]).reshape(1, 2))[0]) / (2 * eps),
            (lensing_potential(lens, (theta0 + [0, eps]).reshape(1, 2))[0]
             - lensing_potential(lens, (theta0 - [0, eps]).reshape(1, 2))[0]) / (2 * eps),
        ])
        analytic = deflection(lens, theta0.reshape(1, 2))
        assert grad == pytest.approx(analytic, abs=1e-4)


class TestSolveImagePositions:
    def test_every_solved_image_satisfies_the_lens_equation(self):
        lens = SIELens(theta_e=1.0, axis_ratio=0.7, position_angle=0.4)
        beta = np.array([0.05, -0.03])
        images = solve_image_positions(lens, beta)
        assert len(images) >= 2
        for image in images:
            residual = lens_equation_residual(lens, image.reshape(1, 2), beta)
            assert np.max(np.abs(residual)) < 1e-6

    def test_source_far_outside_caustic_gives_one_image(self):
        lens = SIELens(theta_e=1.0, axis_ratio=0.8)
        beta = np.array([4.0, 0.0])
        images = solve_image_positions(lens, beta)
        assert len(images) == 1

    def test_source_at_center_of_circular_lens_gives_an_einstein_ring_seed(self):
        # A point source exactly on-axis for a circular lens is a
        # degenerate ring, not distinct point images -- the solver should
        # not crash and should find at least one self-consistent root.
        lens = SIELens(theta_e=1.0, axis_ratio=1.0)
        images = solve_image_positions(lens, np.array([0.0, 0.0]))
        assert len(images) >= 1


class TestFermatPotentialAndTimeDelay:
    def test_time_delay_is_zero_between_identical_images(self):
        lens = SIELens(theta_e=1.0, axis_ratio=0.8)
        beta = np.array([0.05, 0.0])
        theta = np.array([1.0, 0.2])
        delay = time_delay_seconds(lens, theta, theta, beta, z_lens=0.5,
                                   d_l_mpc=1000.0, d_s_mpc=1500.0, d_ls_mpc=800.0)
        assert delay == pytest.approx(0.0, abs=1e-9)

    def test_time_delay_between_real_images_is_nonzero_and_finite(self):
        lens = SIELens(theta_e=1.0, axis_ratio=0.7, position_angle=0.2)
        beta = np.array([0.06, -0.04])
        images = solve_image_positions(lens, beta)
        assert len(images) >= 2
        delay = time_delay_seconds(lens, images[0], images[1], beta, z_lens=0.5,
                                   d_l_mpc=1000.0, d_s_mpc=1500.0, d_ls_mpc=800.0)
        assert np.isfinite(delay)
        assert delay != 0.0


class TestFitLensModel:
    def test_recovers_injected_parameters_from_noiseless_images(self):
        true_lens = SIELens(theta_e=1.1, axis_ratio=0.65, position_angle=0.35)
        beta = np.array([0.05, -0.02])
        images = solve_image_positions(true_lens, beta)
        assert len(images) >= 3
        fit = fit_lens_model(images, initial_theta_e=1.0)
        assert fit["converged"]
        assert fit["theta_e"] == pytest.approx(true_lens.theta_e, rel=0.02)
        assert fit["axis_ratio"] == pytest.approx(true_lens.axis_ratio, rel=0.05)
        assert fit["residual_rms"] < 1e-4

    def test_requires_at_least_two_images(self):
        with pytest.raises(Exception):
            fit_lens_model([np.array([1.0, 0.0])], initial_theta_e=1.0)


class TestExternalShear:
    def test_shear_potential_gradient_matches_shear_deflection(self):
        # Same numerical-gradient discipline as the SIE potential check
        # above, but for shear's own degree-1-homogeneous relation
        # (psi = theta.alpha/2, not theta.alpha).
        shear = ExternalShear(gamma1=0.05, gamma2=-0.03)
        theta0 = np.array([0.8, -0.4])
        eps = 1e-6
        grad = np.array([
            (shear_potential(shear, (theta0 + [eps, 0]).reshape(1, 2))[0]
             - shear_potential(shear, (theta0 - [eps, 0]).reshape(1, 2))[0]) / (2 * eps),
            (shear_potential(shear, (theta0 + [0, eps]).reshape(1, 2))[0]
             - shear_potential(shear, (theta0 - [0, eps]).reshape(1, 2))[0]) / (2 * eps),
        ])
        analytic = shear_deflection(shear, theta0.reshape(1, 2))
        assert grad == pytest.approx(analytic, abs=1e-4)

    def test_zero_shear_leaves_total_deflection_unchanged(self):
        lens = SIELens(theta_e=1.0, axis_ratio=0.7, position_angle=0.2)
        theta = np.array([[1.2, -0.5]])
        with_zero_shear = total_deflection(lens, theta, ExternalShear())
        without_shear = total_deflection(lens, theta, None)
        assert with_zero_shear == pytest.approx(without_shear)

    def test_zero_shear_leaves_total_potential_unchanged(self):
        lens = SIELens(theta_e=1.0, axis_ratio=0.7, position_angle=0.2)
        theta = np.array([[1.2, -0.5]])
        assert total_potential(lens, theta, ExternalShear()) == pytest.approx(
            total_potential(lens, theta, None))

    def test_solve_image_positions_accepts_shear(self):
        lens = SIELens(theta_e=1.0, axis_ratio=0.75, position_angle=0.1)
        shear = ExternalShear(gamma1=0.03, gamma2=0.02)
        beta = np.array([0.05, -0.02])
        images = solve_image_positions(lens, beta, shear=shear)
        assert len(images) >= 1
        from astra.strong_lens import total_deflection as _td
        for image in images:
            theta = image.reshape(1, 2)
            alpha = np.atleast_2d(_td(lens, theta, shear))
            residual = beta - (theta - alpha)
            assert np.max(np.abs(residual)) < 1e-6


class TestMagnification:
    def test_is_finite_and_nonzero_away_from_the_critical_curve(self):
        lens = SIELens(theta_e=1.0, axis_ratio=0.8, position_angle=0.0)
        mu = magnification(lens, np.array([2.0, 0.5]))
        assert math.isfinite(mu)
        assert mu != 0.0

    def test_magnification_decreases_farther_from_the_lens(self):
        lens = SIELens(theta_e=1.0, axis_ratio=1.0)
        mu_near = abs(magnification(lens, np.array([1.2, 0.0])))
        mu_far = abs(magnification(lens, np.array([4.0, 0.0])))
        assert mu_near > mu_far


class TestFitLensModelWithFluxRatios:
    def test_rejects_mismatched_flux_ratio_count(self):
        with pytest.raises(StrongLensError):
            fit_lens_model([np.array([1.0, 0.0]), np.array([-1.0, 0.0])],
                           initial_theta_e=1.0, observed_flux_ratios=[1.0, 2.0])

    def test_two_image_fit_converges_with_a_consistent_flux_ratio(self):
        # Build a real double (source outside the caustic -> 2 images),
        # compute the TRUE model's own flux ratio, and confirm the fit
        # recovers a consistent lens using that ratio as the extra
        # constraint a bare 2-image position fit cannot provide.
        true_lens = SIELens(theta_e=1.0, axis_ratio=0.8, position_angle=0.1)
        beta = np.array([0.6, 0.0])  # outside the caustic but not too far -> exactly 2 images
        images = solve_image_positions(true_lens, beta)
        assert len(images) == 2
        mu0 = magnification(true_lens, images[0])
        mu1 = magnification(true_lens, images[1])
        true_ratio = mu1 / mu0

        fit = fit_lens_model(images, initial_theta_e=1.0, observed_flux_ratios=[true_ratio])
        assert fit["converged"]
        assert fit["residual_rms"] < 1e-3


class TestQueryKidsStrongLensCatalog:
    def test_parses_a_real_matching_row(self, monkeypatch):
        class _FakeResponse:
            text = (
                '<?xml version="1.0"?><VOTABLE><RESOURCE><TABLE>'
                '<FIELD name="KiDSID"/><FIELD name="zphot"/><FIELD name="zspec"/>'
                '<DATA><TABLEDATA><TR><TD>KIDSJ1200-0000</TD><TD>0.45</TD><TD>0.44</TD></TR>'
                '</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>'
            )
            headers = {"Content-Type": "application/x-votable+xml"}

        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse())
        result = query_kids_strong_lens_catalog(180.0, 0.0)
        assert result["kids_id"] == "KIDSJ1200-0000"
        assert result["z_phot"] == pytest.approx(0.45)

    def test_returns_none_when_no_match(self, monkeypatch):
        class _FakeResponse:
            text = ('<?xml version="1.0"?><VOTABLE><RESOURCE><TABLE>'
                   '<FIELD name="KiDSID"/><DATA><TABLEDATA></TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>')
            headers = {"Content-Type": "application/x-votable+xml"}

        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse())
        assert query_kids_strong_lens_catalog(180.0, 0.0) is None

    def test_uses_the_vizier_provider(self, monkeypatch):
        captured: dict = {}

        class _FakeResponse:
            text = ('<?xml version="1.0"?><VOTABLE><RESOURCE><TABLE>'
                   '<FIELD name="KiDSID"/><DATA><TABLEDATA></TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>')
            headers = {"Content-Type": "application/x-votable+xml"}

        def fake_get(url, params, timeout, provider):
            captured["provider"] = provider
            captured["params"] = params
            return _FakeResponse()

        monkeypatch.setattr(netclient, "get", fake_get)
        query_kids_strong_lens_catalog(180.0, 0.0)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == KIDS_LENS_CATALOG


class TestEvaluateMultiSurveyCoverageReal:
    def test_handles_an_empty_candidate_list(self):
        from astra.strong_lens import evaluate_multi_survey_coverage_real

        result = evaluate_multi_survey_coverage_real([])
        assert result["n_candidates"] == 0
        assert result["des_coverage_fraction"] is None


class TestNotWiredIntoRpc:
    def test_strong_lens_is_not_referenced_by_rpc(self):
        import inspect

        source = inspect.getsource(rpc)
        assert "strong_lens" not in source


@pytest.mark.live
class TestKidsLensCatalogLive:
    """Confirmed live this session (2026-08-25): VizieR hosts the real
    "Strong lenses KiDS DR4" catalogue (`J/A+A/688/A34`, Grespan et al.
    2024)."""

    def test_cone_search_reaches_a_real_service(self):
        # A generic sky position with no guaranteed match -- confirms the
        # request/response contract, not a specific candidate.
        result = query_kids_strong_lens_catalog(180.0, 0.0, radius_arcsec=3600.0)
        assert result is None or isinstance(result.get("kids_id"), str)


def _fetch_real_kids_lens_candidates(limit: int = 300) -> list[dict]:
    """The real, population-scale KiDS DR4 candidate list (564 real
    entries confirmed live this session) -- fetched via VizieR's plain
    (non-cone-search) VOTable endpoint, since the SCS variant `query_kids_
    strong_lens_catalog` uses requires a position per query and cannot
    dump the whole real catalogue. This endpoint's real response has 6
    `<TABLE>` blocks in one document, which `tap.parse_votable` used to
    mishandle (a real bug, now fixed -- see `tests/test_tap.py`'s own
    multi-table regression tests and `docs/DEFERRED.txt`); this helper
    now uses that fixed shared parser directly instead of the plain-regex
    workaround an earlier round of this session used before the fix
    landed.
    """
    from astra.tap import parse_votable

    response = netclient.get(
        "https://vizier.cds.unistra.fr/viz-bin/votable",
        {"-source": "J/A+A/688/A34", "-out.max": limit, "-out": "KiDSID,RAJ2000,DEJ2000,zphot"},
        timeout=60, provider="vizier",
    )
    candidates = []
    for row in parse_votable(response.text, limit):
        kidsid, ra, dec = row.get("KiDSID"), row.get("RAJ2000"), row.get("DEJ2000")
        if kidsid is None or ra is None or dec is None:
            continue
        try:
            candidates.append({"kidsid": kidsid, "ra": float(ra), "dec": float(dec)})
        except (TypeError, ValueError):
            continue
    return candidates


@pytest.mark.live
class TestEvaluateMultiSurveyCoverageRealLive:
    """The population-scale real study `DEFERRED.txt` records for this
    item, run live this session against real KiDS DR4 strong-lens
    candidates (564 real entries, fetched live) cross-matched against
    real DES/Pan-STARRS connectors."""

    def test_real_coverage_study_runs_against_the_real_catalog(self):
        from astra.strong_lens import evaluate_multi_survey_coverage_real

        candidates = _fetch_real_kids_lens_candidates(limit=50)
        assert len(candidates) >= 20
        result = evaluate_multi_survey_coverage_real(candidates)
        assert result["n_candidates"] == len(candidates)
        assert 0.0 <= result["des_coverage_fraction"] <= 1.0
        assert 0.0 <= result["panstarrs_coverage_fraction"] <= 1.0
