"""line_profile_fit.py: optimisation and posterior sampling recovery."""

from __future__ import annotations

import numpy as np
import pytest

from astra import line_profile as lp
from astra import line_profile_fit as fitting

emcee = pytest.importorskip("emcee", reason="emcee not installed (research extra)")


def _synthetic_window(params: lp.LineProfileParams, *, continuum: float = 100.0,
                      wave_min: float = 4950.0, wave_max: float = 5050.0,
                      n_points: int = 600, noise_sigma: float = 0.5, seed: int = 42):
    rng = np.random.default_rng(seed)
    wave = np.linspace(wave_min, wave_max, n_points)
    clean = lp.model_flux(wave, continuum, params)
    flux = clean + rng.normal(0.0, noise_sigma, n_points)
    error = np.full(n_points, noise_sigma)
    return wave, flux, error, np.full(n_points, continuum)


class TestSolveLinearAmplitude:
    def test_recovers_a_known_amplitude_at_the_true_shape(self):
        wave = np.linspace(4990.0, 5010.0, 500)
        error = np.full(500, 1.0)
        continuum = np.full(500, 100.0)
        clean = lp.model_flux(wave, continuum, lp.LineProfileParams(
            center=5000.0, sigma=1.0, gamma=0.0, amplitude=40.0))
        amplitude = fitting.solve_linear_amplitude(wave, clean, error, continuum, 5000.0, 1.0, 0.0)
        assert amplitude == pytest.approx(40.0, abs=1e-6)

    def test_returns_zero_for_a_degenerate_profile(self):
        # A profile entirely outside the data has ~zero overlap with it.
        wave = np.linspace(4000.0, 4010.0, 100)
        flux = np.full(100, 100.0)
        error = np.full(100, 1.0)
        continuum = np.full(100, 100.0)
        amplitude = fitting.solve_linear_amplitude(
            wave, flux, error, continuum, center=9000.0, sigma=1.0, gamma=0.0)
        assert amplitude == pytest.approx(0.0, abs=1.0)


class TestDefaultBounds:
    def test_center_defaults_to_the_observed_window(self):
        wave = np.linspace(4000.0, 5000.0, 500)
        (c_lo, c_hi), _, _ = fitting.default_bounds(wave)
        assert c_lo == pytest.approx(4000.0)
        assert c_hi == pytest.approx(5000.0)

    def test_center_hint_narrows_the_window(self):
        wave = np.linspace(4000.0, 5000.0, 500)
        (c_lo, c_hi), _, _ = fitting.default_bounds(
            wave, center_hint=4500.0, window_angstrom=10.0)
        assert c_lo == pytest.approx(4490.0)
        assert c_hi == pytest.approx(4510.0)

    def test_gamma_lower_bound_is_zero(self):
        wave = np.linspace(4000.0, 5000.0, 500)
        _, _, (g_lo, _) = fitting.default_bounds(wave)
        assert g_lo == 0.0


class TestFitLineProfile:
    def test_recovers_a_known_emission_line(self):
        truth = lp.LineProfileParams(center=5000.0, sigma=1.5, gamma=0.0, amplitude=60.0)
        wave, flux, error, continuum = _synthetic_window(truth, seed=1)
        fit = fitting.fit_line_profile(wave, flux, error, continuum)
        assert fit.converged
        assert fit.params.center == pytest.approx(5000.0, abs=0.3)
        assert fit.params.sigma == pytest.approx(1.5, rel=0.3)
        assert fit.params.amplitude == pytest.approx(60.0, rel=0.2)

    def test_recovers_a_known_absorption_line(self):
        truth = lp.LineProfileParams(center=5000.0, sigma=1.5, gamma=0.0, amplitude=-60.0)
        wave, flux, error, continuum = _synthetic_window(truth, seed=2)
        fit = fitting.fit_line_profile(wave, flux, error, continuum)
        assert fit.params.amplitude == pytest.approx(-60.0, rel=0.2)

    def test_rejects_too_few_points(self):
        with pytest.raises(lp.LineProfileError):
            fitting.fit_line_profile([1.0, 2.0], [1.0, 2.0], [1.0, 1.0], [1.0, 1.0])

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(lp.LineProfileError):
            fitting.fit_line_profile([1.0, 2.0, 3.0, 4.0, 5.0],
                                     [1.0, 2.0], [1.0, 1.0], [1.0, 1.0])

    def test_rejects_non_positive_error(self):
        wave = np.linspace(4990.0, 5010.0, 100)
        with pytest.raises(lp.LineProfileError):
            fitting.fit_line_profile(wave, np.full(100, 100.0), np.zeros(100), np.full(100, 100.0))

    def test_proposal_skips_the_global_search(self):
        truth = lp.LineProfileParams(center=5000.0, sigma=1.5, gamma=0.0, amplitude=60.0)
        wave, flux, error, continuum = _synthetic_window(truth, seed=3)

        def proposal(wavelength, flux_, error_, continuum_):
            return np.array([5000.0, 1.5, 0.0])

        fit = fitting.fit_line_profile(wave, flux, error, continuum, proposal=proposal)
        assert "supplied proposal" in fit.note
        assert fit.params.center == pytest.approx(5000.0, abs=0.3)


class TestSamplePosterior:
    def test_reports_convergence_diagnostics(self):
        truth = lp.LineProfileParams(center=5000.0, sigma=1.5, gamma=0.0, amplitude=60.0)
        wave, flux, error, continuum = _synthetic_window(truth, seed=4)
        fit = fitting.fit_line_profile(wave, flux, error, continuum)
        posterior = fitting.sample_posterior(
            wave, flux, error, continuum, fit, n_walkers=24, n_steps=1500, seed=4)
        assert posterior.parameter_names == ("center", "sigma", "gamma")
        assert posterior.samples.shape[1] == 3
        assert "center" in posterior.intervals

    def test_credible_interval_contains_the_true_center(self):
        truth = lp.LineProfileParams(center=5000.0, sigma=1.5, gamma=0.0, amplitude=60.0)
        wave, flux, error, continuum = _synthetic_window(truth, seed=5)
        fit = fitting.fit_line_profile(wave, flux, error, continuum)
        posterior = fitting.sample_posterior(
            wave, flux, error, continuum, fit, n_walkers=24, n_steps=1500, seed=5)
        low, high = posterior.intervals["center"]["0.9"]
        assert low <= 5000.0 <= high


def test_not_referenced_by_rpc():
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "line_profile_fit" not in source
