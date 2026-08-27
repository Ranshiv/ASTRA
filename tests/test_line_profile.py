"""line_profile.py: Voigt/Gaussian forward model shape and validation."""

from __future__ import annotations

import numpy as np
import pytest

from astra import line_profile as lp


class TestLineProfileParams:
    def test_rejects_non_positive_sigma(self):
        with pytest.raises(lp.LineProfileError):
            lp.LineProfileParams(center=5000.0, sigma=0.0, gamma=0.0, amplitude=10.0)

    def test_rejects_negative_gamma(self):
        with pytest.raises(lp.LineProfileError):
            lp.LineProfileParams(center=5000.0, sigma=1.0, gamma=-0.1, amplitude=10.0)

    def test_rejects_non_finite_values(self):
        with pytest.raises(lp.LineProfileError):
            lp.LineProfileParams(center=float("nan"), sigma=1.0, gamma=0.0, amplitude=10.0)

    def test_amplitude_may_be_negative(self):
        params = lp.LineProfileParams(center=5000.0, sigma=1.0, gamma=0.0, amplitude=-10.0)
        assert params.amplitude == -10.0

    def test_round_trips_through_array(self):
        params = lp.LineProfileParams(center=5000.0, sigma=2.0, gamma=0.5, amplitude=15.0)
        restored = lp.LineProfileParams.from_array(params.to_array())
        assert restored == params


class TestVoigtProfile:
    def test_rejects_non_positive_sigma(self):
        with pytest.raises(lp.LineProfileError):
            lp.voigt_profile(np.array([0.0]), sigma=0.0, gamma=1.0)

    def test_rejects_negative_gamma(self):
        with pytest.raises(lp.LineProfileError):
            lp.voigt_profile(np.array([0.0]), sigma=1.0, gamma=-1.0)

    def test_gamma_zero_matches_a_gaussian(self):
        x = np.linspace(-10.0, 10.0, 2001)
        sigma = 1.5
        voigt = lp.voigt_profile(x, sigma=sigma, gamma=0.0)
        gaussian = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        assert np.allclose(voigt, gaussian, atol=1e-8)

    def test_is_symmetric_around_zero(self):
        x = np.linspace(-8.0, 8.0, 801)
        voigt = lp.voigt_profile(x, sigma=1.0, gamma=0.5)
        assert np.allclose(voigt, voigt[::-1], atol=1e-10)

    def test_integrates_to_unit_area(self):
        # A Lorentzian-dominated Voigt (gamma >> sigma) has slowly decaying
        # ~1/x^2 wings, so the integration range must be wide relative to
        # gamma for truncation error to stay below the tolerance here.
        x = np.linspace(-5000.0, 5000.0, 400_001)
        voigt = lp.voigt_profile(x, sigma=1.0, gamma=0.8)
        area = np.trapezoid(voigt, x)
        assert area == pytest.approx(1.0, abs=1e-3)

    def test_peak_decreases_as_gamma_grows_at_fixed_sigma(self):
        x = np.array([0.0])
        narrow = lp.voigt_profile(x, sigma=1.0, gamma=0.1)[0]
        wide = lp.voigt_profile(x, sigma=1.0, gamma=2.0)[0]
        assert wide < narrow


class TestModelFlux:
    def test_recovers_continuum_far_from_the_line(self):
        wave = np.linspace(4000.0, 6000.0, 2000)
        continuum = 100.0
        params = lp.LineProfileParams(center=5000.0, sigma=1.0, gamma=0.0, amplitude=50.0)
        model = lp.model_flux(wave, continuum, params)
        assert model[0] == pytest.approx(100.0, abs=1e-6)
        assert model[-1] == pytest.approx(100.0, abs=1e-6)

    def test_emission_line_rises_above_continuum(self):
        wave = np.linspace(4990.0, 5010.0, 500)
        params = lp.LineProfileParams(center=5000.0, sigma=1.0, gamma=0.0, amplitude=50.0)
        model = lp.model_flux(wave, 100.0, params)
        assert model.max() > 100.0

    def test_absorption_line_dips_below_continuum(self):
        wave = np.linspace(4990.0, 5010.0, 500)
        params = lp.LineProfileParams(center=5000.0, sigma=1.0, gamma=0.0, amplitude=-50.0)
        model = lp.model_flux(wave, 100.0, params)
        assert model.min() < 100.0

    def test_accepts_an_array_continuum(self):
        wave = np.linspace(4990.0, 5010.0, 500)
        continuum = 100.0 + 0.01 * (wave - 5000.0)
        params = lp.LineProfileParams(center=5000.0, sigma=1.0, gamma=0.0, amplitude=50.0)
        model = lp.model_flux(wave, continuum, params)
        assert model.shape == wave.shape


def test_not_referenced_by_rpc():
    """Diagnostic-only discipline: matching every prior roadmap module."""
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "line_profile" not in source
