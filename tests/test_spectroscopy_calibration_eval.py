"""spectroscopy_calibration_eval.py: redshift-residual aggregation and
synthetic line-recovery study."""

from __future__ import annotations

import numpy as np
import pytest

from astra import spectroscopy_calibration_eval as calibration_eval


def _synthetic_spectrum(rest_wave: float, z: float, *, amplitude: float = 60.0,
                        wave_min: float = 4000.0, wave_max: float = 8200.0,
                        n_points: int = 3000, noise_sigma: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    wave = np.linspace(wave_min, wave_max, n_points)
    observed = rest_wave * (1.0 + z)
    flux = 100.0 + amplitude * np.exp(-0.5 * ((wave - observed) / 3.0) ** 2)
    flux = flux + rng.normal(0.0, noise_sigma, n_points)
    error = np.full(n_points, noise_sigma)
    return wave, flux, error


def _synthetic_multiline_spectrum(rest_lines_and_amplitudes: dict[float, float], z: float, *,
                                  wave_min: float = 4000.0, wave_max: float = 8200.0,
                                  n_points: int = 3000, noise_sigma: float = 1.0, seed: int = 0):
    # A single injected line cannot, in principle, disambiguate WHICH rest
    # line it is (Halpha at z=0.02 and Lyman-alpha at z=4.5 place a line at
    # the same observed wavelength) -- the same reason real spectroscopic
    # pipelines require multiple line coincidences before trusting a
    # redshift. `redshift_residuals` is validated with multiple lines per
    # spectrum for exactly that reason; `line_flux_recovery` below checks
    # single-line WAVELENGTH recovery only, which has no such ambiguity.
    rng = np.random.default_rng(seed)
    wave = np.linspace(wave_min, wave_max, n_points)
    flux = np.full(n_points, 100.0)
    for rest_wave, amplitude in rest_lines_and_amplitudes.items():
        observed = rest_wave * (1.0 + z)
        flux = flux + amplitude * np.exp(-0.5 * ((wave - observed) / 3.0) ** 2)
    flux = flux + rng.normal(0.0, noise_sigma, n_points)
    error = np.full(n_points, noise_sigma)
    return wave, flux, error


class TestRedshiftResiduals:
    def test_near_zero_bias_on_accurately_injected_redshifts(self):
        spectra = []
        for i, z in enumerate([0.02, 0.05, 0.10, 0.15, 0.20]):
            wave, flux, error = _synthetic_multiline_spectrum(
                {6562.79: 80.0, 4861.35: 40.0, 5006.84: 60.0}, z, seed=i)
            spectra.append({"wavelength": wave, "flux": flux, "error": error, "released_z": z})
        result = calibration_eval.redshift_residuals(spectra)
        assert result["n_spectra"] == 5
        assert result["n_unresolved"] == 0
        bias = result["bias"]["parameters"]["z"]
        assert bias["n_compared"] == 5
        assert abs(bias["median_absolute_bias"]) < 0.01

    def test_unresolved_spectra_are_excluded_not_scored_as_zero(self):
        rng = np.random.default_rng(3)
        wave = np.linspace(4000.0, 5000.0, 1000)
        flux = 100.0 + rng.normal(0.0, 1.0, 1000)
        error = np.full(1000, 1.0)
        result = calibration_eval.redshift_residuals(
            [{"wavelength": wave, "flux": flux, "error": error, "released_z": 0.1}])
        assert result["n_unresolved"] == 1
        assert result["bias"]["parameters"]["z"]["n_compared"] == 0

    def test_empty_input(self):
        result = calibration_eval.redshift_residuals([])
        assert result["n_spectra"] == 0
        assert result["n_unresolved"] == 0


class TestLineFluxRecovery:
    def test_recovers_an_injected_line_within_tolerance(self):
        trials = []
        for i, z in enumerate([0.0, 0.05, 0.10]):
            wave, flux, error = _synthetic_spectrum(6562.79, z, seed=10 + i)
            trials.append({"wavelength": wave, "flux": flux, "error": error,
                           "true_wavelength_rest": 6562.79, "true_z": z})
        result = calibration_eval.line_flux_recovery(trials)
        assert result["n_trials"] == 3
        assert result["recall"] == pytest.approx(1.0)
        assert "median_offset_kms" in result

    def test_zero_recall_when_no_line_present(self):
        rng = np.random.default_rng(4)
        wave = np.linspace(4000.0, 5000.0, 1000)
        flux = 100.0 + rng.normal(0.0, 1.0, 1000)
        error = np.full(1000, 1.0)
        trials = [{"wavelength": wave, "flux": flux, "error": error,
                  "true_wavelength_rest": 4500.0, "true_z": 0.0}]
        result = calibration_eval.line_flux_recovery(trials)
        assert result["n_recovered"] == 0
        assert result["recall"] == pytest.approx(0.0)

    def test_empty_input(self):
        result = calibration_eval.line_flux_recovery([])
        assert result["n_trials"] == 0
        import math
        assert math.isnan(result["recall"])


def test_not_referenced_by_rpc():
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "spectroscopy_calibration_eval" not in source
