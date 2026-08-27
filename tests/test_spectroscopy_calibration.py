"""spectroscopy_calibration.py: line-candidate detection, redshift recovery
via line-list cross-correlation, continuum-smoothness diagnostic, and the
combined calibration report."""

from __future__ import annotations

import numpy as np
import pytest

from astra import spectroscopy_calibration as calibration


def _synthetic_spectrum(rest_lines_and_amplitudes: dict[str, float], z: float, *,
                        wave_min: float = 3800.0, wave_max: float = 7000.0,
                        n_points: int = 4000, continuum_level: float = 100.0,
                        line_sigma_angstrom: float = 3.0, noise_sigma: float = 1.0,
                        seed: int = 42):
    rng = np.random.default_rng(seed)
    wave = np.linspace(wave_min, wave_max, n_points)
    flux = np.full(n_points, continuum_level)
    for rest_wave, amplitude in rest_lines_and_amplitudes.items():
        observed = float(rest_wave) * (1.0 + z)
        flux = flux + amplitude * np.exp(-0.5 * ((wave - observed) / line_sigma_angstrom) ** 2)
    flux = flux + rng.normal(0.0, noise_sigma, n_points)
    error = np.full(n_points, noise_sigma)
    return wave, flux, error


class TestFindCandidateLines:
    def test_detects_a_strong_injected_emission_line(self):
        wave, flux, error = _synthetic_spectrum({6562.79: 50.0}, z=0.0)
        candidates = calibration.find_candidate_lines(wave, flux, error)
        assert any(abs(c["wavelength"] - 6562.79) < 5.0 and c["kind"] == "emission"
                  for c in candidates)

    def test_no_candidates_on_pure_noise(self):
        rng = np.random.default_rng(1)
        wave = np.linspace(4000.0, 5000.0, 1000)
        flux = 100.0 + rng.normal(0.0, 1.0, 1000)
        error = np.full(1000, 1.0)
        candidates = calibration.find_candidate_lines(wave, flux, error)
        assert candidates == []

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            calibration.find_candidate_lines([1.0, 2.0, 3.0], [1.0, 2.0], [1.0, 2.0])

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError):
            calibration.find_candidate_lines([1.0, 2.0], [1.0, 2.0], [1.0, 1.0])


class TestIndependentRedshiftFromLines:
    def test_recovers_a_known_redshift_from_multiple_lines(self):
        z_true = 0.10
        wave, flux, error = _synthetic_spectrum(
            {4861.35: 40.0, 4958.91: 30.0, 5006.84: 60.0, 6562.79: 80.0, 6583.45: 25.0},
            z=z_true, wave_min=4000.0, wave_max=7800.0)
        result = calibration.independent_redshift_from_lines(wave, flux, error)
        assert result["z_best"] is not None
        assert result["z_best"] == pytest.approx(z_true, abs=0.002)
        assert result["n_lines_matched"] >= 3

    def test_recovers_redshift_zero(self):
        wave, flux, error = _synthetic_spectrum(
            {4861.35: 40.0, 6562.79: 80.0}, z=0.0, wave_min=4000.0, wave_max=7000.0)
        result = calibration.independent_redshift_from_lines(wave, flux, error)
        assert result["z_best"] == pytest.approx(0.0, abs=0.002)

    def test_returns_none_not_zero_when_no_lines_found(self):
        rng = np.random.default_rng(2)
        wave = np.linspace(4000.0, 5000.0, 1000)
        flux = 100.0 + rng.normal(0.0, 1.0, 1000)
        error = np.full(1000, 1.0)
        result = calibration.independent_redshift_from_lines(wave, flux, error)
        assert result["z_best"] is None
        assert result["n_lines_matched"] == 0
        assert "reason" in result

    def test_custom_rest_line_list_is_honoured(self):
        wave, flux, error = _synthetic_spectrum({5000.0: 50.0}, z=0.2,
                                                 wave_min=4000.0, wave_max=8000.0)
        result = calibration.independent_redshift_from_lines(
            wave, flux, error, rest_lines={"custom": 5000.0})
        assert result["z_best"] == pytest.approx(0.2, abs=0.002)
        assert result["matches"][0]["line"] == "custom"

    def test_rejects_empty_rest_line_dict(self):
        wave, flux, error = _synthetic_spectrum({6562.79: 50.0}, z=0.0)
        with pytest.raises(ValueError):
            calibration.independent_redshift_from_lines(wave, flux, error, rest_lines={})


class TestContinuumSmoothnessResidual:
    def test_low_residual_for_a_genuinely_smooth_continuum(self):
        wave = np.linspace(4000.0, 7000.0, 1000)
        continuum = 100.0 + 0.001 * (wave - 5500.0)
        result = calibration.continuum_smoothness_residual(wave, continuum)
        assert result["median_relative_residual"] < 1e-6

    def test_higher_residual_for_a_discontinuous_step(self):
        wave = np.linspace(4000.0, 7000.0, 1000)
        smooth = 100.0 + 0.001 * (wave - 5500.0)
        stepped = smooth.copy()
        stepped[wave > 5500.0] += 20.0
        smooth_result = calibration.continuum_smoothness_residual(wave, smooth)
        stepped_result = calibration.continuum_smoothness_residual(wave, stepped)
        assert stepped_result["residual_rms"] > smooth_result["residual_rms"]

    def test_rejects_invalid_poly_degree(self):
        wave = np.linspace(4000.0, 5000.0, 100)
        with pytest.raises(ValueError):
            calibration.continuum_smoothness_residual(wave, wave, poly_degree=0)

    def test_rejects_too_few_points_for_degree(self):
        with pytest.raises(ValueError):
            calibration.continuum_smoothness_residual([1.0, 2.0], [1.0, 2.0], poly_degree=3)


class TestCalibrationReport:
    def test_combines_redshift_and_instrument_response(self):
        wave, flux, error = _synthetic_spectrum(
            {4861.35: 40.0, 6562.79: 80.0}, z=0.05, wave_min=4000.0, wave_max=7000.0)
        report = calibration.calibration_report(wave, flux, error)
        assert "redshift" in report and "instrument_response" in report
        assert report["redshift"]["z_best"] == pytest.approx(0.05, abs=0.002)

    def test_reports_residual_against_a_released_redshift(self):
        wave, flux, error = _synthetic_spectrum(
            {4861.35: 40.0, 6562.79: 80.0}, z=0.05, wave_min=4000.0, wave_max=7000.0)
        report = calibration.calibration_report(wave, flux, error, released_z=0.049)
        assert report["redshift"]["released_z"] == pytest.approx(0.049)
        assert report["redshift"]["z_residual"] == pytest.approx(
            report["redshift"]["z_best"] - 0.049, abs=1e-9)

    def test_no_released_z_comparison_when_not_given(self):
        wave, flux, error = _synthetic_spectrum(
            {4861.35: 40.0, 6562.79: 80.0}, z=0.05, wave_min=4000.0, wave_max=7000.0)
        report = calibration.calibration_report(wave, flux, error)
        assert "released_z" not in report["redshift"]


def test_not_referenced_by_rpc():
    """Diagnostic-only discipline: matching every prior roadmap module."""
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "spectroscopy_calibration" not in source
