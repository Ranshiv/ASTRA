"""asteroseismology.py: Kjeldsen & Bedding scaling relations and the
pure-numpy power-spectrum measurement path."""

from __future__ import annotations

import numpy as np
import pytest

from astra import asteroseismology as ast


class TestScalingRelations:
    def test_solar_input_gives_exact_solar_numax_and_dnu(self):
        numax = ast.predict_numax(1.0, 1.0, ast.TEFF_SUN_K)
        dnu = ast.predict_delta_nu(1.0, 1.0)
        assert numax == pytest.approx(ast.NUMAX_SUN_UHZ, abs=1e-6)
        assert dnu == pytest.approx(ast.DNU_SUN_UHZ, abs=1e-6)

    def test_solar_round_trip_inversion_is_exact(self):
        seismic = ast.SeismicParameters(numax_uhz=ast.NUMAX_SUN_UHZ, delta_nu_uhz=ast.DNU_SUN_UHZ,
                                        teff_k=ast.TEFF_SUN_K)
        solution = ast.solve_scaling_relations(seismic)
        assert solution.radius_rsun == pytest.approx(1.0, abs=1e-9)
        assert solution.mass_msun == pytest.approx(1.0, abs=1e-9)

    def test_red_giant_round_trip_is_exact(self):
        m, r, teff = 1.2, 10.0, 4800.0
        numax = ast.predict_numax(m, r, teff)
        dnu = ast.predict_delta_nu(m, r)
        seismic = ast.SeismicParameters(numax_uhz=numax, delta_nu_uhz=dnu, teff_k=teff)
        solution = ast.solve_scaling_relations(seismic)
        assert solution.radius_rsun == pytest.approx(r, rel=1e-6)
        assert solution.mass_msun == pytest.approx(m, rel=1e-6)

    def test_equal_mean_density_gives_equal_dnu(self):
        # Dnu ~ sqrt(rho/rho_sun): two stars with equal mean density
        # (M/R^3 constant) must give equal Dnu regardless of absolute scale.
        dnu_a = ast.predict_delta_nu(1.0, 1.0)
        dnu_b = ast.predict_delta_nu(8.0, 2.0)  # same M/R^3 = 1
        assert dnu_a == pytest.approx(dnu_b, rel=1e-9)

    def test_error_propagation_present_when_errors_supplied(self):
        seismic = ast.SeismicParameters(numax_uhz=ast.NUMAX_SUN_UHZ, delta_nu_uhz=ast.DNU_SUN_UHZ,
                                        teff_k=ast.TEFF_SUN_K, numax_uhz_error=30.0,
                                        delta_nu_uhz_error=0.1, teff_k_error=50.0)
        solution = ast.solve_scaling_relations(seismic)
        assert solution.radius_rsun_error is not None
        assert solution.mass_msun_error is not None
        assert solution.radius_rsun_error > 0

    def test_error_propagation_absent_without_all_three_errors(self):
        seismic = ast.SeismicParameters(numax_uhz=ast.NUMAX_SUN_UHZ, delta_nu_uhz=ast.DNU_SUN_UHZ,
                                        teff_k=ast.TEFF_SUN_K, numax_uhz_error=30.0)
        solution = ast.solve_scaling_relations(seismic)
        assert solution.radius_rsun_error is None

    def test_non_positive_inputs_raise(self):
        with pytest.raises(ast.AsteroseismologyError):
            ast.SeismicParameters(numax_uhz=0.0, delta_nu_uhz=1.0, teff_k=5777.0)
        with pytest.raises(ast.AsteroseismologyError):
            ast.predict_numax(-1.0, 1.0, 5777.0)
        with pytest.raises(ast.AsteroseismologyError):
            ast.predict_delta_nu(1.0, 0.0)


class TestEnvelopeWindow:
    def test_returns_ordered_bounds(self):
        lo, hi = ast.envelope_window(3090.0)
        assert lo < 3090.0 < hi

    def test_non_positive_numax_raises(self):
        with pytest.raises(ast.AsteroseismologyError):
            ast.envelope_window(0.0)


def _synthetic_oscillating_flux(numax_uhz: float, *, n: int = 20000, dt_days: float = 2.0 / 1440.0,
                                noise_sigma: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    dnu_true = ast.STELLO_DNU_COEFF * numax_uhz ** ast.STELLO_DNU_EXPONENT
    time = np.arange(n) * dt_days
    lo, hi = ast.envelope_window(numax_uhz)
    freqs_uhz = np.arange(lo, hi, dnu_true)
    sigma = (hi - lo) / 2.355
    flux = np.zeros(n)
    for f in freqs_uhz:
        amplitude = np.exp(-0.5 * ((f - numax_uhz) / sigma) ** 2)
        freq_per_day = f / ast.UHZ_PER_DAY_INVERSE
        phase = rng.uniform(0.0, 2.0 * np.pi)
        flux += amplitude * np.sin(2.0 * np.pi * freq_per_day * time + phase)
    flux += rng.normal(0.0, noise_sigma, n)
    return time, flux, dnu_true


class TestMeasure:
    def test_recovers_injected_numax_and_dnu(self):
        numax_true = 1200.0
        time, flux, dnu_true = _synthetic_oscillating_flux(numax_true)
        result = ast.measure(time, flux, teff_k=5777.0)
        assert result["quality"] == "usable"
        assert result["numax_uhz"] == pytest.approx(numax_true, rel=0.05)
        assert result["delta_nu_uhz"] == pytest.approx(dnu_true, rel=0.05)
        assert result["solution"] is not None

    def test_white_noise_gives_no_detection(self):
        rng = np.random.default_rng(1)
        n = 5000
        time = np.arange(n) * (2.0 / 1440.0)
        flux = rng.normal(0.0, 1.0, n)
        result = ast.measure(time, flux)
        assert result["quality"] == "insufficient"
        assert result["numax_uhz"] is None

    def test_constant_flux_gives_insufficient_not_nan(self):
        n = 5000
        time = np.arange(n) * (2.0 / 1440.0)
        flux = np.ones(n)
        result = ast.measure(time, flux)
        assert result["quality"] == "insufficient"
        assert result["numax_uhz"] is None

    def test_too_few_points_gives_insufficient(self):
        time = np.arange(50) * (2.0 / 1440.0)
        flux = np.sin(time)
        result = ast.measure(time, flux)
        assert result["quality"] == "insufficient"

    def test_result_is_json_serializable(self):
        import json
        numax_true = 1200.0
        time, flux, _ = _synthetic_oscillating_flux(numax_true)
        result = ast.measure(time, flux, teff_k=5777.0)
        json.dumps(result)  # must not raise


class TestEchelle:
    def test_folded_frequencies_are_within_delta_nu(self):
        frequency = np.linspace(1000.0, 1400.0, 400)
        power = np.ones_like(frequency)
        result = ast.echelle(frequency, power, delta_nu_uhz=60.0)
        folded = np.array(result["folded_uhz"])
        assert np.all(folded >= 0.0)
        assert np.all(folded < 60.0)

    def test_non_positive_delta_nu_raises(self):
        with pytest.raises(ast.AsteroseismologyError):
            ast.echelle(np.array([1.0]), np.array([1.0]), delta_nu_uhz=0.0)
