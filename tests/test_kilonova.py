"""kilonova.py: generalized-Arnett kilonova model shape, validation, and
physical sanity checks."""

from __future__ import annotations

import numpy as np
import pytest

from astra import kilonova as kn


class TestKilonovaParams:
    def test_rejects_non_positive_mass(self):
        with pytest.raises(kn.KilonovaError):
            kn.KilonovaParams(m_ej=0.0, v_ej=0.1, kappa=1.0)

    def test_rejects_velocity_outside_unit_interval(self):
        with pytest.raises(kn.KilonovaError):
            kn.KilonovaParams(m_ej=0.01, v_ej=1.5, kappa=1.0)
        with pytest.raises(kn.KilonovaError):
            kn.KilonovaParams(m_ej=0.01, v_ej=0.0, kappa=1.0)

    def test_rejects_non_positive_opacity(self):
        with pytest.raises(kn.KilonovaError):
            kn.KilonovaParams(m_ej=0.01, v_ej=0.1, kappa=0.0)

    def test_rejects_non_finite_values(self):
        with pytest.raises(kn.KilonovaError):
            kn.KilonovaParams(m_ej=float("nan"), v_ej=0.1, kappa=1.0)


class TestDiffusionTimescale:
    def test_increases_with_mass(self):
        low = kn.KilonovaParams(m_ej=0.001, v_ej=0.1, kappa=1.0)
        high = kn.KilonovaParams(m_ej=0.01, v_ej=0.1, kappa=1.0)
        assert kn.diffusion_timescale_s(high) > kn.diffusion_timescale_s(low)

    def test_increases_with_opacity(self):
        low = kn.KilonovaParams(m_ej=0.01, v_ej=0.1, kappa=0.5)
        high = kn.KilonovaParams(m_ej=0.01, v_ej=0.1, kappa=10.0)
        assert kn.diffusion_timescale_s(high) > kn.diffusion_timescale_s(low)

    def test_decreases_with_velocity(self):
        slow = kn.KilonovaParams(m_ej=0.01, v_ej=0.05, kappa=1.0)
        fast = kn.KilonovaParams(m_ej=0.01, v_ej=0.3, kappa=1.0)
        assert kn.diffusion_timescale_s(fast) < kn.diffusion_timescale_s(slow)


class TestRadioactiveHeatingRate:
    def test_decays_at_late_times(self):
        early = kn.radioactive_heating_rate(np.array([10.0]), m_rp_g=1e31)[0]
        late = kn.radioactive_heating_rate(np.array([1e6]), m_rp_g=1e31)[0]
        assert late < early

    def test_scales_linearly_with_mass(self):
        rate_1x = kn.radioactive_heating_rate(np.array([1e5]), m_rp_g=1e31)[0]
        rate_2x = kn.radioactive_heating_rate(np.array([1e5]), m_rp_g=2e31)[0]
        assert rate_2x == pytest.approx(2.0 * rate_1x, rel=1e-9)

    def test_matches_the_literature_heating_rate_scale_at_one_day(self):
        # ~2e10 erg/s/g at t=1 day is the commonly cited r-process
        # heating-rate scale -- verified against the raw Villar et al.
        # 2017 arXiv source this session (see module docstring).
        rate_per_gram = kn.radioactive_heating_rate(np.array([86400.0]), m_rp_g=1.0)[0]
        assert rate_per_gram == pytest.approx(2e10, rel=0.5)


class TestThermalizationEfficiency:
    def test_starts_near_072_at_t_zero(self):
        eps = kn.thermalization_efficiency(np.array([0.0]))[0]
        assert eps == pytest.approx(0.72, abs=1e-6)

    def test_decreases_monotonically(self):
        times = np.array([0.0, 1.0, 100.0, 1e4, 1e6])
        eps = kn.thermalization_efficiency(times)
        assert np.all(np.diff(eps) <= 0)

    def test_approaches_zero_at_very_late_times(self):
        eps = kn.thermalization_efficiency(np.array([1e10]))[0]
        assert eps < 1e-3


class TestBolometricLuminosity:
    def test_rejects_negative_time(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.1, kappa=1.0)
        with pytest.raises(kn.KilonovaError):
            kn.bolometric_luminosity(np.array([-1.0]), params)

    def test_is_finite_and_positive(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        times = np.array([3600.0, 86400.0, 5 * 86400.0])
        luminosity = kn.bolometric_luminosity(times, params)
        assert np.all(np.isfinite(luminosity))
        assert np.all(luminosity > 0)

    def test_zero_at_time_zero(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        assert kn.bolometric_luminosity(np.array([0.0]), params)[0] == 0.0

    def test_matches_realistic_peak_scale_for_a_small_ejecta_mass(self):
        # A real, physically sensible order-of-magnitude check (not an
        # exact GW170817 reproduction -- see module docstring): a small,
        # single-component ejecta's luminosity near t~1 day should land in
        # the ~1e41-1e42 erg/s range real kilonovae are observed at.
        params = kn.KilonovaParams(m_ej=0.001, v_ej=0.27, kappa=0.5)
        luminosity = kn.bolometric_luminosity(np.array([86400.0]), params)[0]
        assert 1e40 < luminosity < 1e43

    def test_increases_super_linearly_with_mass(self):
        # A real, verified consequence of t_d itself scaling with
        # sqrt(m_ej) -- see module docstring's "found via direct numerical
        # comparison" note. Regression-pins the qualitative behaviour, not
        # an exact exponent.
        low = kn.KilonovaParams(m_ej=0.001, v_ej=0.27, kappa=0.5)
        high = kn.KilonovaParams(m_ej=0.01, v_ej=0.27, kappa=0.5)
        l_low = kn.bolometric_luminosity(np.array([86400.0]), low)[0]
        l_high = kn.bolometric_luminosity(np.array([86400.0]), high)[0]
        mass_ratio = 0.01 / 0.001
        luminosity_ratio = l_high / l_low
        assert luminosity_ratio > mass_ratio  # super-linear, not just linear


class TestPhotosphericTemperatureAndRadius:
    def test_temperature_never_drops_below_the_floor(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.1, kappa=5.0, temperature_floor_k=3000.0)
        times = np.array([3600.0, 86400.0, 10 * 86400.0, 30 * 86400.0])
        temperature = kn.photospheric_temperature_k(times, params)
        assert np.all(temperature >= 3000.0 - 1e-6)

    def test_radius_expands_homologously_before_the_floor(self):
        params = kn.KilonovaParams(m_ej=0.05, v_ej=0.2, kappa=1.0, temperature_floor_k=100.0)
        t = np.array([3600.0])
        radius = kn.photospheric_radius_cm(t, params)
        expected = params.v_ej * kn.SPEED_OF_LIGHT_CM_S * t
        assert radius[0] == pytest.approx(expected[0], rel=1e-6)

    def test_radius_recedes_once_the_floor_is_reached(self):
        # A very high temperature floor forces the receding-photosphere
        # branch at essentially all physically reachable times.
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.1, kappa=1.0, temperature_floor_k=50000.0)
        t = np.array([86400.0])
        radius = kn.photospheric_radius_cm(t, params)
        homologous = params.v_ej * kn.SPEED_OF_LIGHT_CM_S * t
        assert radius[0] < homologous[0]


class TestBlackbodyBandFluxAndMagnitude:
    def test_flux_is_positive_and_finite(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        flux = kn.blackbody_band_flux(np.array([86400.0]), params, 6231.0, distance_mpc=40.0)
        assert np.all(np.isfinite(flux))
        assert np.all(flux > 0)

    def test_flux_decreases_with_distance_as_inverse_square(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        near = kn.blackbody_band_flux(np.array([86400.0]), params, 6231.0, distance_mpc=10.0)[0]
        far = kn.blackbody_band_flux(np.array([86400.0]), params, 6231.0, distance_mpc=20.0)[0]
        assert far == pytest.approx(near / 4.0, rel=1e-6)

    def test_ab_mag_round_trips_a_known_flux(self):
        # A source at the AB zero-point f_nu, converted to f_lambda at a
        # reference wavelength, must map back to AB mag 0.
        wavelength_angstrom = 5000.0
        wavelength_cm = wavelength_angstrom * 1e-8
        f_nu = kn.AB_ZEROPOINT_F_NU_CGS
        f_lambda_per_cm = f_nu * kn.SPEED_OF_LIGHT_CM_S / wavelength_cm ** 2
        f_lambda_per_angstrom = f_lambda_per_cm * 1e-8
        mag = kn.flux_density_to_ab_mag(
            np.array([f_lambda_per_angstrom]), wavelength_angstrom)[0]
        assert mag == pytest.approx(0.0, abs=1e-6)

    def test_fainter_flux_gives_a_larger_magnitude(self):
        bright = kn.flux_density_to_ab_mag(np.array([1e-15]), 6000.0)[0]
        faint = kn.flux_density_to_ab_mag(np.array([1e-18]), 6000.0)[0]
        assert faint > bright


class TestMultiComponentLightCurve:
    def test_rejects_empty_components(self):
        with pytest.raises(kn.KilonovaError):
            kn.multi_component_light_curve(np.array([86400.0]), [], 6231.0)

    def test_equals_the_sum_of_each_component(self):
        blue = kn.KilonovaParams(m_ej=0.005, v_ej=0.27, kappa=0.5)
        red = kn.KilonovaParams(m_ej=0.003, v_ej=0.14, kappa=10.0)
        t = np.array([86400.0])
        combined = kn.multi_component_light_curve(t, [blue, red], 6231.0)
        separate = (kn.blackbody_band_flux(t, blue, 6231.0)
                   + kn.blackbody_band_flux(t, red, 6231.0))
        assert combined[0] == pytest.approx(separate[0], rel=1e-9)


def test_not_referenced_by_rpc():
    """Diagnostic-only discipline: matching every prior roadmap module."""
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "kilonova" not in source
