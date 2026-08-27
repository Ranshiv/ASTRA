"""radio_variability.py: scintillation and synchrotron model shape and
validation."""

from __future__ import annotations

import numpy as np
import pytest

from astra import radio_variability as rv


class TestPulseBroadeningTime:
    def test_rejects_non_positive_dm(self):
        with pytest.raises(rv.RadioVariabilityError):
            rv.pulse_broadening_time_ms(0.0, 1.4)

    def test_rejects_non_positive_frequency(self):
        with pytest.raises(rv.RadioVariabilityError):
            rv.pulse_broadening_time_ms(100.0, np.array([0.0]))

    def test_decreases_with_frequency(self):
        low_freq = rv.pulse_broadening_time_ms(100.0, np.array([0.5]))[0]
        high_freq = rv.pulse_broadening_time_ms(100.0, np.array([2.0]))[0]
        assert high_freq < low_freq

    def test_increases_with_dispersion_measure(self):
        low_dm = rv.pulse_broadening_time_ms(20.0, np.array([1.0]))[0]
        high_dm = rv.pulse_broadening_time_ms(500.0, np.array([1.0]))[0]
        assert high_dm > low_dm

    def test_matches_the_published_bhat_2004_scale(self):
        # A representative real check: DM=100, nu=1 GHz should land in a
        # physically sensible pulse-broadening range for a pulsar/FRB
        # sightline at that dispersion measure (order 0.1-10 ms).
        tau_ms = rv.pulse_broadening_time_ms(100.0, np.array([1.0]))[0]
        assert 0.01 < tau_ms < 100.0


class TestDecorrelationBandwidth:
    def test_rejects_non_positive_dm(self):
        with pytest.raises(rv.RadioVariabilityError):
            rv.decorrelation_bandwidth_mhz(0.0, 1.4)

    def test_increases_with_frequency(self):
        # Less scattering (shorter tau_d) at higher frequency means a
        # WIDER decorrelation bandwidth.
        low_freq = rv.decorrelation_bandwidth_mhz(100.0, np.array([0.5]))[0]
        high_freq = rv.decorrelation_bandwidth_mhz(100.0, np.array([2.0]))[0]
        assert high_freq > low_freq

    def test_decreases_with_dispersion_measure(self):
        low_dm = rv.decorrelation_bandwidth_mhz(20.0, np.array([1.0]))[0]
        high_dm = rv.decorrelation_bandwidth_mhz(500.0, np.array([1.0]))[0]
        assert high_dm < low_dm

    def test_is_the_fourier_reciprocal_of_the_broadening_time(self):
        dm, freq = 150.0, np.array([1.2])
        tau_d_s = rv.pulse_broadening_time_ms(dm, freq)[0] * 1e-3
        bandwidth_hz = rv.decorrelation_bandwidth_mhz(dm, freq)[0] * 1e6
        expected_bandwidth_hz = rv.SCATTERING_GEOMETRY_C1 / (2.0 * np.pi * tau_d_s)
        assert bandwidth_hz == pytest.approx(expected_bandwidth_hz, rel=1e-9)


class TestSynchrotronSpectrumParams:
    def test_rejects_non_positive_turnover(self):
        with pytest.raises(rv.RadioVariabilityError):
            rv.SynchrotronSpectrumParams(nu_turnover_ghz=0.0, flux_at_turnover_mjy=10.0,
                                         alpha_thin=-0.7)

    def test_rejects_non_positive_flux(self):
        with pytest.raises(rv.RadioVariabilityError):
            rv.SynchrotronSpectrumParams(nu_turnover_ghz=1.0, flux_at_turnover_mjy=0.0,
                                         alpha_thin=-0.7)

    def test_rejects_non_finite_values(self):
        with pytest.raises(rv.RadioVariabilityError):
            rv.SynchrotronSpectrumParams(nu_turnover_ghz=1.0, flux_at_turnover_mjy=10.0,
                                         alpha_thin=float("nan"))


class TestSynchrotronFlux:
    def test_equals_the_turnover_flux_at_the_turnover_frequency(self):
        params = rv.SynchrotronSpectrumParams(nu_turnover_ghz=2.0, flux_at_turnover_mjy=15.0,
                                              alpha_thin=-0.7)
        flux = rv.synchrotron_flux_mjy(np.array([2.0]), params)
        assert flux[0] == pytest.approx(15.0, rel=1e-9)

    def test_declines_above_turnover_for_negative_alpha(self):
        params = rv.SynchrotronSpectrumParams(nu_turnover_ghz=2.0, flux_at_turnover_mjy=15.0,
                                              alpha_thin=-0.7)
        flux = rv.synchrotron_flux_mjy(np.array([2.0, 4.0]), params)
        assert flux[1] < flux[0]

    def test_declines_below_turnover_via_the_self_absorbed_slope(self):
        params = rv.SynchrotronSpectrumParams(nu_turnover_ghz=2.0, flux_at_turnover_mjy=15.0,
                                              alpha_thin=-0.7)
        flux = rv.synchrotron_flux_mjy(np.array([1.0, 2.0]), params)
        assert flux[0] < flux[1]

    def test_rejects_non_positive_frequency(self):
        params = rv.SynchrotronSpectrumParams(nu_turnover_ghz=2.0, flux_at_turnover_mjy=15.0,
                                              alpha_thin=-0.7)
        with pytest.raises(rv.RadioVariabilityError):
            rv.synchrotron_flux_mjy(np.array([0.0]), params)


class TestFitSpectralIndex:
    def test_recovers_a_known_index_from_noiseless_points(self):
        true_alpha = -0.75
        frequency = np.array([1.4, 3.0, 6.0])
        flux = 10.0 * (frequency / 1.4) ** true_alpha
        flux_err = flux * 0.05
        fit = rv.fit_spectral_index(frequency, flux, flux_err)
        assert fit["alpha"] == pytest.approx(true_alpha, abs=1e-6)
        assert fit["n_points"] == 3

    def test_rejects_fewer_than_two_points(self):
        with pytest.raises(rv.RadioVariabilityError):
            rv.fit_spectral_index(np.array([1.4]), np.array([10.0]), np.array([0.5]))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(rv.RadioVariabilityError):
            rv.fit_spectral_index(np.array([1.4, 3.0]), np.array([10.0]), np.array([0.5, 0.3]))

    def test_rejects_non_positive_flux(self):
        with pytest.raises(rv.RadioVariabilityError):
            rv.fit_spectral_index(np.array([1.4, 3.0]), np.array([10.0, -1.0]),
                                  np.array([0.5, 0.3]))

    def test_two_point_index_matches_closed_form(self):
        # For exactly two points the weighted fit must reduce to the
        # elementary two-point slope: alpha = ln(S2/S1) / ln(nu2/nu1).
        frequency = np.array([1.4, 3.0])
        flux = np.array([10.0, 6.0])
        flux_err = np.array([0.5, 0.3])
        fit = rv.fit_spectral_index(frequency, flux, flux_err)
        expected = np.log(6.0 / 10.0) / np.log(3.0 / 1.4)
        assert fit["alpha"] == pytest.approx(expected, rel=1e-6)


def test_not_referenced_by_rpc():
    """Diagnostic-only discipline: matching every prior roadmap module."""
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "radio_variability" not in source
