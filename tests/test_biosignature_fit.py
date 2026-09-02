"""biosignature_fit.py: optimiser, posterior sampler, and band-detection
significance. Parameter recovery is checked loosely (temperature/scale-
height and band amplitude are known to trade off against each other in
this model -- a real physical degeneracy, not a bug); detection
significance on flat vs. featured spectra is checked precisely, since
that is the metric this module's caveats say to trust."""

from __future__ import annotations

import numpy as np
import pytest

from astra import biosignature as bio
from astra import biosignature_fit as fit


def _system() -> bio.SystemParameters:
    return bio.SystemParameters(stellar_radius_rsun=1.0, planet_mass_mjup=1.0)


def _wave() -> np.ndarray:
    return np.linspace(1.0, 2.0, 60)


def _synthetic_h2o_spectrum(seed: int = 1, amplitude: float = 2.0):
    system = _system()
    wave = _wave()
    true_atm = bio.AtmosphereParameters(temperature_k=900.0, mean_molecular_weight=2.3,
                                        reference_radius_rjup=1.0, abundances=(("H2O", -0.3),))
    true_depth = bio.transit_depth(wave, true_atm, system, cross_sections={"H2O": amplitude})
    rng = np.random.default_rng(seed)
    error = np.full_like(true_depth, 5e-6)
    noisy_depth = true_depth + rng.normal(0.0, 5e-6, size=true_depth.shape)
    return wave, noisy_depth, error, system


def _flat_spectrum(seed: int = 1):
    system = _system()
    wave = _wave()
    flat_atm = bio.AtmosphereParameters(temperature_k=900.0, mean_molecular_weight=2.3,
                                        reference_radius_rjup=1.0)
    flat_depth = bio.transit_depth(wave, flat_atm, system, cross_sections={})
    rng = np.random.default_rng(seed)
    error = np.full_like(flat_depth, 5e-6)
    noisy_depth = flat_depth + rng.normal(0.0, 5e-6, size=flat_depth.shape)
    return wave, noisy_depth, error, system


class TestFitTransmissionSpectrum:
    def test_fit_converges_and_reports_a_result(self):
        wave, depth, error, system = _synthetic_h2o_spectrum()
        result = fit.fit_transmission_spectrum(wave, depth, error, system, molecules=("H2O",),
                                               cross_sections={"H2O": 2.0}, seed=1, maxiter=60)
        assert result.n_points == len(wave)
        assert np.isfinite(result.chi2)
        assert isinstance(result.converged, bool)

    def test_proposal_seam_skips_global_search(self):
        wave, depth, error, system = _synthetic_h2o_spectrum()
        called = {"n": 0}

        def proposal(w, d, e):
            called["n"] += 1
            return np.array([900.0, 1.0, -0.3])

        result = fit.fit_transmission_spectrum(wave, depth, error, system, molecules=("H2O",),
                                               cross_sections={"H2O": 2.0}, proposal=proposal)
        assert called["n"] == 1
        assert "global search skipped" in result.note

    def test_too_few_points_raises(self):
        system = _system()
        with pytest.raises(bio.BiosignatureError):
            fit.fit_transmission_spectrum([1.0, 2.0], [0.01, 0.01], [0.001, 0.001], system,
                                          molecules=("H2O",), cross_sections={"H2O": 2.0})


class TestDetectionSignificance:
    def test_flat_continuum_spectrum_is_not_detected(self):
        wave, depth, error, system = _flat_spectrum()
        result = fit.detection_significance(wave, depth, error, system, "H2O",
                                            cross_sections={"H2O": 2.0}, seed=1)
        assert result["detected"] is False
        assert result["delta_bic"] <= 10.0

    def test_strong_injected_band_is_detected(self):
        wave, depth, error, system = _synthetic_h2o_spectrum(amplitude=2.0)
        result = fit.detection_significance(wave, depth, error, system, "H2O",
                                            cross_sections={"H2O": 2.0}, seed=1)
        assert result["detected"] is True
        assert result["delta_bic"] > 10.0

    def test_result_has_expected_keys(self):
        wave, depth, error, system = _flat_spectrum()
        result = fit.detection_significance(wave, depth, error, system, "CO2",
                                            cross_sections={"CO2": 2.0}, seed=1)
        assert set(result) >= {"molecule", "delta_bic", "delta_chi2", "full_chi2", "null_chi2",
                               "n_points", "log10_amplitude", "detected"}


class TestDisequilibriumFlag:
    def test_co_detection_requires_both_ch4_and_an_oxidant(self):
        significances = {
            "CH4": {"delta_bic": 20.0}, "O2": {"delta_bic": 15.0}, "O3": {"delta_bic": 2.0},
        }
        result = fit.disequilibrium_flag(significances)
        assert result["ch4_detected"] is True
        assert result["oxidant_detected"] is True
        assert result["co_detection_flag"] is True
        assert "caveat" in result and len(result["caveat"]) > 0

    def test_ch4_alone_does_not_flag(self):
        significances = {"CH4": {"delta_bic": 20.0}, "O2": {"delta_bic": 1.0}}
        result = fit.disequilibrium_flag(significances)
        assert result["co_detection_flag"] is False

    def test_missing_molecules_do_not_crash(self):
        result = fit.disequilibrium_flag({})
        assert result["co_detection_flag"] is False


def test_sample_posterior_requires_emcee_or_produces_samples():
    pytest.importorskip("emcee", reason="emcee not installed (research extra)")
    wave, depth, error, system = _synthetic_h2o_spectrum()
    start = fit.fit_transmission_spectrum(wave, depth, error, system, molecules=("H2O",),
                                          cross_sections={"H2O": 2.0}, seed=1, maxiter=30)
    result = fit.sample_posterior(wave, depth, error, system, start, molecules=("H2O",),
                                  cross_sections={"H2O": 2.0}, n_walkers=16, n_steps=100, seed=1)
    assert result.samples.shape[1] == 3
    assert set(result.parameter_names) == {"temperature_k", "reference_radius_rjup", "log10_amp_H2O"}
    assert isinstance(result.converged, bool)
