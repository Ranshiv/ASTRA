"""microlensing_fit.py: optimisation and posterior sampling (backlog
item 15)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import microlensing as ml
from astra import microlensing_fit as mf

emcee = pytest.importorskip("emcee", reason="emcee not installed (research extra)")


def synthetic_event(t0=100.0, tE=25.0, u0=0.15, f_source=5.0, f_blend=2.0,
                    n=250, span=(20, 180), noise_frac=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(*span, n))
    truth = ml.PointLensParams(t0=t0, tE=tE, u0=u0)
    clean = ml.model_flux(t, truth, f_source, f_blend)
    err = np.full_like(clean, noise_frac * np.median(clean))
    noisy = clean + rng.normal(0, err)
    return t, noisy, err, truth


class TestChiSquared:
    def test_zero_for_a_perfect_noiseless_fit(self):
        t, _, _, truth = synthetic_event(noise_frac=0.0)
        flux = ml.model_flux(t, truth, 5.0, 2.0)
        err = np.full_like(flux, 0.01)
        assert mf.chi_squared(t, flux, err, truth) == pytest.approx(0.0, abs=1e-6)

    def test_worse_parameters_increase_chi2(self):
        t, flux, err, truth = synthetic_event(noise_frac=0.0)
        err = np.full_like(flux, 0.01)
        wrong = ml.PointLensParams(t0=truth.t0 + 20.0, tE=truth.tE, u0=truth.u0)
        assert mf.chi_squared(t, flux, err, wrong) > mf.chi_squared(t, flux, err, truth)


class TestFitPointLens:
    def test_recovers_known_parameters_on_noiseless_data(self):
        t, flux, err, truth = synthetic_event(noise_frac=0.0)
        err = np.full_like(flux, 1e-3)
        fit = mf.fit_point_lens(t, flux, err, seed=1)
        assert fit.params.t0 == pytest.approx(truth.t0, abs=0.5)
        assert fit.params.tE == pytest.approx(truth.tE, rel=0.05)
        assert fit.params.u0 == pytest.approx(truth.u0, rel=0.1)
        assert fit.chi2 == pytest.approx(0.0, abs=1e-6)

    def test_recovers_known_parameters_on_noisy_data(self):
        t, flux, err, truth = synthetic_event(noise_frac=0.02, seed=3)
        fit = mf.fit_point_lens(t, flux, err, seed=3)
        assert fit.params.tE == pytest.approx(truth.tE, rel=0.1)
        assert fit.params.u0 == pytest.approx(truth.u0, rel=0.3)

    def test_reduced_chi2_is_near_one_for_correctly_scaled_errors(self):
        t, flux, err, truth = synthetic_event(noise_frac=0.02, n=400, seed=5)
        fit = mf.fit_point_lens(t, flux, err, seed=5)
        assert 0.5 < fit.reduced_chi2 < 2.0

    def test_rejects_too_few_points(self):
        t = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ml.MicrolensingError):
            mf.fit_point_lens(t, t, np.ones_like(t))

    def test_rejects_mismatched_lengths(self):
        t = np.linspace(0, 10, 20)
        with pytest.raises(ml.MicrolensingError):
            mf.fit_point_lens(t, t[:10], np.ones(10))

    def test_rejects_non_positive_errors(self):
        t, flux, err, _ = synthetic_event()
        err = err.copy()
        err[0] = 0.0
        with pytest.raises(ml.MicrolensingError):
            mf.fit_point_lens(t, flux, err)

    def test_proposal_seam_skips_the_global_search(self):
        t, flux, err, truth = synthetic_event(noise_frac=0.0)
        err = np.full_like(flux, 1e-3)
        calls = []

        def proposal(time, observed_flux, observed_err):
            calls.append(1)
            return truth.to_array()

        fit = mf.fit_point_lens(t, flux, err, proposal=proposal)
        assert calls == [1]
        assert "proposal" in fit.note
        assert fit.params.tE == pytest.approx(truth.tE, rel=0.05)


class TestSamplePosterior:
    def test_intervals_contain_the_truth_for_a_well_constrained_event(self):
        t, flux, err, truth = synthetic_event(noise_frac=0.02, n=300, seed=2)
        fit = mf.fit_point_lens(t, flux, err, seed=2)
        posterior = mf.sample_posterior(t, flux, err, fit, n_walkers=24,
                                        n_steps=1200, seed=2)
        for name, value in (("t0", truth.t0), ("tE", truth.tE), ("u0", truth.u0)):
            low, high = posterior.intervals[name]["0.9"]
            assert low <= value <= high

    def test_reports_finite_autocorrelation_times(self):
        t, flux, err, truth = synthetic_event(noise_frac=0.02, seed=4)
        fit = mf.fit_point_lens(t, flux, err, seed=4)
        posterior = mf.sample_posterior(t, flux, err, fit, n_walkers=16,
                                        n_steps=400, seed=4)
        for value in posterior.autocorrelation_time.values():
            assert value is None or np.isfinite(value)

    def test_short_chain_is_flagged_not_converged(self):
        t, flux, err, truth = synthetic_event(noise_frac=0.02, seed=6)
        fit = mf.fit_point_lens(t, flux, err, seed=6)
        posterior = mf.sample_posterior(t, flux, err, fit, n_walkers=12,
                                        n_steps=60, seed=6)
        assert not posterior.converged
        assert "not certified converged" in posterior.note

    def test_sample_shape_matches_walkers_steps_and_burn(self):
        t, flux, err, truth = synthetic_event(noise_frac=0.02, seed=8)
        fit = mf.fit_point_lens(t, flux, err, seed=8)
        posterior = mf.sample_posterior(t, flux, err, fit, n_walkers=10,
                                        n_steps=200, burn_fraction=0.5, seed=8)
        assert posterior.samples.shape == ((200 - 100) * 10, 3)


class TestFitBinaryLens:
    def test_returns_a_well_formed_result(self):
        vbmicrolensing = pytest.importorskip("VBMicrolensing")
        truth = ml.BinaryLensParams(t0=50.0, tE=20.0, u0=0.1, s=1.1, q=0.3,
                                    alpha=1.0, rho=1e-3)
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(20, 80, 150))
        amplification = ml.binary_magnification(t, truth)
        flux = 4.0 * amplification + 1.0
        err = np.full_like(flux, 0.02 * np.median(flux))
        flux = flux + rng.normal(0, err)

        result = mf.fit_binary_lens(t, flux, err, seed=0, maxiter=15)
        assert set(result["params"].keys()) == {"t0", "tE", "u0", "s", "q", "alpha"}
        assert np.isfinite(result["chi2"])
        assert result["n_points"] == 150

    def test_proposal_seam_is_honoured(self):
        pytest.importorskip("VBMicrolensing")
        truth = ml.BinaryLensParams(t0=50.0, tE=20.0, u0=0.1, s=1.1, q=0.3, alpha=1.0)
        rng = np.random.default_rng(1)
        t = np.sort(rng.uniform(20, 80, 60))
        flux = 4.0 * ml.binary_magnification(t, truth) + 1.0
        err = np.full_like(flux, 0.05)

        def proposal(time, observed_flux, observed_err):
            return np.array([50.0, 20.0, 0.1, 1.1, np.log10(0.3), 1.0])

        result = mf.fit_binary_lens(t, flux, err, proposal=proposal)
        assert "proposal" in result["note"]
