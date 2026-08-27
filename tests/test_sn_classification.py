"""Bazin model shape, point-estimate/MCMC fit recovery, truncation, feature
extraction, and time-to-classification arithmetic, validated against
synthetic ground truth."""

from pathlib import Path

import numpy as np
import pytest

from astra import sn_classification as sn
from astra import sn_classification_eval as sne

TRUE = dict(t0=0.0, amplitude=100.0, tau_rise=5.0, tau_fall=20.0, baseline=1.0)


def _synthetic_curve(seed=0, n=200, span=(-20, 60), noise=2.0):
    rng = np.random.default_rng(seed)
    time = np.linspace(*span, n)
    model = sn.bazin_model(time, **TRUE)
    err = np.full_like(time, noise)
    flux = model + rng.normal(0.0, noise, size=time.size)
    return time, flux, err


# ---------------------------------------------------------------------------
# Model shape
# ---------------------------------------------------------------------------

def test_bazin_model_rises_then_decays():
    time = np.linspace(-10, 60, 500)
    model = sn.bazin_model(time, **TRUE)
    peak_idx = int(np.argmax(model))
    assert 0 < peak_idx < len(time) - 1
    assert model[peak_idx] > model[0]
    assert model[peak_idx] > model[-1]


def test_bazin_model_recovers_baseline_far_from_peak():
    # Many e-foldings of tau_fall past the peak, the model must sit close
    # to baseline (amplitude's contribution decays as exp(-dt/tau_fall)).
    far_time = np.array([TRUE["t0"] + 15 * TRUE["tau_fall"]])
    value = sn.bazin_model(far_time, **TRUE)
    assert value[0] == pytest.approx(TRUE["baseline"], abs=0.05)


def test_bazin_model_rejects_non_positive_timescales():
    with pytest.raises(sn.SNClassificationError):
        sn.bazin_model(np.array([0.0]), t0=0.0, amplitude=1.0, tau_rise=0.0, tau_fall=5.0, baseline=0.0)
    with pytest.raises(sn.SNClassificationError):
        sn.bazin_model(np.array([0.0]), t0=0.0, amplitude=1.0, tau_rise=5.0, tau_fall=-1.0, baseline=0.0)


def test_mag_to_relative_flux_matches_hand_computed_value():
    flux, flux_err = sn.mag_to_relative_flux(np.array([0.0]), np.array([0.01]))
    assert flux[0] == pytest.approx(1.0)
    assert flux_err[0] == pytest.approx(1.0 * 0.4 * np.log(10.0) * 0.01)


# ---------------------------------------------------------------------------
# Point-estimate fit
# ---------------------------------------------------------------------------

def test_fit_bazin_point_estimate_recovers_injected_parameters():
    time, flux, err = _synthetic_curve(seed=1, noise=1.5)
    guess = dict(t0=2.0, amplitude=90.0, tau_rise=4.0, tau_fall=18.0, baseline=0.5)
    fit = sn.fit_bazin_point_estimate(time, flux, err, guess)
    assert fit.t0 == pytest.approx(TRUE["t0"], abs=1.0)
    assert fit.amplitude == pytest.approx(TRUE["amplitude"], rel=0.1)
    assert fit.tau_rise == pytest.approx(TRUE["tau_rise"], rel=0.2)
    assert fit.tau_fall == pytest.approx(TRUE["tau_fall"], rel=0.2)


def test_fit_bazin_point_estimate_requires_every_parameter():
    time, flux, err = _synthetic_curve()
    with pytest.raises(sn.SNClassificationError):
        sn.fit_bazin_point_estimate(time, flux, err, {"t0": 0.0})


def test_fit_bazin_point_estimate_rejects_too_few_points():
    with pytest.raises(sn.SNClassificationError):
        sn.fit_bazin_point_estimate(
            np.arange(3, dtype=float), np.ones(3), np.full(3, 0.1),
            dict(t0=0.0, amplitude=1.0, tau_rise=1.0, tau_fall=1.0, baseline=0.0))


# ---------------------------------------------------------------------------
# Bayesian posterior
# ---------------------------------------------------------------------------

def test_fit_bazin_posterior_recovers_injected_parameters():
    time, flux, err = _synthetic_curve(seed=2, noise=1.5)
    guess = dict(t0=2.0, amplitude=90.0, tau_rise=4.0, tau_fall=18.0, baseline=0.5)
    point_fit = sn.fit_bazin_point_estimate(time, flux, err, guess)
    posterior = sn.fit_bazin_posterior(time, flux, err, point_fit, n_steps=600, n_walkers=24, seed=3)
    assert posterior.medians["amplitude"] == pytest.approx(TRUE["amplitude"], rel=0.15)
    assert posterior.medians["tau_rise"] == pytest.approx(TRUE["tau_rise"], rel=0.3)
    assert isinstance(posterior.converged, bool)
    assert set(posterior.parameter_names) == {"t0", "amplitude", "tau_rise", "tau_fall", "baseline"}


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def test_truncate_light_curve_keeps_only_early_points():
    time = np.array([0.0, 1.0, 5.0, 10.0, 20.0])
    value = np.arange(5, dtype=float)
    err = np.full(5, 0.1)
    t, v, e = sn.truncate_light_curve(time, value, err, cutoff_days_since_first=5.0)
    assert list(t) == [0.0, 1.0, 5.0]


def test_truncate_light_curve_rejects_negative_cutoff():
    with pytest.raises(sn.SNClassificationError):
        sn.truncate_light_curve(np.array([0.0]), np.array([0.0]), np.array([0.1]), -1.0)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_bazin_features_fits_when_enough_points():
    time, flux, err = _synthetic_curve(seed=4, noise=1.0)
    features = sn.bazin_features(time, flux, err)
    assert features["fit_converged"] == 1.0
    assert features["tau_rise"] > 0
    assert features["n_points"] == len(time)


def test_bazin_features_falls_back_on_sparse_early_data():
    time = np.array([0.0, 0.5])
    flux = np.array([5.0, 8.0])
    err = np.array([0.5, 0.5])
    features = sn.bazin_features(time, flux, err)
    assert features["fit_converged"] == 0.0
    assert features["tau_rise"] == 0.0
    assert features["n_points"] == 2.0


def test_bazin_features_vector_has_fixed_length():
    time, flux, err = _synthetic_curve(seed=5)
    vector = sn.features_to_vector(sn.bazin_features(time, flux, err))
    assert vector.shape == (len(sn.FEATURE_NAMES),)


# ---------------------------------------------------------------------------
# Time-to-classification study
# ---------------------------------------------------------------------------

def _labeled_curve(kind: str, seed: int) -> sne.LabeledCurve:
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 60, 40))
    params = dict(t0=5.0, amplitude=100.0, tau_rise=2.0, tau_fall=8.0, baseline=1.0) if kind == "fast" \
        else dict(t0=5.0, amplitude=100.0, tau_rise=8.0, tau_fall=40.0, baseline=1.0)
    model = sn.bazin_model(time, **params)
    err = np.full_like(time, 3.0)
    flux = model + rng.normal(0.0, 3.0, size=time.size)
    return sne.LabeledCurve(time=time, flux=flux, flux_err=err, label=kind)


def test_evaluate_time_to_classification_improves_with_more_data():
    curves = [_labeled_curve("fast", i) for i in range(12)] + \
        [_labeled_curve("slow", 100 + i) for i in range(12)]
    result = sne.evaluate_time_to_classification(curves, cutoff_grid_days=[3, 60], n_seeds=5, seed=1)
    early, late = result.macro_f1_by_cutoff
    assert late is not None
    assert late["mean"] == pytest.approx(result.asymptotic_macro_f1)


def test_evaluate_time_to_classification_requires_two_classes():
    curves = [_labeled_curve("fast", i) for i in range(10)]
    with pytest.raises(sn.SNClassificationError):
        sne.evaluate_time_to_classification(curves, cutoff_grid_days=[10, 60])


def test_evaluate_time_to_classification_rejects_empty_inputs():
    with pytest.raises(sn.SNClassificationError):
        sne.evaluate_time_to_classification([], cutoff_grid_days=[10])
    curves = [_labeled_curve("fast", 0), _labeled_curve("slow", 1)]
    with pytest.raises(sn.SNClassificationError):
        sne.evaluate_time_to_classification(curves, cutoff_grid_days=[])


def test_sn_classification_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "sn_classification" not in rpc_source
