"""DRW fit recovery, TDE flare model shape, change-point evidence
correctness, and significance calibration, validated against synthetic
ground truth.

celerite2 is an opt-in research dependency (`engine pyproject.toml`'s
`research` extra), gated like torch -- skip this whole file when it is not
installed, the same as `test_multiband_hier.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

celerite2 = pytest.importorskip("celerite2", reason="celerite2 not installed (opt-in 'research' extra)")
from celerite2 import terms  # noqa: E402

from astra import agn_changepoint as agn  # noqa: E402

TRUE_SIGMA, TRUE_TAU = 0.3, 30.0


def _synthetic_drw(span_days=5000.0, cadence_days=2.0, noise_sigma=0.02, seed=1,
                   sigma=TRUE_SIGMA, tau=TRUE_TAU):
    term = terms.RealTerm(a=sigma ** 2, c=1.0 / tau)
    time = np.arange(0.0, span_days, cadence_days)
    gp = celerite2.GaussianProcess(term, mean=0.0)
    gp.compute(time, diag=noise_sigma ** 2 * np.ones_like(time))
    np.random.seed(seed)
    value = gp.sample()
    err = np.full_like(time, noise_sigma)
    return time, value, err


# ---------------------------------------------------------------------------
# DRW fit
# ---------------------------------------------------------------------------

def test_fit_drw_recovers_injected_parameters_on_a_long_baseline():
    # A baseline much longer than tau is needed for tight DRW recovery --
    # a real, documented property of DRW parameter estimation, not a
    # tunable test tolerance.
    time, value, err = _synthetic_drw()
    fit = agn.fit_drw(time, value, err)
    assert fit.sigma == pytest.approx(TRUE_SIGMA, rel=0.2)
    assert fit.tau == pytest.approx(TRUE_TAU, rel=0.2)


def test_fit_drw_rejects_too_few_points():
    with pytest.raises(agn.AGNChangepointError):
        agn.fit_drw(np.arange(5, dtype=float), np.ones(5), np.full(5, 0.01))


# ---------------------------------------------------------------------------
# TDE flare model
# ---------------------------------------------------------------------------

def test_tde_flare_model_is_continuous_and_peaks_near_t0():
    time = np.linspace(-50, 200, 1000)
    model = agn.tde_flare_model(time, t0=0.0, amplitude=10.0, rise_sigma=5.0, t_decay_ref=10.0)
    peak_idx = int(np.argmax(model))
    assert abs(time[peak_idx]) < 1.0
    assert model[peak_idx] == pytest.approx(10.0, rel=0.05)


def test_tde_flare_model_decay_matches_analytic_power_law():
    dt = 100.0
    expected = 10.0 * ((dt + 10.0) / 10.0) ** (-5.0 / 3.0)
    actual = agn.tde_flare_model(np.array([dt]), t0=0.0, amplitude=10.0,
                                 rise_sigma=5.0, t_decay_ref=10.0)[0]
    assert actual == pytest.approx(expected)


def test_tde_flare_model_validates_parameters():
    with pytest.raises(agn.AGNChangepointError):
        agn.tde_flare_model(np.array([0.0]), t0=0.0, amplitude=0.0, rise_sigma=1.0, t_decay_ref=1.0)
    with pytest.raises(agn.AGNChangepointError):
        agn.tde_flare_model(np.array([0.0]), t0=0.0, amplitude=1.0, rise_sigma=-1.0, t_decay_ref=1.0)


# ---------------------------------------------------------------------------
# Change-point evidence
# ---------------------------------------------------------------------------

def test_changepoint_evidence_favors_drw_only_on_pure_noise():
    time, value, err = _synthetic_drw(span_days=500.0, seed=2)
    fit = agn.fit_drw(time, value, err)
    guess = agn.default_flare_guess(time, value, fit)
    evidence = agn.changepoint_evidence(time, value, err, fit, guess)
    assert evidence.delta_bic > 0


def test_changepoint_evidence_favors_flare_on_a_real_injection():
    time, value, err = _synthetic_drw(span_days=500.0, seed=2)
    fit = agn.fit_drw(time, value, err)
    t0_true = 250.0
    flare = agn.tde_flare_model(time, t0_true, amplitude=1.5, rise_sigma=15.0, t_decay_ref=20.0)
    with_flare = value + flare

    guess = agn.default_flare_guess(time, with_flare, fit)
    evidence = agn.changepoint_evidence(time, with_flare, err, fit, guess)
    assert evidence.delta_bic < -6.0  # Kass & Raftery "strong evidence" scale
    assert evidence.flare_params["t0"] == pytest.approx(t0_true, abs=5.0)
    assert evidence.flare_params["amplitude"] == pytest.approx(1.5, rel=0.3)


def test_default_flare_guess_seeds_near_the_largest_excursion():
    time, value, err = _synthetic_drw(span_days=500.0, seed=2)
    fit = agn.fit_drw(time, value, err)
    flare = agn.tde_flare_model(time, 250.0, amplitude=3.0, rise_sigma=10.0, t_decay_ref=10.0)
    guess = agn.default_flare_guess(time, value + flare, fit)
    assert guess["t0"] == pytest.approx(250.0, abs=10.0)


def test_changepoint_evidence_requires_every_flare_parameter():
    time, value, err = _synthetic_drw(span_days=500.0, seed=2)
    fit = agn.fit_drw(time, value, err)
    with pytest.raises(agn.AGNChangepointError):
        agn.changepoint_evidence(time, value, err, fit, {"t0": 100.0})


# ---------------------------------------------------------------------------
# Significance calibration
# ---------------------------------------------------------------------------

def test_calibrate_changepoint_significance_returns_a_finite_threshold():
    time, value, err = _synthetic_drw(span_days=300.0, seed=4)
    fit = agn.fit_drw(time, value, err)
    threshold = agn.calibrate_changepoint_significance(
        fit, time, err, n_realizations=15, target_fpr=0.1, seed=5)
    assert np.isfinite(threshold)


def test_calibrate_changepoint_significance_rejects_bad_parameters():
    time, value, err = _synthetic_drw(span_days=300.0, seed=4)
    fit = agn.fit_drw(time, value, err)
    with pytest.raises(agn.AGNChangepointError):
        agn.calibrate_changepoint_significance(fit, time, err, n_realizations=0)
    with pytest.raises(agn.AGNChangepointError):
        agn.calibrate_changepoint_significance(fit, time, err, target_fpr=1.5)


def test_agn_changepoint_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "agn_changepoint" not in rpc_source
