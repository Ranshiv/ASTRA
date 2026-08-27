"""Davenport template shape, relative-flux-excess conversion, detection,
fit convergence, and ED arithmetic, validated against synthetic ground truth."""

from pathlib import Path

import numpy as np
import pytest

from astra import flare as fl


def _synthetic_flux_curve(t_peak, fwhm=0.05, amplitude=0.2, *, span_days=10.0,
                          cadence_days=0.02, noise_sigma=0.0, seed=0, baseline=1000.0):
    time = np.arange(0.0, span_days, cadence_days)
    model = fl.flare_model(time, t_peak, fwhm, amplitude)
    flux = baseline * (1.0 + model)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        flux = flux + rng.normal(0.0, noise_sigma * baseline, size=flux.size)
    err = np.full_like(time, max(noise_sigma * baseline, 1e-6))
    return time, flux, err


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

def test_template_peak_is_at_t_prime_zero():
    values = fl.davenport_flare_template(np.array([-2.0, -0.5, 0.0, 0.5, 2.0, 5.0]))
    assert values[2] == pytest.approx(0.992, abs=1e-6)  # decay side at t'=0
    assert np.all(values >= 0.0)
    assert values[0] < values[1] < values[2]  # rising toward the peak
    assert values[3] > values[4] > values[5]  # decaying afterward


def test_template_decays_toward_zero():
    far = fl.davenport_flare_template(np.array([20.0]))
    assert far[0] < 0.01


def test_flare_model_validates_parameters():
    with pytest.raises(fl.FlareError):
        fl.flare_model(np.array([0.0]), 0.0, fwhm=0.0, amplitude=0.1)
    with pytest.raises(fl.FlareError):
        fl.flare_model(np.array([0.0]), 0.0, fwhm=0.1, amplitude=0.0)


# ---------------------------------------------------------------------------
# Relative flux excess
# ---------------------------------------------------------------------------

def test_relative_flux_excess_flux_kind_matches_hand_computed_ratio():
    time = np.arange(0.0, 5.0, 0.02)
    flux = np.full_like(time, 1000.0)
    flux[100] = 1200.0  # a single, isolated 20% excursion
    err = np.full_like(time, 1.0)
    excess, excess_err = fl.relative_flux_excess(time, flux, err, "flux", baseline_window_days=1.0)
    assert excess[100] == pytest.approx(0.2, rel=1e-3)
    assert np.allclose(excess[:90], 0.0, atol=1e-6)


def test_relative_flux_excess_mag_kind_matches_flux_kind_for_the_same_signal():
    time, flux, err = _synthetic_flux_curve(t_peak=3.0, amplitude=0.3, noise_sigma=0.0)
    mag_baseline = 12.0
    mag = mag_baseline - 2.5 * np.log10(flux / 1000.0)
    mag_err = np.full_like(time, 0.001)

    excess_flux, _ = fl.relative_flux_excess(time, flux, err, "flux", baseline_window_days=1.0)
    excess_mag, _ = fl.relative_flux_excess(time, mag, mag_err, "mag", baseline_window_days=1.0)
    assert np.max(np.abs(excess_flux - excess_mag)) < 1e-4


def test_relative_flux_excess_rejects_unknown_value_kind():
    time = np.arange(0.0, 5.0, 0.02)
    with pytest.raises(fl.FlareError):
        fl.relative_flux_excess(time, np.ones_like(time), np.full_like(time, 1.0), "counts")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_detect_flare_candidates_recovers_injected_flare():
    time, flux, err = _synthetic_flux_curve(t_peak=5.0, amplitude=0.3, noise_sigma=0.002, seed=2)
    candidates = fl.detect_flare_candidates(time, flux, err, "flux")
    assert len(candidates) >= 1
    assert any(abs(c.peak_time - 5.0) < 0.1 for c in candidates)


def test_detect_flare_candidates_no_false_positive_on_pure_noise():
    rng = np.random.default_rng(3)
    time = np.arange(0.0, 10.0, 0.02)
    flux = 1000.0 + rng.normal(0.0, 1.0, size=time.size)
    err = np.full_like(time, 1.0)
    candidates = fl.detect_flare_candidates(time, flux, err, "flux",
                                            sigma_threshold=5.0, min_consecutive_points=4)
    assert candidates == []


def test_detect_flare_candidates_rejects_bad_parameters():
    time = np.arange(0.0, 5.0, 0.02)
    flux = np.full_like(time, 1000.0)
    err = np.full_like(time, 1.0)
    with pytest.raises(fl.FlareError):
        fl.detect_flare_candidates(time, flux, err, "flux", sigma_threshold=0.0)
    with pytest.raises(fl.FlareError):
        fl.detect_flare_candidates(time, flux, err, "flux", min_consecutive_points=0)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def test_fit_flare_template_recovers_injected_parameters():
    time = np.arange(0.0, 10.0, 0.005)
    true_model = fl.flare_model(time, t_peak=5.0, fwhm=0.05, amplitude=0.2)
    rng = np.random.default_rng(4)
    excess = true_model + rng.normal(0.0, 0.003, size=time.size)
    excess_err = np.full_like(time, 0.003)

    fit = fl.fit_flare_template(time, excess, excess_err,
                                {"t_peak": 4.9, "fwhm": 0.04, "amplitude": 0.15})
    assert fit.t_peak == pytest.approx(5.0, abs=0.01)
    assert fit.fwhm == pytest.approx(0.05, abs=0.01)
    assert fit.amplitude == pytest.approx(0.2, abs=0.03)


def test_fit_flare_template_converges_at_kepler_scale_absolute_time():
    # Real bug found and fixed this session, running this function
    # against a real Kepler light curve for the first time (BJD_TDB ~
    # 2.455e6): scipy's default xtol termination falsely triggered after
    # 2 evaluations because t_peak's absolute magnitude swamped fwhm/
    # amplitude's. Same injected-parameter shape as the test above, but
    # with a large absolute time offset added -- this is a regression
    # test for that exact failure mode, not (only) a synthetic-recovery
    # check.
    time_offset = 2_455_020.0
    time = time_offset + np.arange(0.0, 10.0, 0.005)
    true_model = fl.flare_model(time - time_offset, t_peak=5.0, fwhm=0.05, amplitude=0.2)
    rng = np.random.default_rng(4)
    excess = true_model + rng.normal(0.0, 0.003, size=time.size)
    excess_err = np.full_like(time, 0.003)

    fit = fl.fit_flare_template(time, excess, excess_err,
                                {"t_peak": time_offset + 4.9, "fwhm": 0.04, "amplitude": 0.15})
    assert fit.n_evaluations > 2  # the bug's signature: false convergence after exactly 2
    assert fit.t_peak == pytest.approx(time_offset + 5.0, abs=0.01)
    assert fit.fwhm == pytest.approx(0.05, abs=0.01)
    assert fit.amplitude == pytest.approx(0.2, abs=0.03)


def test_fit_flare_template_requires_every_parameter():
    time = np.arange(0.0, 1.0, 0.01)
    excess = np.zeros_like(time)
    err = np.full_like(time, 0.01)
    with pytest.raises(fl.FlareError):
        fl.fit_flare_template(time, excess, err, {"t_peak": 0.5})


def test_fit_flare_template_rejects_too_few_points():
    with pytest.raises(fl.FlareError):
        fl.fit_flare_template(np.arange(3, dtype=float), np.zeros(3), np.full(3, 0.01),
                              {"t_peak": 1.0, "fwhm": 0.05, "amplitude": 0.1})


# ---------------------------------------------------------------------------
# Equivalent duration
# ---------------------------------------------------------------------------

def test_equivalent_duration_of_a_known_triangular_pulse():
    # A triangular pulse of half-width w and peak height h has area h*w
    # (days), converted to seconds.
    time = np.array([0.0, 1.0, 2.0])
    excess = np.array([0.0, 1.0, 0.0])
    ed = fl.equivalent_duration(time, excess)
    assert ed == pytest.approx(1.0 * fl.SECONDS_PER_DAY)


def test_equivalent_duration_restricts_to_window():
    time = np.linspace(0.0, 4.0, 401)
    excess = np.ones_like(time)
    ed_full = fl.equivalent_duration(time, excess)
    ed_half = fl.equivalent_duration(time, excess, window=(0.0, 2.0))
    assert ed_half == pytest.approx(ed_full / 2.0, rel=1e-2)


def test_equivalent_duration_raises_on_too_few_points():
    with pytest.raises(fl.FlareError):
        fl.equivalent_duration(np.array([1.0]), np.array([0.5]))


def test_flare_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "flare" not in rpc_source
