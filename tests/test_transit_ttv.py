"""BLS search, limb-darkened transit model, fit convergence, TTV O-C, and
vetting-heuristic correctness, validated against synthetic ground truth."""

from pathlib import Path

import numpy as np
import pytest

from astra import transit_ttv as tt
from astra import transit_vetting as tv

RP_RS, PERIOD, A_RS, INC, U1, U2 = 0.1, 3.0, 10.0, 90.0, 0.3, 0.2


def _synthetic_curve(t0, period=PERIOD, rp_rs=RP_RS, a_rs=A_RS, inc=INC, u1=U1, u2=U2,
                     span_days=27.0, cadence_days=0.02, noise=0.0, seed=0):
    time = np.arange(0, span_days, cadence_days)
    flux = tt.limb_darkened_transit_model(time, t0, period, rp_rs, a_rs, inc, u1, u2)
    err = np.full_like(time, max(noise, 1e-4))
    if noise > 0:
        rng = np.random.default_rng(seed)
        flux = flux + rng.normal(0.0, noise, size=flux.size)
    return time, flux, err


# ---------------------------------------------------------------------------
# BLS search
# ---------------------------------------------------------------------------

def test_bls_search_recovers_injected_period():
    time, value, err = _synthetic_curve(t0=1.3, noise=0.0005, seed=1)
    result = tt.bls_search(time, value, err, period_min_days=1.0, period_max_days=10.0, n_periods=3000)
    assert result.period_days == pytest.approx(PERIOD, rel=0.01)
    assert result.depth == pytest.approx(RP_RS ** 2, rel=0.2)
    assert result.snr > 5.0


def test_bls_search_rejects_bad_period_bounds():
    time, value, err = _synthetic_curve(t0=1.3)
    with pytest.raises(tt.TransitTTVError):
        tt.bls_search(time, value, err, period_min_days=10.0, period_max_days=1.0)
    with pytest.raises(tt.TransitTTVError):
        tt.bls_search(time, value, err, period_min_days=1.0, period_max_days=1000.0)


def test_bls_search_rejects_too_few_points():
    with pytest.raises(tt.TransitTTVError):
        tt.bls_search(np.arange(5, dtype=float), np.ones(5), np.full(5, 0.01))


# ---------------------------------------------------------------------------
# Limb-darkened transit model
# ---------------------------------------------------------------------------

def test_quadratic_limb_darkening_intensity_uniform_disk_is_one():
    mu = np.array([0.0, 0.3, 0.7, 1.0])
    assert np.allclose(tt.quadratic_limb_darkening_intensity(mu, 0.0, 0.0), 1.0)


def test_quadratic_limb_darkening_intensity_clips_at_zero():
    intensity = tt.quadratic_limb_darkening_intensity(np.array([0.0]), u1=1.5, u2=1.5)
    assert intensity[0] == 0.0


def test_limb_darkened_transit_model_baseline_outside_transit_is_one():
    time = np.array([-5.0, 5.0])  # far from t0=0, well outside any transit
    flux = tt.limb_darkened_transit_model(time, 0.0, PERIOD, RP_RS, A_RS, INC, U1, U2)
    assert np.allclose(flux, 1.0)


def test_limb_darkened_transit_model_uniform_disk_depth_matches_rp_rs_squared():
    time = np.array([0.0])
    flux = tt.limb_darkened_transit_model(time, 0.0, PERIOD, RP_RS, A_RS, 90.0, 0.0, 0.0)
    assert (1.0 - flux[0]) == pytest.approx(RP_RS ** 2, rel=0.01)


def test_limb_darkened_transit_model_no_secondary_eclipse_dimming():
    # Phase 0.5 (superior conjunction) must show no dimming: this model
    # covers primary transits only, by explicit scope (module docstring).
    time = np.array([PERIOD / 2.0])
    flux = tt.limb_darkened_transit_model(time, 0.0, PERIOD, RP_RS, A_RS, INC, U1, U2)
    assert flux[0] == 1.0


@pytest.mark.parametrize("kwargs", [
    dict(rp_rs=0.0), dict(rp_rs=1.5), dict(a_rs=0.5), dict(inc_deg=95.0), dict(period_days=-1.0),
])
def test_limb_darkened_transit_model_validates_parameters(kwargs):
    params = dict(t0=0.0, period_days=PERIOD, rp_rs=RP_RS, a_rs=A_RS, inc_deg=INC, u1=U1, u2=U2)
    params.update(kwargs)
    with pytest.raises(tt.TransitTTVError):
        tt.limb_darkened_transit_model(np.array([0.0]), **params)


# ---------------------------------------------------------------------------
# Least-squares model fit
# ---------------------------------------------------------------------------

def _initial_guess(t0_guess, period_guess):
    return {"t0": t0_guess, "period_days": period_guess, "rp_rs": 0.08, "a_rs": 9.0,
           "inc_deg": 89.0, "u1": 0.25, "u2": 0.15}


def test_fit_transit_model_recovers_rp_rs_and_period():
    time, value, err = _synthetic_curve(t0=1.3, noise=0.0005, seed=2)
    fit = tt.fit_transit_model(time, value, err, _initial_guess(1.3, PERIOD))
    assert fit.rp_rs == pytest.approx(RP_RS, abs=0.01)
    assert fit.period_days == pytest.approx(PERIOD, abs=0.01)
    assert fit.residual_rms < 2.0


def test_fit_transit_model_requires_every_parameter():
    time, value, err = _synthetic_curve(t0=1.3)
    incomplete = {"t0": 1.3, "period_days": PERIOD}
    with pytest.raises(tt.TransitTTVError):
        tt.fit_transit_model(time, value, err, incomplete)


def test_fit_transit_model_rejects_guess_outside_box_bounds():
    time, value, err = _synthetic_curve(t0=1.3)
    guess = _initial_guess(1.3, PERIOD)
    guess["rp_rs"] = 1.5  # outside (0, 1]
    with pytest.raises(tt.TransitTTVError):
        tt.fit_transit_model(time, value, err, guess)


def test_fit_transit_model_rejects_unphysical_limb_darkening_guess():
    time, value, err = _synthetic_curve(t0=1.3)
    guess = _initial_guess(1.3, PERIOD)
    guess["u1"], guess["u2"] = 0.9, 0.9  # u1 + u2 >= 1
    with pytest.raises(tt.TransitTTVError):
        tt.fit_transit_model(time, value, err, guess)


def test_fit_transit_model_rejects_too_few_points():
    with pytest.raises(tt.TransitTTVError):
        tt.fit_transit_model(np.arange(5, dtype=float), np.ones(5), np.full(5, 0.01),
                             _initial_guess(1.3, PERIOD))


# ---------------------------------------------------------------------------
# Duration estimate
# ---------------------------------------------------------------------------

def _make_fit(**overrides):
    defaults = dict(t0=1.3, period_days=PERIOD, rp_rs=RP_RS, a_rs=A_RS, inc_deg=INC,
                    u1=U1, u2=U2, residual_rms=0.0, n_evaluations=0)
    defaults.update(overrides)
    return tt.TransitFit(**defaults)


def test_estimate_duration_days_matches_closed_form():
    fit = _make_fit(inc_deg=90.0)  # b = 0
    expected = (PERIOD / np.pi) * np.arcsin((1.0 + RP_RS) / A_RS)
    assert tt.estimate_duration_days(fit) == pytest.approx(expected)


def test_estimate_duration_days_raises_for_grazing_geometry():
    fit = _make_fit(a_rs=2.0, inc_deg=0.0)  # impact parameter b = a_rs, too large
    with pytest.raises(tt.TransitTTVError):
        tt.estimate_duration_days(fit)


# ---------------------------------------------------------------------------
# Per-transit midpoints and O-C TTV
# ---------------------------------------------------------------------------

def test_per_transit_midpoints_recovers_true_transit_centers():
    fit = _make_fit(t0=1.3, residual_rms=0.0, n_evaluations=0)
    time, value, err = _synthetic_curve(t0=1.3, noise=0.0, span_days=15.0)
    midpoints, skipped = tv.per_transit_midpoints(time, value, err, fit)
    assert not skipped
    assert len(midpoints) >= 4
    for pt in midpoints:
        expected = fit.t0 + pt.epoch * fit.period_days
        assert pt.midpoint == pytest.approx(expected, abs=0.01)


def test_per_transit_midpoints_skips_sparse_epoch():
    fit = _make_fit(t0=1.3)
    time, value, err = _synthetic_curve(t0=1.3, noise=0.0, span_days=10.0)
    # Remove all points near the epoch-1 transit (t0 + period) so it cannot be fit.
    predicted = fit.t0 + fit.period_days
    keep = np.abs(time - predicted) > 1.0
    midpoints, skipped = tv.per_transit_midpoints(time[keep], value[keep], err[keep], fit)
    assert 1 in skipped
    assert all(pt.epoch != 1 for pt in midpoints)


def test_ttv_o_minus_c_zero_for_perfect_linear_ephemeris():
    fits = [tv.PerTransitFit(epoch=e, midpoint=1.3 + e * PERIOD, n_points=10) for e in range(5)]
    result = tv.ttv_o_minus_c(fits, period_days=PERIOD, t0=1.3)
    assert result.rms_minutes == pytest.approx(0.0, abs=1e-9)
    assert result.amplitude_minutes == pytest.approx(0.0, abs=1e-9)


def test_ttv_o_minus_c_reports_known_injected_residuals():
    offsets_days = {0: 0.0, 1: 0.005, 2: -0.005, 3: 0.0}
    fits = [tv.PerTransitFit(epoch=e, midpoint=1.3 + e * PERIOD + off, n_points=10)
           for e, off in offsets_days.items()]
    result = tv.ttv_o_minus_c(fits, period_days=PERIOD, t0=1.3)
    expected_amplitude = max(offsets_days.values()) * 24 * 60
    assert result.amplitude_minutes == pytest.approx(expected_amplitude, abs=1e-6)
    assert result.n_skipped == 0


def test_ttv_o_minus_c_raises_on_empty_input():
    with pytest.raises(tt.TransitTTVError):
        tv.ttv_o_minus_c([], period_days=PERIOD, t0=1.3)


# ---------------------------------------------------------------------------
# False-positive vetting heuristics
# ---------------------------------------------------------------------------

def _flat_light_curve(fit, duration_days, n_transits=6, cadence_days=0.01):
    span = fit.period_days * n_transits + 2.0
    time = np.arange(0, span, cadence_days)
    value = np.ones_like(time)
    return time, value


def test_vet_candidate_no_flags_for_a_clean_planet_transit():
    fit = _make_fit(t0=1.3)
    time, value, err = _synthetic_curve(t0=1.3, noise=0.0002, span_days=25.0, seed=3)
    result = tv.vet_candidate(time, value, err, fit)
    assert not result.odd_even_flagged
    assert not result.secondary_eclipse_flagged


def test_vet_candidate_flags_odd_even_depth_mismatch():
    fit = _make_fit(t0=1.3)
    duration = tt.estimate_duration_days(fit)
    time, value = _flat_light_curve(fit, duration)
    err = np.full_like(time, 1e-4)
    for epoch in range(6):
        predicted = fit.t0 + epoch * fit.period_days
        mask = np.abs(time - predicted) <= duration / 2.0
        # Odd epochs get a much deeper dip than even epochs -- an eclipsing-
        # binary signature a real planet would never show.
        value[mask] = 1.0 - (0.03 if epoch % 2 else 0.005)
    result = tv.vet_candidate(time, value, err, fit, duration_days=duration)
    assert result.odd_even_flagged
    assert result.odd_even_mismatch_sigma > 3.0


def test_vet_candidate_flags_secondary_eclipse():
    fit = _make_fit(t0=1.3)
    duration = tt.estimate_duration_days(fit)
    time, value = _flat_light_curve(fit, duration)
    err = np.full_like(time, 1e-4)
    primary_depth = 0.01
    for epoch in range(6):
        predicted = fit.t0 + epoch * fit.period_days
        in_mask = np.abs(time - predicted) <= duration / 2.0
        value[in_mask] = 1.0 - primary_depth
        secondary_predicted = predicted + fit.period_days / 2.0
        sec_mask = np.abs(time - secondary_predicted) <= duration / 2.0
        value[sec_mask] = 1.0 - primary_depth * 0.8  # comparably deep -> EB, not a planet
    result = tv.vet_candidate(time, value, err, fit, duration_days=duration)
    assert result.secondary_eclipse_flagged


def test_vet_candidate_flags_v_shaped_grazing_transit():
    fit = _make_fit(t0=1.3)
    duration = tt.estimate_duration_days(fit)
    time, value = _flat_light_curve(fit, duration)
    err = np.full_like(time, 1e-4)
    for epoch in range(6):
        predicted = fit.t0 + epoch * fit.period_days
        offset = time - predicted
        in_mask = np.abs(offset) <= duration / 2.0
        # Linear ramp down to the center and back up: a pure V, essentially
        # no flat bottom -- the grazing/EB shape signature.
        value[in_mask] = 1.0 - 0.02 * (1.0 - np.abs(offset[in_mask]) / (duration / 2.0))
    result = tv.vet_candidate(time, value, err, fit, duration_days=duration)
    assert result.v_shape_flagged
    assert result.shape_flat_fraction < 0.2


def test_vet_candidate_requires_at_least_two_observed_transits():
    fit = _make_fit(t0=1.3)
    time = np.arange(0, 1.0, 0.01)  # spans less than one period
    value = np.ones_like(time)
    err = np.full_like(time, 1e-4)
    with pytest.raises(tt.TransitTTVError):
        tv.vet_candidate(time, value, err, fit)


def test_transit_ttv_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "transit_ttv" not in rpc_source
    assert "transit_vetting" not in rpc_source
