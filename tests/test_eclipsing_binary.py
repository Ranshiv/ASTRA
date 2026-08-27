"""Eclipse geometry correctness, fit convergence, and depth/temperature-ratio
arithmetic, validated against synthetic ground truth."""

from pathlib import Path

import numpy as np
import pytest

from astra import eclipsing_binary as eb

PERIOD = 3.0
R1_A, R2_A = 0.1, 0.08
INC = 90.0
U1_1, U2_1, U1_2, U2_2 = 0.3, 0.2, 0.3, 0.2
TEFF_RATIO = 0.9


def _synthetic_curve(t0, period=PERIOD, r1_a=R1_A, r2_a=R2_A, inc=INC,
                     u1_1=U1_1, u2_1=U2_1, u1_2=U1_2, u2_2=U2_2, teff_ratio=TEFF_RATIO,
                     span_days=27.0, cadence_days=0.02, noise=0.0, seed=0):
    time = np.arange(0, span_days, cadence_days)
    flux = eb.eclipsing_binary_model(time, t0, period, r1_a, r2_a, inc,
                                     u1_1, u2_1, u1_2, u2_2, teff_ratio)
    err = np.full_like(time, max(noise, 1e-4))
    if noise > 0:
        rng = np.random.default_rng(seed)
        flux = flux + rng.normal(0.0, noise, size=flux.size)
    return time, flux, err


def _make_fit(**overrides):
    defaults = dict(t0=1.3, period_days=PERIOD, r1_a=R1_A, r2_a=R2_A, inc_deg=INC,
                    u1_1=U1_1, u2_1=U2_1, u1_2=U1_2, u2_2=U2_2, teff_ratio=TEFF_RATIO,
                    residual_rms=0.0, n_evaluations=0)
    defaults.update(overrides)
    return eb.EclipsingBinaryFit(**defaults)


def _initial_guess(**overrides):
    guess = dict(t0=1.3, period_days=PERIOD, r1_a=0.09, r2_a=0.07, inc_deg=89.0,
                 u1_1=0.25, u2_1=0.15, u1_2=0.25, u2_2=0.15, teff_ratio=0.85)
    guess.update(overrides)
    return guess


# ---------------------------------------------------------------------------
# Geometry model
# ---------------------------------------------------------------------------

def test_model_baseline_outside_eclipse_is_one():
    time = np.array([-5.0, 5.0])
    flux = eb.eclipsing_binary_model(time, 0.0, PERIOD, R1_A, R2_A, INC,
                                     U1_1, U2_1, U1_2, U2_2, TEFF_RATIO)
    assert np.allclose(flux, 1.0)


def test_primary_deeper_than_secondary_for_cooler_secondary():
    fit = _make_fit(t0=0.0, teff_ratio=0.7)  # body 2 cooler -> dimmer -> shallower secondary
    primary_depth, secondary_depth = eb.primary_secondary_depths(fit)
    assert primary_depth > secondary_depth > 0


def test_equal_radius_equal_temperature_gives_equal_depths():
    fit = _make_fit(t0=0.0, r1_a=0.1, r2_a=0.1, u1_1=0.3, u2_1=0.2, u1_2=0.3, u2_2=0.2, teff_ratio=1.0)
    primary_depth, secondary_depth = eb.primary_secondary_depths(fit)
    assert primary_depth == pytest.approx(secondary_depth, rel=1e-9)


def test_total_eclipse_by_larger_occulter_matches_luminosity_ratio():
    # Body 2 (radius 0.15) completely covers body 1 (radius 0.05) with equal
    # temperature -> remaining flux at mid-primary-eclipse is exactly
    # L2 / (L1 + L2) = r2_a^2 / (r1_a^2 + r2_a^2) for identical limb darkening.
    r1_a, r2_a = 0.05, 0.15
    flux = eb.eclipsing_binary_model(np.array([0.0]), 0.0, PERIOD, r1_a, r2_a, 90.0,
                                     0.3, 0.2, 0.3, 0.2, 1.0)
    expected = r2_a ** 2 / (r1_a ** 2 + r2_a ** 2)
    assert flux[0] == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("kwargs", [
    dict(r1_a=0.0), dict(r1_a=1.0), dict(r2_a=0.0), dict(inc_deg=95.0),
    dict(period_days=-1.0), dict(teff_ratio=0.0),
])
def test_model_validates_parameters(kwargs):
    params = dict(t0=0.0, period_days=PERIOD, r1_a=R1_A, r2_a=R2_A, inc_deg=INC,
                 u1_1=U1_1, u2_1=U2_1, u1_2=U1_2, u2_2=U2_2, teff_ratio=TEFF_RATIO)
    params.update(kwargs)
    with pytest.raises(eb.EclipsingBinaryError):
        eb.eclipsing_binary_model(np.array([0.0]), **params)


# ---------------------------------------------------------------------------
# Least-squares fit
# ---------------------------------------------------------------------------

def test_fit_recovers_geometry_and_teff_ratio():
    time, value, err = _synthetic_curve(t0=1.3, noise=0.0005, seed=1)
    fit = eb.fit_eclipsing_binary(time, value, err, _initial_guess())
    assert fit.r1_a == pytest.approx(R1_A, abs=0.01)
    assert fit.r2_a == pytest.approx(R2_A, abs=0.01)
    assert fit.period_days == pytest.approx(PERIOD, abs=0.01)
    assert fit.teff_ratio == pytest.approx(TEFF_RATIO, abs=0.05)


def test_fit_requires_every_parameter():
    time, value, err = _synthetic_curve(t0=1.3)
    incomplete = {"t0": 1.3, "period_days": PERIOD}
    with pytest.raises(eb.EclipsingBinaryError):
        eb.fit_eclipsing_binary(time, value, err, incomplete)


def test_fit_rejects_guess_with_overlapping_radii():
    time, value, err = _synthetic_curve(t0=1.3)
    guess = _initial_guess(r1_a=0.6, r2_a=0.5)  # sum >= 1
    with pytest.raises(eb.EclipsingBinaryError):
        eb.fit_eclipsing_binary(time, value, err, guess)


def test_fit_rejects_unphysical_limb_darkening_guess():
    time, value, err = _synthetic_curve(t0=1.3)
    guess = _initial_guess(u1_1=0.9, u2_1=0.9)
    with pytest.raises(eb.EclipsingBinaryError):
        eb.fit_eclipsing_binary(time, value, err, guess)


def test_fit_rejects_too_few_points():
    with pytest.raises(eb.EclipsingBinaryError):
        eb.fit_eclipsing_binary(np.arange(5, dtype=float), np.ones(5), np.full(5, 0.01),
                                _initial_guess())


# ---------------------------------------------------------------------------
# Depth / temperature-ratio arithmetic
# ---------------------------------------------------------------------------

def test_temperature_ratio_recovers_injected_value_approximately():
    fit = _make_fit(t0=0.0, teff_ratio=0.85)
    primary_depth, secondary_depth = eb.primary_secondary_depths(fit)
    recovered = eb.temperature_ratio_from_depths(primary_depth, secondary_depth)
    assert recovered == pytest.approx(0.85, abs=0.05)


def test_temperature_ratio_raises_for_non_positive_primary_depth():
    with pytest.raises(eb.EclipsingBinaryError):
        eb.temperature_ratio_from_depths(0.0, 0.1)


def test_eclipsing_binary_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "eclipsing_binary" not in rpc_source
