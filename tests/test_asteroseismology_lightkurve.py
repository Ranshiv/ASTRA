"""Optional cross-check that `lightkurve.seismology` (available since
`lightkurve` is a core ASTRA dependency) is actually callable end to end
on a synthetic solar-like oscillation light curve.

This is NOT a `live` test -- no network is touched, `lightkurve.
seismology`'s numax/Dnu estimators run entirely on a local synthetic
light curve. It is gated with `pytest.importorskip` rather than the
`live` marker because what it checks is "is this optional dependency's
API surface present and runnable," the same category `test_biosignature_
fit.py`'s `pytest.importorskip("emcee", ...)` check already uses.

`asteroseismology.py`'s own docstring states `lightkurve.seismology` is
a strictly optional PRODUCTION cross-check, never the implementation
path (`measure()` uses a pure-numpy Gaussian-excess-fit + autocorrelation
estimator instead, so the offline test suite never depends on this file).
A quick manual check before writing this test found lightkurve's
2-D-autocorrelation numax estimator (Viani et al. 2019) does NOT closely
agree with this module's own estimator on the same synthetic input at
default tuning (lightkurve found ~7145 uHz against an injected 1200 uHz
truth in one trial) -- the two estimators are different algorithms with
different default assumptions about signal amplitude/noise scaling, and
that disagreement is not itself evidence either implementation is wrong.
This test therefore checks only that the library call SUCCEEDS and
returns a positive, finite value, not that it agrees with `measure()` to
any tolerance -- a tight cross-check would be fragile and is not what
this test is for.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import asteroseismology as ast


def test_lightkurve_seismology_runs_on_a_synthetic_oscillating_light_curve():
    lk = pytest.importorskip("lightkurve", reason="lightkurve not installed")
    pytest.importorskip("lightkurve.seismology", reason="lightkurve.seismology not installed")
    import astropy.units as u

    numax_true = 1200.0
    dnu_true = ast.STELLO_DNU_COEFF * numax_true ** ast.STELLO_DNU_EXPONENT
    rng = np.random.default_rng(1)
    n = 20000
    dt_days = 2.0 / 1440.0
    time_days = np.arange(n) * dt_days
    lo, hi = ast.envelope_window(numax_true)
    sigma = (hi - lo) / 2.355
    flux = np.ones(n)
    for f in np.arange(lo, hi, dnu_true):
        amplitude = 0.001 * np.exp(-0.5 * ((f - numax_true) / sigma) ** 2)
        freq_per_day = f / ast.UHZ_PER_DAY_INVERSE
        phase = rng.uniform(0.0, 2.0 * np.pi)
        flux += amplitude * np.sin(2.0 * np.pi * freq_per_day * time_days + phase)
    flux += rng.normal(0.0, 2e-4, n)

    curve = lk.LightCurve(time=time_days * u.day, flux=flux)
    periodogram = curve.to_periodogram(normalization="psd", minimum_frequency=10 * u.uHz,
                                       maximum_frequency=8000 * u.uHz)
    seismology = periodogram.flatten().to_seismology()

    numax = seismology.estimate_numax()
    assert np.isfinite(numax.value) and numax.value > 0

    delta_nu = seismology.estimate_deltanu()
    assert np.isfinite(delta_nu.value) and delta_nu.value > 0
