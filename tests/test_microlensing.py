"""microlensing.py: point-lens and binary-lens forward models (backlog
item 15)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import microlensing as ml

vbmicrolensing = pytest.importorskip(
    "VBMicrolensing", reason="VBMicrolensing not installed (research extra)")


class TestPointLensParams:
    def test_rejects_non_positive_te(self):
        with pytest.raises(ml.MicrolensingError):
            ml.PointLensParams(t0=0.0, tE=0.0, u0=0.1)

    def test_rejects_negative_u0(self):
        with pytest.raises(ml.MicrolensingError):
            ml.PointLensParams(t0=0.0, tE=10.0, u0=-0.1)

    def test_rejects_non_finite_values(self):
        with pytest.raises(ml.MicrolensingError):
            ml.PointLensParams(t0=float("nan"), tE=10.0, u0=0.1)

    def test_array_round_trip(self):
        p = ml.PointLensParams(t0=5.0, tE=12.0, u0=0.3)
        recovered = ml.PointLensParams.from_array(p.to_array())
        assert recovered == p


class TestMagnification:
    def test_approaches_one_far_from_the_peak(self):
        p = ml.PointLensParams(t0=100.0, tE=20.0, u0=0.1)
        far = ml.magnification(np.array([100.0 + 20.0 * 100.0]), p)
        assert far[0] == pytest.approx(1.0, abs=1e-4)

    def test_high_magnification_limit_is_one_over_u(self):
        # A(u) ~ 1/u for u << 1, the well-known high-magnification limit.
        p = ml.PointLensParams(t0=100.0, tE=20.0, u0=1e-3)
        peak = ml.magnification(np.array([100.0]), p)
        assert peak[0] == pytest.approx(1.0 / 1e-3, rel=1e-2)

    def test_symmetric_about_t0(self):
        p = ml.PointLensParams(t0=100.0, tE=20.0, u0=0.2)
        before = ml.magnification(np.array([80.0]), p)
        after = ml.magnification(np.array([120.0]), p)
        assert before[0] == pytest.approx(after[0])

    def test_peak_magnification_matches_the_closed_form(self):
        # At t=t0, u=u0, and A(u0) has an exact closed form.
        u0 = 0.25
        p = ml.PointLensParams(t0=0.0, tE=1.0, u0=u0)
        peak = ml.magnification(np.array([0.0]), p)[0]
        expected = (u0 ** 2 + 2) / (u0 * np.sqrt(u0 ** 2 + 4))
        assert peak == pytest.approx(expected)

    def test_always_at_least_one(self):
        p = ml.PointLensParams(t0=0.0, tE=5.0, u0=0.5)
        t = np.linspace(-100, 100, 500)
        assert np.all(ml.magnification(t, p) >= 1.0 - 1e-9)


class TestLinearFlux:
    def test_recovers_exact_source_and_blend_on_noiseless_data(self):
        p = ml.PointLensParams(t0=50.0, tE=15.0, u0=0.2)
        t = np.linspace(0, 100, 300)
        f_source, f_blend = 4.2, 1.1
        flux = ml.model_flux(t, p, f_source, f_blend)
        err = np.full_like(flux, 0.01)
        fitted_source, fitted_blend = ml.solve_linear_flux(t, flux, err, p)
        assert fitted_source == pytest.approx(f_source, rel=1e-8)
        assert fitted_blend == pytest.approx(f_blend, rel=1e-8)

    def test_degenerate_design_does_not_raise(self):
        # A single point (or a curve with no time variation at all in the
        # amplification) makes the design matrix singular.
        p = ml.PointLensParams(t0=0.0, tE=1e6, u0=100.0)  # ~flat A(t)
        t = np.linspace(0, 1, 5)
        flux = np.full(5, 3.0)
        err = np.full(5, 0.1)
        source, blend = ml.solve_linear_flux(t, flux, err, p)
        assert np.isfinite(source) and np.isfinite(blend)


class TestMagFluxConversion:
    def test_round_trips(self):
        mag = np.array([15.0, 17.5, 20.1])
        assert ml.flux_to_mag(ml.mag_to_flux(mag)) == pytest.approx(mag)

    def test_brighter_magnitude_is_more_flux(self):
        bright, faint = ml.mag_to_flux(np.array([15.0])), ml.mag_to_flux(np.array([20.0]))
        assert bright[0] > faint[0]

    def test_error_propagation_scales_with_flux(self):
        mag = np.array([15.0])
        err_small = ml.mag_err_to_flux_err(mag, np.array([0.01]))
        err_large = ml.mag_err_to_flux_err(mag, np.array([0.1]))
        assert err_large[0] > err_small[0]


class TestBinaryLensParams:
    def test_rejects_bad_mass_ratio(self):
        with pytest.raises(ml.MicrolensingError):
            ml.BinaryLensParams(t0=0, tE=10, u0=0.1, s=1.0, q=1.5, alpha=0.0)

    def test_rejects_non_positive_separation(self):
        with pytest.raises(ml.MicrolensingError):
            ml.BinaryLensParams(t0=0, tE=10, u0=0.1, s=0.0, q=0.5, alpha=0.0)


class TestBinaryMagnification:
    def test_output_is_finite_and_at_least_one(self):
        p = ml.BinaryLensParams(t0=100.0, tE=20.0, u0=0.1, s=1.0, q=0.5,
                                alpha=0.5, rho=1e-3)
        t = np.linspace(80, 120, 30)
        A = ml.binary_magnification(t, p)
        assert np.all(np.isfinite(A))
        assert np.all(A >= 1.0 - 1e-6)

    def test_vanishing_mass_ratio_converges_to_the_point_lens(self):
        # As q -> 0 the secondary's influence vanishes and the binary
        # magnification must converge to the exact point-lens formula --
        # this is checked against the CLOSED-FORM model, not a fixture, so
        # it validates the physics, not just internal consistency.
        point = ml.PointLensParams(t0=100.0, tE=20.0, u0=0.3)
        binary = ml.BinaryLensParams(t0=100.0, tE=20.0, u0=0.3, s=1.0,
                                     q=1e-6, alpha=0.5, rho=1e-3)
        t = np.linspace(80, 120, 25)
        assert ml.binary_magnification(t, binary) == pytest.approx(
            ml.magnification(t, point), abs=1e-3)
