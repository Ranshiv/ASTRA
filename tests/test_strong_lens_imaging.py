"""Pixel-level strong-lens imaging fit: GaussianSource validation, the
forward-rendering ray-tracer, synthetic recovery, PS1 cutout fetching,
and one real live cutout (roadmap item 29 follow-up)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import netclient, rpc
from astra.strong_lens import SIELens
from astra.strong_lens_imaging import (
    PS1_PIXEL_SCALE_ARCSEC, GaussianSource, StrongLensImagingError, fetch_ps1_cutout,
    fit_pixel_lens_model, render_lensed_image,
)


class TestGaussianSource:
    def test_rejects_non_positive_amplitude(self):
        with pytest.raises(StrongLensImagingError):
            GaussianSource(beta_x=0.0, beta_y=0.0, amplitude=0.0, scale_radius_arcsec=0.2)

    def test_rejects_non_positive_scale_radius(self):
        with pytest.raises(StrongLensImagingError):
            GaussianSource(beta_x=0.0, beta_y=0.0, amplitude=1.0, scale_radius_arcsec=0.0)

    def test_rejects_axis_ratio_out_of_range(self):
        with pytest.raises(StrongLensImagingError):
            GaussianSource(beta_x=0.0, beta_y=0.0, amplitude=1.0, scale_radius_arcsec=0.2,
                           axis_ratio=1.5)

    def test_peaks_at_its_own_center(self):
        source = GaussianSource(beta_x=0.3, beta_y=-0.2, amplitude=5.0, scale_radius_arcsec=0.2)
        peak = source.evaluate(np.array([0.3]), np.array([-0.2]))
        off_center = source.evaluate(np.array([0.3]), np.array([0.0]))
        assert peak[0] == pytest.approx(5.0)
        assert off_center[0] < peak[0]


class TestRenderLensedImage:
    def test_produces_the_requested_shape(self):
        lens = SIELens(theta_e=1.0, axis_ratio=0.8, position_angle=0.2)
        source = GaussianSource(beta_x=0.1, beta_y=-0.05, amplitude=10.0, scale_radius_arcsec=0.15)
        image = render_lensed_image(lens, source, shape=(40, 50), pixel_scale_arcsec=0.25,
                                    psf_fwhm_arcsec=1.0)
        assert image.shape == (40, 50)
        assert np.all(np.isfinite(image))

    def test_a_source_far_outside_the_lens_still_renders_finite_flux(self):
        lens = SIELens(theta_e=1.0, axis_ratio=1.0)
        source = GaussianSource(beta_x=5.0, beta_y=5.0, amplitude=10.0, scale_radius_arcsec=0.15)
        image = render_lensed_image(lens, source, shape=(30, 30), pixel_scale_arcsec=0.25)
        assert np.all(np.isfinite(image))
        assert np.all(image >= 0.0)

    def test_background_is_additive(self):
        lens = SIELens(theta_e=1.0, axis_ratio=1.0)
        source = GaussianSource(beta_x=0.0, beta_y=0.0, amplitude=1.0, scale_radius_arcsec=0.1)
        no_bg = render_lensed_image(lens, source, shape=(20, 20), pixel_scale_arcsec=0.25,
                                    background=0.0)
        with_bg = render_lensed_image(lens, source, shape=(20, 20), pixel_scale_arcsec=0.25,
                                      background=3.0)
        assert with_bg == pytest.approx(no_bg + 3.0)


class TestFitPixelLensModel:
    def test_recovers_injected_parameters_from_a_noiseless_synthetic_image(self):
        true_lens = SIELens(theta_e=1.2, axis_ratio=0.75, position_angle=0.3)
        true_source = GaussianSource(beta_x=0.05, beta_y=-0.03, amplitude=20.0,
                                     scale_radius_arcsec=0.12)
        image = render_lensed_image(true_lens, true_source, shape=(60, 60),
                                    pixel_scale_arcsec=0.25, psf_fwhm_arcsec=0.8, background=2.0)

        initial_guess = GaussianSource(beta_x=0.0, beta_y=0.0, amplitude=15.0,
                                       scale_radius_arcsec=0.15)
        fit = fit_pixel_lens_model(image, pixel_scale_arcsec=0.25, initial_theta_e=1.0,
                                   initial_source=initial_guess, psf_fwhm_arcsec=0.8,
                                   initial_axis_ratio=0.8)
        assert fit["converged"]
        assert fit["theta_e"] == pytest.approx(1.2, rel=0.05)
        assert fit["axis_ratio"] == pytest.approx(0.75, rel=0.1)
        assert fit["background"] == pytest.approx(2.0, abs=0.1)

    def test_recovers_the_right_order_of_magnitude_with_realistic_noise(self):
        rng = np.random.default_rng(17)
        true_lens = SIELens(theta_e=1.0, axis_ratio=0.85, position_angle=0.1)
        true_source = GaussianSource(beta_x=0.02, beta_y=0.01, amplitude=15.0,
                                     scale_radius_arcsec=0.1)
        clean = render_lensed_image(true_lens, true_source, shape=(50, 50),
                                    pixel_scale_arcsec=0.25, psf_fwhm_arcsec=1.0, background=1.0)
        noisy = clean + rng.normal(0.0, 0.3, clean.shape)

        initial_guess = GaussianSource(beta_x=0.0, beta_y=0.0, amplitude=10.0,
                                       scale_radius_arcsec=0.15)
        fit = fit_pixel_lens_model(noisy, pixel_scale_arcsec=0.25, initial_theta_e=0.9,
                                   initial_source=initial_guess, psf_fwhm_arcsec=1.0,
                                   noise_sigma=0.3)
        assert fit["converged"]
        assert fit["theta_e"] == pytest.approx(1.0, rel=0.2)


class TestFetchPs1Cutout:
    def test_rejects_invalid_size(self):
        with pytest.raises(StrongLensImagingError):
            fetch_ps1_cutout(180.0, 0.0, size_pixels=0)
        with pytest.raises(StrongLensImagingError):
            fetch_ps1_cutout(180.0, 0.0, size_pixels=99999)

    def test_raises_when_no_stacked_image_exists(self, monkeypatch, isolated_root):
        class _EmptyFilenamesResponse:
            text = "projcell subcell ra dec filter mjd type filename shortname badflag\n"

        monkeypatch.setattr(netclient, "get", lambda *a, **k: _EmptyFilenamesResponse())
        with pytest.raises(StrongLensImagingError):
            fetch_ps1_cutout(180.0, 89.9, size_pixels=60)

    def test_uses_ps1images_provider_for_filename_lookup(self, monkeypatch, isolated_root):
        captured: dict = {}

        class _EmptyFilenamesResponse:
            text = "projcell subcell ra dec filter mjd type filename shortname badflag\n"

        def fake_get(url, params, timeout, provider):
            captured["provider"] = provider
            return _EmptyFilenamesResponse()

        monkeypatch.setattr(netclient, "get", fake_get)
        with pytest.raises(StrongLensImagingError):
            fetch_ps1_cutout(180.0, 0.0, size_pixels=60)
        assert captured["provider"] == "ps1images"


class TestNotWiredIntoRpc:
    def test_strong_lens_imaging_is_not_referenced_by_rpc(self):
        import inspect

        assert "strong_lens_imaging" not in inspect.getsource(rpc)


@pytest.mark.live
class TestPs1CutoutLive:
    """Confirmed live this session (2026-08-26): PS1's own
    `ps1filenames.py`/`fitscut.cgi` cutout service is real, credential-
    free, and returns a real FITS cutout -- a real cone around the same
    redMaPPer cluster item 30's real cluster-mass study already used
    (RA=140.3977, Dec=-0.2416, both real-checked to be within PS1's
    dec>-30 footprint). The real cutout's own header confirmed
    `PS1_PIXEL_SCALE_ARCSEC` (CDELT1/2 = 6.944e-5 deg = 0.25 arcsec)."""

    def test_fetch_and_load_a_real_cutout(self, isolated_root):
        from astra.strong_lens_imaging import load_cutout

        path = fetch_ps1_cutout(140.3977, -0.2416, size_pixels=120)
        image = load_cutout(path)
        assert image.shape == (120, 120)
        assert np.isfinite(image).sum() > 0

    def test_pixel_fit_machinery_runs_end_to_end_on_real_pixels(self, isolated_root):
        from astra.strong_lens_imaging import load_cutout

        path = fetch_ps1_cutout(140.3977, -0.2416, size_pixels=60)
        image = load_cutout(path)
        image = np.nan_to_num(image, nan=float(np.nanmedian(image)))

        initial_guess = GaussianSource(beta_x=0.0, beta_y=0.0, amplitude=float(np.nanmax(image)),
                                       scale_radius_arcsec=0.5)
        fit = fit_pixel_lens_model(image, pixel_scale_arcsec=PS1_PIXEL_SCALE_ARCSEC,
                                   initial_theta_e=1.0, initial_source=initial_guess,
                                   psf_fwhm_arcsec=1.2)
        # This is a real, generic candidate position, not a confirmed
        # lens system -- the assertion is that the real pipeline runs
        # end to end and returns finite numbers, not that it detects a
        # lens (which real, uncertain single-image data cannot promise).
        assert np.isfinite(fit["theta_e"])
        assert np.isfinite(fit["residual_rms"])
