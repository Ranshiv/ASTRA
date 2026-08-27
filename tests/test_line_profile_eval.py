"""line_profile_eval.py: posterior-coverage study and line-parameter
residuals against real released SDSS line measurements."""

from __future__ import annotations

import numpy as np
import pytest

from astra import line_profile as lp
from astra import line_profile_eval as evaluation


def _real_baseline(n_points: int = 800, wave_min: float = 4900.0, wave_max: float = 5100.0,
                   noise_sigma: float = 0.5):
    # Stands in for a real spectrum's wavelength grid/error bars, the same
    # "synthetic signal, real baseline" discipline this module's own
    # docstring describes.
    wave = np.linspace(wave_min, wave_max, n_points)
    error = np.full(n_points, noise_sigma)
    continuum = np.full(n_points, 100.0)
    return wave, error, continuum


class TestRunCoverageStudy:
    def test_reports_all_named_metrics(self):
        pytest.importorskip("emcee", reason="emcee not installed (research extra)")
        wave, error, continuum = _real_baseline()
        result = evaluation.run_coverage_study(
            wave, error, continuum, n_trials=6, n_steps=800, n_walkers=16, seed=1)
        assert result["n_trials_requested"] == 6
        assert "parameter_bias" in result
        assert "posterior_coverage" in result
        assert "sbc" in result
        assert result["baseline"]["n_points"] == 800

    def test_low_bias_on_recovered_centers(self):
        pytest.importorskip("emcee", reason="emcee not installed (research extra)")
        wave, error, continuum = _real_baseline()
        result = evaluation.run_coverage_study(
            wave, error, continuum, n_trials=8, n_steps=800, n_walkers=16, seed=2)
        center_bias = result["parameter_bias"]["parameters"]["center"]
        assert center_bias["n_compared"] >= 6
        assert abs(center_bias["median_absolute_bias"]) < 1.0


class TestLineParameterResiduals:
    def test_recovers_injected_lines_matching_released_values(self):
        wave, error, continuum = _real_baseline(n_points=2000, wave_min=4000.0, wave_max=6000.0)
        rng = np.random.default_rng(7)
        released = [
            {"name": "H-beta", "observed_wavelength_angstrom": 4861.35,
            "sigma_angstrom": 1.2, "sigma_kms": 74.0, "area": 500.0},
            {"name": "[O III]", "observed_wavelength_angstrom": 5006.84,
            "sigma_angstrom": 1.0, "sigma_kms": 60.0, "area": 800.0},
        ]
        flux = np.full(wave.shape, 100.0)
        for line in released:
            params = lp.LineProfileParams(
                center=line["observed_wavelength_angstrom"], sigma=line["sigma_angstrom"],
                gamma=0.0, amplitude=50.0)
            flux = flux + (lp.model_flux(wave, 0.0, params))
        flux = flux + rng.normal(0.0, 0.5, wave.shape)

        result = evaluation.line_parameter_residuals(wave, flux, error, continuum, released)
        assert result["n_compared"] == 2
        assert result["n_out_of_range"] == 0
        center_bias = result["bias"]["parameters"]["center"]
        assert abs(center_bias["median_absolute_bias"]) < 0.5

    def test_out_of_range_lines_are_counted_not_dropped_silently(self):
        wave, error, continuum = _real_baseline(wave_min=4900.0, wave_max=5100.0)
        flux = np.full(wave.shape, 100.0)
        released = [{"name": "far away", "observed_wavelength_angstrom": 9000.0,
                    "sigma_angstrom": 1.0, "sigma_kms": 60.0, "area": 100.0}]
        result = evaluation.line_parameter_residuals(wave, flux, error, continuum, released)
        assert result["n_out_of_range"] == 1
        assert result["n_compared"] == 0

    def test_empty_released_lines(self):
        wave, error, continuum = _real_baseline()
        flux = np.full(wave.shape, 100.0)
        result = evaluation.line_parameter_residuals(wave, flux, error, continuum, [])
        assert result["n_released_lines"] == 0
        assert result["n_compared"] == 0


@pytest.mark.live
class TestLineParameterResidualsLive:
    """Confirmed live this session (2026-08-24): a real SDSS spec-lite FITS
    (`spec-0266-51630-0336.fits`) carries a real `SPZLINE` extension with
    genuine SDSS-pipeline-fitted lines -- see `surveys.sdss.
    extract_sdss_line_measurements`'s docstring for the full finding."""

    def test_fits_a_real_downloaded_spectrum_against_its_own_released_lines(self):
        from astra import spectral_features
        from astra.surveys import sdss
        from astra.surveys.base import SourceRef

        source = SourceRef(survey="SDSS", object_id="1", ra_deg=0.0, dec_deg=0.0,
                           extra={"plate": 266, "mjd": 51630, "fiber_id": 336, "run2d": "26"})
        path = sdss.fetch_spectrum(source)
        released = sdss.extract_sdss_line_measurements(path)
        assert len(released) > 0, (
            "no SPZLINE measurements parsed from a real spec-lite file known "
            "(this session) to carry real fitted lines")

        from astropy.io import fits
        with fits.open(path, memmap=True) as hdul:
            coadd = hdul["COADD"].data
            wave = 10.0 ** np.asarray(coadd["loglam"], dtype=np.float64)
            flux = np.asarray(coadd["flux"], dtype=np.float64)
            ivar = np.asarray(coadd["ivar"], dtype=np.float64)
        error = np.where(ivar > 0, 1.0 / np.sqrt(np.clip(ivar, 1e-30, None)), 0.0)
        good = error > 0
        wave, flux, error = wave[good], flux[good], error[good]

        window = max(5, min(101, (len(flux) // 20) * 2 + 1))
        from scipy.ndimage import median_filter
        continuum = median_filter(flux, size=window, mode="nearest")

        result = evaluation.line_parameter_residuals(wave, flux, error, continuum, released)
        assert result["n_released_lines"] == len(released)


def test_not_referenced_by_rpc():
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "line_profile_eval" not in source
