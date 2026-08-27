"""Pixel-level strong-lens imaging fit (roadmap item 29 follow-up).

Everything in `strong_lens.py` fits ISOLATED IMAGE POSITIONS -- it never
touches real pixels. This module closes that specific, previously-named
gap: a real image cutout, a forward-rendered lensed-image model, and a
pixel-level least-squares fit, the same `tess_psf.py`-shaped "forward
model fit to real pixel data" pattern this codebase already uses for
TESS PSF deblending.

Real image source: Pan-STARRS1 (PS1), via its own public, credential-free
cutout service (`ps1images.stsci.edu`) -- confirmed live this session:
`ps1filenames.py` (resolves a real position to a real stacked-image
filename) followed by `fitscut.cgi` (returns a real FITS cutout) both
work with no authentication. DES was checked first and confirmed broken
(see `docs/DEFERRED.txt`'s item-29 entry: `surveys/des.py`'s TAP cone
search currently fails server-side); Hubble/JWST are pointed, narrow-
footprint programmes unlikely to cover a generic candidate; PS1 is the
practical real, all-sky (`dec > -30`) choice. PS1's real plate scale
(0.25 arcsec/pixel) was confirmed live this session by inspecting a real
cutout's own `CDELT1`/`CDELT2` header keywords (6.944e-5 deg = 0.25
arcsec), not assumed from documentation.

The source light profile is a single elliptical GAUSSIAN, deliberately
not a full Sersic profile: a lensed source is typically compact and
barely resolved even in good seeing, so a Gaussian is a real, standard,
much simpler starting model (e.g. used as the default source shape in
several public lens-modelling codes' quick-look fits) -- adding Sersic's
extra shape parameter (n) is a real, stated possible extension, not
attempted here, the same "avoid a needless dependency/complexity when a
simpler real model already does the job" choice `transit_ttv.py` makes
against a compiled transit-modelling library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import config, netclient
from .strong_lens import ExternalShear, SIELens, total_deflection

SCHEMA_VERSION = 1
PS1_FILENAME_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PS1_CUTOUT_URL = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
PS1_PIXEL_SCALE_ARCSEC = 0.25  # confirmed live this session via a real cutout's own header
MAX_CUTOUT_BYTES = 8 * 1024 * 1024


class StrongLensImagingError(RuntimeError):
    """A real cutout or pixel-level fit could not be produced."""


def fetch_ps1_cutout(ra_deg: float, dec_deg: float, *, size_pixels: int = 120,
                     filter_name: str = "r", root: Path | None = None,
                     overwrite: bool = False) -> Path:
    """Download (once, cached) a real PS1 FITS cutout centred on
    `(ra_deg, dec_deg)`. Two real requests, exactly as PS1's own service
    requires: `ps1filenames.py` resolves the position to a real stacked-
    image filename, then `fitscut.cgi` returns the actual cutout FITS
    built from that file."""
    if size_pixels <= 0 or size_pixels > 2000:
        raise StrongLensImagingError("size_pixels must be a positive value up to 2000")
    root = root or config.PATHS.datasets
    key = f"ra{ra_deg:.5f}_dec{dec_deg:.5f}_{filter_name}_{size_pixels}px"
    destination = root / "PS1_CUTOUTS" / f"{key}.fits"
    if destination.exists() and not overwrite:
        return destination

    filenames_response = netclient.get(
        PS1_FILENAME_URL, {"ra": ra_deg, "dec": dec_deg, "filters": filter_name},
        timeout=30, provider="ps1images",
    )
    lines = [line for line in filenames_response.text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise StrongLensImagingError("PS1 has no stacked image at this position")
    header_cols = lines[0].split()
    values = lines[1].split()
    row = dict(zip(header_cols, values))
    remote_filename = row.get("filename")
    if not remote_filename:
        raise StrongLensImagingError("PS1 filename lookup did not return a usable filename")

    destination.parent.mkdir(parents=True, exist_ok=True)
    netclient.download(
        PS1_CUTOUT_URL, destination,
        params={"ra": ra_deg, "dec": dec_deg, "size": size_pixels, "format": "fits",
               "red": remote_filename},
        provider="ps1images", max_bytes=MAX_CUTOUT_BYTES, overwrite=overwrite,
    )
    return destination


def load_cutout(path: Path) -> np.ndarray:
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
    if data.ndim != 2:
        raise StrongLensImagingError(f"expected a 2-D image, got shape {data.shape}")
    return data


@dataclass(frozen=True)
class GaussianSource:
    """An elliptical Gaussian background-source light profile, in the
    SOURCE plane (beta coordinates, arcsec relative to the lens centre)."""

    beta_x: float
    beta_y: float
    amplitude: float
    scale_radius_arcsec: float
    axis_ratio: float = 1.0
    position_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.amplitude <= 0 or self.scale_radius_arcsec <= 0:
            raise StrongLensImagingError("amplitude and scale_radius_arcsec must be positive")
        if not 0.0 < self.axis_ratio <= 1.0:
            raise StrongLensImagingError("axis_ratio must be in (0, 1]")

    def evaluate(self, beta_x: np.ndarray, beta_y: np.ndarray) -> np.ndarray:
        cos_pa, sin_pa = math.cos(self.position_angle_rad), math.sin(self.position_angle_rad)
        dx, dy = beta_x - self.beta_x, beta_y - self.beta_y
        x_rot = dx * cos_pa + dy * sin_pa
        y_rot = -dx * sin_pa + dy * cos_pa
        radius_sq = x_rot ** 2 + (y_rot / self.axis_ratio) ** 2
        return self.amplitude * np.exp(-0.5 * radius_sq / self.scale_radius_arcsec ** 2)


def _gaussian_psf_convolve(image: np.ndarray, fwhm_arcsec: float, pixel_scale_arcsec: float
                           ) -> np.ndarray:
    if fwhm_arcsec <= 0:
        return image
    from scipy.ndimage import gaussian_filter

    sigma_pixels = (fwhm_arcsec / pixel_scale_arcsec) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return gaussian_filter(image, sigma=sigma_pixels)


def render_lensed_image(lens: SIELens, source: GaussianSource, *, shape: tuple[int, int],
                        pixel_scale_arcsec: float, shear: ExternalShear | None = None,
                        psf_fwhm_arcsec: float = 1.2, background: float = 0.0) -> np.ndarray:
    """Forward-render a lensed image: for every image-plane pixel `theta`
    (arcsec, relative to the lens centre at the image's own centre),
    ray-trace back to the source plane via `beta = theta - alpha(theta)`
    (`strong_lens.total_deflection`, reused unchanged), evaluate the
    source profile there, then convolve with a Gaussian PSF -- the
    standard ray-tracing approach to rendering a strong-lens image (e.g.
    Kochanek 2006 lecture notes, sec. 1)."""
    n_y, n_x = shape
    y_indices, x_indices = np.mgrid[0:n_y, 0:n_x]
    theta_x = (x_indices - n_x / 2.0 + 0.5) * pixel_scale_arcsec
    theta_y = (y_indices - n_y / 2.0 + 0.5) * pixel_scale_arcsec
    theta = np.stack([theta_x.ravel(), theta_y.ravel()], axis=-1)

    alpha = total_deflection(lens, theta, shear)
    beta = theta - alpha
    unconvolved = source.evaluate(beta[:, 0], beta[:, 1]).reshape(n_y, n_x)
    convolved = _gaussian_psf_convolve(unconvolved, psf_fwhm_arcsec, pixel_scale_arcsec)
    return convolved + background


def fit_pixel_lens_model(image: np.ndarray, *, pixel_scale_arcsec: float,
                         initial_theta_e: float, initial_source: GaussianSource,
                         psf_fwhm_arcsec: float = 1.2, noise_sigma: float | None = None,
                         initial_axis_ratio: float = 0.8, initial_position_angle: float = 0.0
                         ) -> dict[str, Any]:
    """Least-squares refinement of `(theta_e, axis_ratio, position_angle,
    source_beta_x, source_beta_y, source_amplitude, source_scale_radius,
    background)` directly against real pixel VALUES -- the pixel-level
    fit this roadmap item's imaging gap needed, going beyond `strong_
    lens.fit_lens_model`'s point-image-position fit. `noise_sigma`
    (constant per-pixel noise, the simplest real error model) weights the
    residuals when supplied; otherwise every pixel is weighted equally
    (an unweighted least-squares shape fit)."""
    from scipy.optimize import least_squares

    image = np.asarray(image, dtype=np.float64)
    n_y, n_x = image.shape
    weight = 1.0 / noise_sigma if noise_sigma else 1.0

    param_names = ("theta_e", "axis_ratio", "position_angle", "beta_x", "beta_y",
                   "amplitude", "scale_radius", "background")
    initial = np.array([
        initial_theta_e, initial_axis_ratio, initial_position_angle,
        initial_source.beta_x, initial_source.beta_y, initial_source.amplitude,
        initial_source.scale_radius_arcsec, 0.0,
    ])
    lower = np.array([1e-3, 1e-3, -np.pi, -np.inf, -np.inf, 1e-6, 1e-3, -np.inf])
    upper = np.array([np.inf, 1.0, np.pi, np.inf, np.inf, np.inf, np.inf, np.inf])

    def residuals(params: np.ndarray) -> np.ndarray:
        theta_e, axis_ratio, position_angle, beta_x, beta_y, amplitude, scale_radius, background = params
        lens = SIELens(theta_e=theta_e, axis_ratio=float(np.clip(axis_ratio, 1e-3, 1.0)),
                       position_angle=position_angle)
        source = GaussianSource(beta_x=beta_x, beta_y=beta_y, amplitude=amplitude,
                                scale_radius_arcsec=scale_radius)
        model = render_lensed_image(lens, source, shape=(n_y, n_x),
                                    pixel_scale_arcsec=pixel_scale_arcsec,
                                    psf_fwhm_arcsec=psf_fwhm_arcsec, background=background)
        return ((image - model) * weight).ravel()

    result = least_squares(residuals, np.clip(initial, lower, upper), bounds=(lower, upper),
                           method="trf", max_nfev=2000)
    fitted = dict(zip(param_names, (float(x) for x in result.x)))
    return {
        "schema_version": SCHEMA_VERSION, **fitted,
        "converged": bool(result.success),
        "residual_rms": float(np.sqrt(np.mean(result.fun ** 2))),
        "n_evaluations": int(result.nfev),
    }
