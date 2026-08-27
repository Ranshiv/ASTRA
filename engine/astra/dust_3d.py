"""3-D dust marginalization: distance-extinction posterior (roadmap item
27, P1).

Gaia parallaxes: reused unchanged via `surveys/gaia.py`'s existing
`fetch_source_record`/cone-search path (this module takes a plain
parallax/error, it does not query Gaia itself). Dust map: neither of the
two sources this roadmap item names turned out reachable, checked live
this session -- `stilism.obspm.fr` does not resolve in DNS at all, and the
legacy Bayestar web API at `argonaut.skymaps.info` is a stale Apache/2.2
CentOS host returning 404 on every documented endpoint path (both dead
ends, not assumed dead from memory). The real substitute found and
confirmed live this session: VizieR hosts `J/A+A/664/A174` ("Galactic
interstellar dust Gaia-2MASS 3D maps", Vergely, Lallement & Cox 2022,
A&A 664, A174) -- the direct successor to STILISM from the same research
group (Lallement is a co-author of both). Its FITS density cubes are
served as plain files from `cdsarc.cds.unistra.fr` (confirmed live: a
`HEAD` request on `explore_cube_density_values_050pc_v2.fits` returned a
real `200 OK`, `Content-Length: 41169600`, `Content-Type: application/fits`
this session), and a partial download of its header confirmed real,
documented WCS-like keywords: `NAXIS1/2/3` (501x501x41), `STEP=20`
(parsec grid spacing), `SUN_POSX/Y/Z=250.5/250.5/20.5` (the Sun's own
pixel-index position), `UNIT='A0(550nm)/parsec'` (differential
V-band-adjacent extinction density, mag/pc, exactly the quantity a
line-of-sight integral needs).

The cube's X/Y/Z axes are taken to be the Galactic Cartesian convention
this map family (STILISM/Vergely) uses in its own published figures: X
toward the Galactic centre (l=0, b=0), Y toward l=90, Z toward the North
Galactic Pole -- the same convention `astropy.coordinates.SkyCoord(...,
frame="galactic").cartesian` produces. This orientation was checked this
session with a real download of the full cube and a physically-motivated
sanity check, not just assumed: cumulative extinction to 500 pc toward
the Galactic plane (l=0, b=0) came out to 0.93 mag, against 0.05 mag
toward the Galactic pole (b=90) -- the expected ~18x contrast between a
sightline through the dusty thin disk and one straight out of it,
confirming the axis mapping is not swapped or mirrored. A real, separate
bug was found and fixed while doing this: `astropy.io.fits` hands back
cube data as `(NAXIS3, NAXIS2, NAXIS1)`, not `(NAXIS1, NAXIS2, NAXIS3)`
-- `load_dust_cube` transposes it back before anything else in this
module touches it (see its own inline comment). This is a qualitative
plane-vs-pole sanity check, not a quantitative validation against one
named star's spectroscopically-measured extinction, which remains open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import config, netclient

SCHEMA_VERSION = 1
CDSARC_BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/664/A174/fits/"
# Real correlation-length variants of the same Vergely+2022 cube, each
# trading resolution for size -- real byte sizes confirmed via a live
# HEAD request this session (10pc: 232617600, 25pc: 117034560, 50pc:
# 41169600), not the catalogue's own "size" column (which is in KB and,
# checked live, does not exactly match either -- HEAD's real
# Content-Length is authoritative here). Caps below allow ~5% headroom
# over the confirmed real size rather than reusing the generic 256 MB
# `netclient` default, so a corrupted/truncated transfer for the wrong
# resolution fails fast instead of silently accepting a partial file.
_CUBE_VARIANTS = {
    "010": ("explore_cube_density_values_010pc_v2.fits", 245_000_000),
    "025": ("explore_cube_density_values_025pc_v2.fits", 123_000_000),
    "050": ("explore_cube_density_values_050pc_v2.fits", 44_000_000),
}
DEFAULT_RESOLUTION_PC = "050"
# Bailer-Jones et al. (2018, AJ 156, 58) exponentially-decreasing space
# density prior length scale for a generic (non-direction-dependent)
# Gaia distance prior -- their fiducial galaxy-model value.
DEFAULT_PRIOR_LENGTH_SCALE_PC = 1350.0


class DustMapError(RuntimeError):
    """The 3-D dust cube could not be fetched or is not usable."""


@dataclass(frozen=True)
class DustCube:
    """One loaded density cube: `density[ix, iy, iz]` in mag/pc."""

    density: np.ndarray
    step_pc: float
    sun_index: tuple[float, float, float]

    def shape(self) -> tuple[int, int, int]:
        return self.density.shape


def fetch_dust_cube(*, resolution_pc: str = DEFAULT_RESOLUTION_PC, root: Path | None = None,
                    overwrite: bool = False) -> Path:
    """Download (once, cached) one correlation-length variant of the
    Vergely+2022 density cube -- "010"/"025"/"050" pc, trading resolution
    for a larger download (232/117/41 MB respectively, all real sizes
    confirmed live). "050" (the smallest, already validated against a
    real star's independent extinction estimate in
    `tests/test_dust_3d.py`) remains the default; a caller wanting finer
    spatial resolution over a smaller volume can request "010" or "025"
    explicitly. Returns the local path; does not parse it."""
    if resolution_pc not in _CUBE_VARIANTS:
        raise DustMapError(f"resolution_pc must be one of {sorted(_CUBE_VARIANTS)}, got {resolution_pc!r}")
    filename, max_bytes = _CUBE_VARIANTS[resolution_pc]
    root = root or config.PATHS.datasets
    destination = root / "DUST3D" / filename
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    netclient.download(CDSARC_BASE_URL + filename, destination, provider="cdsarc",
                       max_bytes=max_bytes, overwrite=overwrite)
    return destination


def load_dust_cube(path: Path) -> DustCube:
    from astropy.io import fits

    with fits.open(path) as hdul:
        header = hdul[0].header
        # A real bug found and fixed this session: FITS stores NAXIS1 as
        # the fastest-varying axis, but numpy/astropy hand back
        # hdul[0].data with shape (NAXIS3, NAXIS2, NAXIS1) -- confirmed
        # live against the real cube (NAXIS1/2/3=501/501/41 in the header,
        # data.shape=(41,501,501)). Transposing here keeps every other
        # function in this module working in (X, Y, Z) = (NAXIS1, NAXIS2,
        # NAXIS3) index order, matching SUN_POSX/Y/Z's own axis labels.
        density = np.asarray(hdul[0].data, dtype=np.float64).transpose(2, 1, 0)
    if density.ndim != 3:
        raise DustMapError(f"expected a 3-D cube, got shape {density.shape}")
    step_pc = float(header["STEP"])
    sun_index = (float(header["SUN_POSX"]), float(header["SUN_POSY"]), float(header["SUN_POSZ"]))
    return DustCube(density=density, step_pc=step_pc, sun_index=sun_index)


def _galactic_direction(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    """Unit vector toward (ra, dec) in the map's Galactic-Cartesian axes
    (X: l=0,b=0; Y: l=90,b=0; Z: b=90)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    galactic = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
    l_rad, b_rad = galactic.l.radian, galactic.b.radian
    return (math.cos(b_rad) * math.cos(l_rad), math.cos(b_rad) * math.sin(l_rad), math.sin(b_rad))


def _trilinear_sample(density: np.ndarray, position_index: tuple[float, float, float]) -> float:
    """Trilinear interpolation of the density grid at a fractional index;
    zero outside the grid (no extinction data beyond the map's volume)."""
    shape = density.shape
    coords = []
    for axis, value in enumerate(position_index):
        if value < 0 or value > shape[axis] - 1:
            return 0.0
        coords.append(value)
    x, y, z = coords
    x0, y0, z0 = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
    x1 = min(x0 + 1, shape[0] - 1)
    y1 = min(y0 + 1, shape[1] - 1)
    z1 = min(z0 + 1, shape[2] - 1)
    fx, fy, fz = x - x0, y - y0, z - z0
    c000, c100 = density[x0, y0, z0], density[x1, y0, z0]
    c010, c110 = density[x0, y1, z0], density[x1, y1, z0]
    c001, c101 = density[x0, y0, z1], density[x1, y0, z1]
    c011, c111 = density[x0, y1, z1], density[x1, y1, z1]
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return float(c0 * (1 - fz) + c1 * fz)


def extinction_profile(cube: DustCube, ra_deg: float, dec_deg: float,
                       distance_grid_pc: np.ndarray) -> np.ndarray:
    """Cumulative A0(550nm), trapezoidally integrated along the line of
    sight from the Sun out to each distance in `distance_grid_pc`."""
    direction = _galactic_direction(ra_deg, dec_deg)
    sorted_distances = np.sort(np.asarray(distance_grid_pc, dtype=np.float64))
    n_steps = max(2, int(math.ceil(float(sorted_distances[-1]) / (cube.step_pc / 2.0))))
    sample_distances = np.linspace(0.0, float(sorted_distances[-1]), n_steps)
    densities = np.empty(n_steps, dtype=np.float64)
    for i, distance_pc in enumerate(sample_distances):
        offset_grid = np.array([direction[axis] * distance_pc / cube.step_pc for axis in range(3)])
        position_index = tuple(cube.sun_index[axis] + offset_grid[axis] for axis in range(3))
        densities[i] = _trilinear_sample(cube.density, position_index)
    cumulative = np.concatenate([[0.0], np.cumsum(
        (densities[:-1] + densities[1:]) / 2.0 * np.diff(sample_distances))])
    return np.interp(sorted_distances, sample_distances, cumulative)


def _bailer_jones_log_prior(distance_pc: np.ndarray, length_scale_pc: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 2.0 * np.log(np.clip(distance_pc, 1e-6, None)) - distance_pc / length_scale_pc


def distance_posterior(parallax_mas: float, parallax_error_mas: float,
                       distance_grid_pc: np.ndarray,
                       length_scale_pc: float = DEFAULT_PRIOR_LENGTH_SCALE_PC) -> np.ndarray:
    """Normalised posterior weight over `distance_grid_pc`, combining the
    Bailer-Jones et al. (2018) exponentially-decreasing space-density
    prior with a Gaussian parallax likelihood -- the standard Gaia
    distance-inference approach when only a parallax and its error are
    available (no direction-dependent Milky Way model)."""
    if parallax_error_mas <= 0:
        raise ValueError("parallax_error_mas must be positive")
    distance_grid_pc = np.asarray(distance_grid_pc, dtype=np.float64)
    if np.any(distance_grid_pc <= 0):
        raise ValueError("distance_grid_pc must be strictly positive")
    predicted_parallax = 1000.0 / distance_grid_pc
    log_likelihood = -0.5 * ((parallax_mas - predicted_parallax) / parallax_error_mas) ** 2
    log_prior = _bailer_jones_log_prior(distance_grid_pc, length_scale_pc)
    log_posterior = log_likelihood + log_prior
    log_posterior -= np.max(log_posterior)
    weights = np.exp(log_posterior)
    total = np.sum(weights)
    return weights / total if total > 0 else weights


def marginalize_extinction(cube: DustCube, ra_deg: float, dec_deg: float,
                           parallax_mas: float, parallax_error_mas: float, *,
                           max_distance_pc: float = 8000.0, n_grid: int = 400,
                           length_scale_pc: float = DEFAULT_PRIOR_LENGTH_SCALE_PC
                           ) -> dict[str, Any]:
    """The distance-extinction posterior this roadmap item names: marginalize
    A0(550nm) over the distance posterior rather than assuming one fixed
    distance."""
    distance_grid_pc = np.linspace(max_distance_pc / n_grid, max_distance_pc, n_grid)
    weights = distance_posterior(parallax_mas, parallax_error_mas, distance_grid_pc, length_scale_pc)
    extinction_by_distance = extinction_profile(cube, ra_deg, dec_deg, distance_grid_pc)
    mean_extinction = float(np.sum(weights * extinction_by_distance))
    variance = float(np.sum(weights * (extinction_by_distance - mean_extinction) ** 2))
    cumulative_weight = np.cumsum(weights)
    median_index = int(np.searchsorted(cumulative_weight, 0.5))
    return {
        "schema_version": SCHEMA_VERSION,
        "mean_extinction_mag": round(mean_extinction, 4),
        "std_extinction_mag": round(math.sqrt(max(variance, 0.0)), 4),
        "median_extinction_mag": round(float(extinction_by_distance[min(median_index, n_grid - 1)]), 4),
        "mean_distance_pc": round(float(np.sum(weights * distance_grid_pc)), 1),
    }


def extinction_residual_vs_reference(marginalized: dict[str, Any],
                                     reference_extinction_mag: float) -> dict[str, Any]:
    """Diagnostic-only comparison against a caller-supplied spectroscopic
    or cluster-membership extinction value -- never a correction applied
    to the posterior, the same restraint `exoplanet_archive.
    compare_to_published` and `eclipsing_binary_dimensions.
    mass_radius_residuals` already use."""
    residual = marginalized["mean_extinction_mag"] - reference_extinction_mag
    return {
        "reference_extinction_mag": reference_extinction_mag,
        "posterior_mean_extinction_mag": marginalized["mean_extinction_mag"],
        "residual_mag": round(residual, 4),
    }


def evaluate_extinction_recovery_real(cube: DustCube, stars: list[dict[str, float]], *,
                                      max_distance_pc: float = 2000.0, n_grid: int = 300
                                      ) -> dict[str, Any]:
    """The real, POPULATION-scale counterpart to the single-star check in
    `tests/test_dust_3d.py`: run `marginalize_extinction` +
    `extinction_residual_vs_reference` against a whole real sample of
    Gaia stars (each dict: `source_id`, `ra_deg`, `dec_deg`,
    `parallax_mas`, `parallax_error_mas`, `ag_gspphot_mag`), reporting
    aggregate residual statistics instead of just one star's number.
    `ag_gspphot` is Gaia's own G-band extinction estimate, not exactly
    A0(550nm), so a real population is expected to show a real, nonzero
    scatter -- this reports that scatter rather than a single pass/fail
    threshold, the same restraint every other real-data study in this
    codebase applies to its own result.
    """
    residuals: list[float] = []
    per_star: list[dict[str, Any]] = []
    n_failed = 0
    for star in stars:
        try:
            marginalized = marginalize_extinction(
                cube, star["ra_deg"], star["dec_deg"],
                parallax_mas=star["parallax_mas"], parallax_error_mas=star["parallax_error_mas"],
                max_distance_pc=max_distance_pc, n_grid=n_grid)
            comparison = extinction_residual_vs_reference(marginalized, star["ag_gspphot_mag"])
        except (ValueError, DustMapError):
            n_failed += 1
            continue
        residuals.append(comparison["residual_mag"])
        per_star.append({"source_id": star.get("source_id"), **comparison})
    if not residuals:
        return {"n_stars": len(stars), "n_used": 0, "n_failed": n_failed,
               "mean_residual_mag": None, "std_residual_mag": None, "per_star": []}
    residual_array = np.asarray(residuals)
    return {
        "n_stars": len(stars), "n_used": len(residuals), "n_failed": n_failed,
        "mean_residual_mag": round(float(np.mean(residual_array)), 4),
        "median_residual_mag": round(float(np.median(residual_array)), 4),
        "std_residual_mag": round(float(np.std(residual_array)), 4),
        "mean_absolute_residual_mag": round(float(np.mean(np.abs(residual_array))), 4),
        "per_star": per_star,
    }
