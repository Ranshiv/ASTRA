"""Live calibrated spectroscopy: wavelength-calibration and instrument-
response diagnostics for an already-acquired spectrum (roadmap item 24, P0).

Two bounded, explainable diagnostics, matching `spectral_features.py`'s own
stated scope (continuum/S/N/line statistics, not an astrophysical classifier
or a physical instrument-response simulator):

* `independent_redshift_from_lines` cross-correlates continuum-subtracted
  S/N peaks/troughs against a small, standard list of strong rest-frame
  spectral lines (values are the well-established vacuum wavelengths used by
  the SDSS pipeline and NIST ASD) over a coarse-to-fine redshift grid. This
  gives an independent redshift estimate that does not depend on whichever
  pipeline produced the survey's own released `z` -- the wavelength-
  calibration check this module's name promises.
* `continuum_smoothness_residual` is a bounded proxy for instrument-response
  quality: a genuinely flux-calibrated spectrum's continuum should be well
  described by a low-order polynomial trend; a large residual after
  subtracting one is a plausible (not definitive) sign of a stitching or
  response-calibration artifact.

Both operate on any wavelength/flux/error triple already fetched by a survey
connector (`sdss.extract_sdss_spectrum`'s inputs, or a future DESI/LAMOST
equivalent) -- this module does not fetch data itself.

A genuine, stated limitation of `independent_redshift_from_lines`, confirmed
by a synthetic single-line recovery check while building this module: a
SINGLE detected line cannot, in principle, disambiguate which rest-frame
line it is (H-alpha at z=0.02 and Lyman-alpha at z=4.5 both place a line at
the same observed wavelength), so a spectrum with only one significant
feature can score several very different redshifts equally well and this
function is not guaranteed to pick the astrophysically correct one. This is
the same reason real spectroscopic pipelines require multiple line
coincidences before trusting a redshift -- not a bug this module can fix
without adding physical priors (expected line ratios, a template) outside
its stated bounded scope. It is reliable when multiple real lines are
present, which is the common case for any spectrum with a measurable z.
"""

from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT_KMS = 299_792.458

# Strong, commonly used rest-frame vacuum wavelengths (Angstrom), the same
# lines the SDSS spectroscopic pipeline itself targets for redshift fitting
# (see SDSS's own `emLines.par`/pipeline documentation) and NIST ASD.
# [O II] is reported as its blended doublet centroid (3726.03/3728.82), not
# a resolved pair -- this module's peak-finder operates on a smoothed S/N
# array, not individual-line-resolved profiles (that is item 25's job).
DEFAULT_REST_LINES: dict[str, float] = {
    "Lyman-alpha": 1215.67,
    "C IV": 1549.48,
    "Mg II": 2799.12,
    "[O II]": 3727.42,
    "Ca II K": 3933.66,
    "Ca II H": 3968.47,
    "H-delta": 4101.73,
    "H-gamma": 4340.46,
    "H-beta": 4861.35,
    "[O III] 4959": 4958.91,
    "[O III] 5007": 5006.84,
    "[O I]": 6300.30,
    "H-alpha": 6562.79,
    "[N II]": 6583.45,
    "[S II] 6716": 6716.44,
    "[S II] 6731": 6730.82,
}

DEFAULT_SNR_THRESHOLD = 5.0
DEFAULT_VELOCITY_TOLERANCE_KMS = 500.0


def _prepare(wavelength, flux, error) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and sort a spectrum, matching `spectral_features._finite_arrays`'s
    contract (kept as an independent, small implementation here rather than
    importing that private helper across modules).
    """
    wave = np.asarray(wavelength, dtype=np.float64)
    values = np.asarray(flux, dtype=np.float64)
    errors = np.asarray(error, dtype=np.float64)
    if not (wave.ndim == values.ndim == errors.ndim == 1):
        raise ValueError("wavelength/flux/error must be one-dimensional")
    if not (len(wave) == len(values) == len(errors)):
        raise ValueError("wavelength/flux/error must have equal lengths")
    mask = np.isfinite(wave) & np.isfinite(values) & np.isfinite(errors) & (errors >= 0)
    wave, values, errors = wave[mask], values[mask], errors[mask]
    order = np.argsort(wave, kind="stable")
    wave, values, errors = wave[order], values[order], errors[order]
    if len(wave) < 5 or np.any(np.diff(wave) <= 0):
        raise ValueError("spectrum needs at least five strictly increasing finite wavelengths")
    return wave, values, errors


def _continuum(values: np.ndarray) -> np.ndarray:
    """Rolling-median continuum, not a rolling mean.

    A real bug, found and fixed via a synthetic single-line recovery check
    while building this module: a plain moving average (a plausible first
    choice, and `spectral_features.py`'s own approach for its aggregate
    statistics) leaks a strong, narrow line's own flux into the "continuum"
    estimate at nearby wavelengths, understating it there -- confirmed
    directly, an isolated, cleanly-detected 56-sigma emission line produced
    a RING of ten spurious ~5-sigma "absorption" detections flanking it
    before this fix, each one just this module's own continuum artifact,
    not a real spectral feature. A median filter is robust to that specific
    failure mode as long as the line occupies less than half the window,
    which `find_candidate_lines`' peak detection already implicitly
    assumes. `mode="nearest"` also avoids `np.convolve`'s implicit
    zero-padding at the array edges (the boundary-contamination fix in
    `find_candidate_lines` is kept regardless, since nearest-padding still
    biases the estimate somewhat within the last half-window).
    """
    from scipy.ndimage import median_filter

    window = max(5, min(101, (len(values) // 20) * 2 + 1))
    return median_filter(values, size=window, mode="nearest")


def find_candidate_lines(wavelength, flux, error, *,
                         snr_threshold: float = DEFAULT_SNR_THRESHOLD) -> list[dict]:
    """Continuum-subtracted S/N peaks (emission) and troughs (absorption)
    as candidate spectral-line wavelengths, with the same robust-sigma
    error fallback `spectral_features.extract` uses.

    A real bug, found and fixed via a synthetic pure-noise check while
    building this function: `_continuum`'s `np.convolve(..., mode="same")`
    implicitly zero-pads outside the array, so the moving-average estimate
    collapses toward zero within half a window of either edge while the
    real flux does not -- confirmed directly, this produced spurious
    S/N > 40 "line" detections at both edges of an otherwise pure-noise
    spectrum. The half-window margin at each edge is excluded from peak
    detection below; it is real continuum-estimation contamination, not a
    site where a genuine line could be distinguished from noise anyway.
    """
    from scipy.signal import find_peaks

    wave, values, errors = _prepare(wavelength, flux, error)
    window = max(5, min(101, (len(values) // 20) * 2 + 1))
    continuum = _continuum(values)
    residual = values - continuum
    fallback_sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    fallback_sigma = float(fallback_sigma) if np.isfinite(fallback_sigma) and fallback_sigma > 0 else 1.0
    effective_error = np.where(errors > 0, errors, fallback_sigma)
    snr = residual / effective_error

    margin = window // 2
    valid = np.zeros(len(values), dtype=bool)
    valid[margin: len(values) - margin] = True
    masked_snr = np.where(valid, snr, -np.inf)

    emission_idx, _ = find_peaks(masked_snr, height=snr_threshold)
    absorption_idx, _ = find_peaks(-masked_snr, height=snr_threshold)

    candidates = [{"wavelength": float(wave[i]), "snr": float(snr[i]), "kind": "emission"}
                 for i in emission_idx]
    candidates += [{"wavelength": float(wave[i]), "snr": float(-snr[i]), "kind": "absorption"}
                  for i in absorption_idx]
    return candidates


def independent_redshift_from_lines(wavelength, flux, error, *,
                                    rest_lines: dict[str, float] | None = None,
                                    z_grid: np.ndarray | None = None,
                                    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
                                    velocity_tolerance_kms: float = DEFAULT_VELOCITY_TOLERANCE_KMS
                                    ) -> dict:
    """A redshift estimate independent of any pipeline-released value.

    Grid-searches redshift, scoring each trial by how many rest-frame lines
    land within `velocity_tolerance_kms` of a detected S/N peak/trough,
    weighted by that peak's own significance and how close the match is.
    This is a coarse cross-correlation against a known line list, not a
    full template fit -- deliberately bounded, matching this module's
    stated scope. Returns `z_best=None` (not a fabricated 0.0) when no
    candidate line is found at all, or none matches at any trial redshift.
    """
    wave, values, errors = _prepare(wavelength, flux, error)
    # `rest_lines or DEFAULT_REST_LINES` would silently replace an explicit
    # empty dict with the default list (an empty dict is falsy) -- checked
    # for None specifically so an explicit `{}` is rejected below instead.
    rest_lines = DEFAULT_REST_LINES if rest_lines is None else rest_lines
    if not rest_lines:
        raise ValueError("rest_lines must be non-empty")

    candidates = find_candidate_lines(wave, values, errors, snr_threshold=snr_threshold)
    if not candidates:
        return {"z_best": None, "n_lines_matched": 0, "matches": [],
               "reason": "no significant line candidates found"}

    if z_grid is None:
        z_max = max(0.0, float(wave.max()) / min(rest_lines.values()) - 1.0)
        # A fixed point count over a variable-width [0, z_max] range is a
        # real bug, found via a synthetic-recovery check while building
        # this function: for a wide z_max the resulting grid spacing can
        # exceed the tolerance window being searched for, so a true
        # redshift falls between grid points and is missed even though a
        # real, matching line was detected. Grid spacing is tied directly
        # to `velocity_tolerance_kms` instead, oversampled 5x so the true
        # redshift always has a grid point within tolerance.
        dz_step = (velocity_tolerance_kms / SPEED_OF_LIGHT_KMS) / 5.0
        n_points = int(np.clip(z_max / max(dz_step, 1e-9), 100, 20_000)) + 1
        z_grid = np.linspace(0.0, z_max, n_points)
    else:
        z_grid = np.asarray(z_grid, dtype=np.float64)

    candidate_waves = np.array([c["wavelength"] for c in candidates])
    candidate_snrs = np.array([c["snr"] for c in candidates])
    rest_names = list(rest_lines.keys())
    rest_values = np.array([rest_lines[name] for name in rest_names])

    best_score = -1.0
    best_z: float | None = None
    best_matches: list[dict] = []
    for z in z_grid:
        observed = rest_values * (1.0 + z)
        in_range = (observed >= wave.min()) & (observed <= wave.max())
        if not np.any(in_range):
            continue
        score = 0.0
        matches: list[dict] = []
        for name, rest_wave, obs_wave in zip(
                np.asarray(rest_names)[in_range], rest_values[in_range], observed[in_range]):
            offsets_kms = np.abs(candidate_waves - obs_wave) / obs_wave * SPEED_OF_LIGHT_KMS
            nearest = int(np.argmin(offsets_kms))
            if offsets_kms[nearest] <= velocity_tolerance_kms:
                weight = candidate_snrs[nearest] * (1.0 - offsets_kms[nearest] / velocity_tolerance_kms)
                score += float(weight)
                matches.append({
                    "line": str(name), "rest_wavelength": float(rest_wave),
                    "expected_wavelength": float(obs_wave),
                    "matched_wavelength": float(candidate_waves[nearest]),
                    "velocity_offset_kms": float(offsets_kms[nearest]),
                })
        if score > best_score:
            best_score, best_z, best_matches = score, float(z), matches

    if best_z is None or not best_matches:
        return {"z_best": None, "n_lines_matched": 0, "matches": [],
               "reason": "no candidate line matched a rest line within tolerance "
                        "at any trial redshift"}

    return {"z_best": best_z, "score": float(best_score), "n_lines_matched": len(best_matches),
           "matches": best_matches, "velocity_tolerance_kms": velocity_tolerance_kms}


def continuum_smoothness_residual(wavelength, continuum, *, poly_degree: int = 3) -> dict:
    """How much of the continuum shape is NOT explained by a low-order
    polynomial trend -- a bounded proxy for instrument-response quality
    (see module docstring). Not a physical flux-calibration model.
    """
    if poly_degree < 1:
        raise ValueError("poly_degree must be at least 1")
    wave = np.asarray(wavelength, dtype=np.float64)
    cont = np.asarray(continuum, dtype=np.float64)
    mask = np.isfinite(wave) & np.isfinite(cont)
    wave, cont = wave[mask], cont[mask]
    if len(wave) < poly_degree + 2:
        raise ValueError(f"need at least {poly_degree + 2} finite continuum points")

    span = max(float(wave.max() - wave.min()), 1e-9)
    x = 2.0 * (wave - wave.min()) / span - 1.0
    coeffs = np.polyfit(x, cont, poly_degree)
    trend = np.polyval(coeffs, x)
    residual = cont - trend
    scale = np.maximum(np.abs(trend), np.finfo(float).eps)
    relative_residual = np.abs(residual) / scale

    return {
        "poly_degree": poly_degree,
        "median_relative_residual": float(np.median(relative_residual)),
        "max_relative_residual": float(np.max(relative_residual)),
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
    }


def calibration_report(wavelength, flux, error, *, released_z: float | None = None,
                       rest_lines: dict[str, float] | None = None,
                       z_grid: np.ndarray | None = None) -> dict:
    """The full instrument-response and wavelength-calibration report for
    one spectrum, combining both diagnostics and, when `released_z` is
    given, the residual against it (the metric roadmap item 24 names).
    """
    wave, values, errors = _prepare(wavelength, flux, error)
    redshift = independent_redshift_from_lines(
        wave, values, errors, rest_lines=rest_lines, z_grid=z_grid)
    continuum = _continuum(values)
    instrument_response = continuum_smoothness_residual(wave, continuum)

    if released_z is not None and redshift["z_best"] is not None:
        released_z = float(released_z)
        redshift = dict(redshift)
        redshift["released_z"] = released_z
        redshift["z_residual"] = redshift["z_best"] - released_z
        redshift["z_residual_fractional"] = (
            (redshift["z_best"] - released_z) / (1.0 + released_z))

    return {"redshift": redshift, "instrument_response": instrument_response}
