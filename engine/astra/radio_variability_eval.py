"""Validation for `radio_variability.py`: cross-survey association recall
and spectral-index uncertainty (roadmap item 22's two named metrics).

Cross-survey association recall reuses `frb.within_error_ellipse` UNCHANGED
for the burst-position half of the check (do not reimplement positional
matching, exactly as `frb.top_k_counterpart_recall` already demonstrates
for a related problem). Spectral-index uncertainty uses Monte Carlo error
propagation (perturb each real flux measurement within its own reported
error and refit), not row-resampling bootstrap: a genuine VLASS+NVSS
measurement is typically only 2-3 frequency points, far too few for
resampling ROWS to be meaningful, but perfectly suited to propagating each
point's own Gaussian flux uncertainty through the fit many times.
"""

from __future__ import annotations

import numpy as np

from . import radio_variability as rv
from .frb import FrbBurst, within_error_ellipse
from .surveys import vlass

DEFAULT_SIGMA_THRESHOLD = 3.0
DEFAULT_MATCH_RADIUS_ARCSEC = 5.0
# VLASS's nominal S-band survey frequency (2-4 GHz band, quoted at its
# band-center); paired here with NVSS's well-known 1.4 GHz for a real
# two-point spectral index.
VLASS_NOMINAL_FREQUENCY_GHZ = 3.0


class RadioVariabilityEvalError(ValueError):
    """A radio-variability evaluation study could not be run."""


def _angular_offset_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    """RA offset scaled by cos(dec), matching `frb.within_error_ellipse`'s
    own convention."""
    delta_ra = (ra1_deg - ra2_deg) * np.cos(np.radians(dec2_deg)) * 3600.0
    delta_dec = (dec1_deg - dec2_deg) * 3600.0
    return float(np.sqrt(delta_ra ** 2 + delta_dec ** 2))


def nearest_radio_counterpart(ra_deg: float, dec_deg: float, radio_sources: list[dict],
                              match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC
                              ) -> dict | None:
    """The nearest `radio_sources` entry (each a dict with `object_id`/
    `ra_deg`/`dec_deg`) within `match_radius_arcsec`, or `None` if none
    falls inside it."""
    best: dict | None = None
    best_offset: float | None = None
    for source in radio_sources:
        offset = _angular_offset_arcsec(ra_deg, dec_deg, source["ra_deg"], source["dec_deg"])
        if offset <= match_radius_arcsec and (best_offset is None or offset < best_offset):
            best, best_offset = source, offset
    if best is None:
        return None
    return {**best, "offset_arcsec": best_offset}


def cross_survey_association_recall(trials: list[dict], *,
                                    match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
                                    sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD) -> dict:
    """Each trial is `{"burst": FrbBurst, "query_ra_deg", "query_dec_deg",
    "radio_sources": [{"object_id", "ra_deg", "dec_deg"}, ...],
    "true_counterpart_id"}` -- a position with a KNOWN true FRB burst
    identity and a KNOWN true radio-counterpart identity among the
    candidates. Reports two joint recall numbers: whether
    `frb.within_error_ellipse` correctly associates the query position
    with its burst, and whether the nearest candidate radio source within
    `match_radius_arcsec` is the true counterpart -- a real (if VLASS-
    Epoch-1-limited; see `surveys/vlass.py`'s own docstring) stand-in for
    a genuine cross-survey FRB/radio-transient association check.
    """
    n = len(trials)
    if n == 0:
        raise RadioVariabilityEvalError("trials must be non-empty")

    n_burst_associated = 0
    n_counterpart_correct = 0
    for trial in trials:
        inside, _offset = within_error_ellipse(
            trial["query_ra_deg"], trial["query_dec_deg"], trial["burst"], sigma_threshold)
        n_burst_associated += int(inside)

        match = nearest_radio_counterpart(
            trial["query_ra_deg"], trial["query_dec_deg"], trial["radio_sources"],
            match_radius_arcsec)
        if match is not None and match.get("object_id") == trial["true_counterpart_id"]:
            n_counterpart_correct += 1

    return {
        "n_trials": n, "match_radius_arcsec": match_radius_arcsec,
        "sigma_threshold": sigma_threshold,
        "burst_association_recall": n_burst_associated / n,
        "counterpart_recall": n_counterpart_correct / n,
    }


def spectral_index_monte_carlo(frequency_ghz, flux_mjy, flux_err_mjy, *,
                               n_samples: int = 5000, seed: int = 42,
                               levels: tuple[float, ...] = (0.68, 0.9)) -> dict:
    """Monte Carlo error-propagation uncertainty on the fitted spectral
    index (see module docstring for why this, not row-resampling
    bootstrap). A perturbed draw that goes non-positive (a real
    possibility for a low-S/N flux point) is skipped and counted, not
    clipped to a fabricated small positive value.
    """
    frequency_ghz = np.asarray(frequency_ghz, dtype=np.float64)
    flux_mjy = np.asarray(flux_mjy, dtype=np.float64)
    flux_err_mjy = np.asarray(flux_err_mjy, dtype=np.float64)
    point_estimate = rv.fit_spectral_index(frequency_ghz, flux_mjy, flux_err_mjy)

    rng = np.random.default_rng(seed)
    alphas: list[float] = []
    n_rejected = 0
    for _ in range(n_samples):
        perturbed = rng.normal(flux_mjy, flux_err_mjy)
        if np.any(perturbed <= 0):
            n_rejected += 1
            continue
        try:
            fit = rv.fit_spectral_index(frequency_ghz, perturbed, flux_err_mjy)
        except rv.RadioVariabilityError:
            n_rejected += 1
            continue
        alphas.append(fit["alpha"])

    if not alphas:
        return {"point_estimate": point_estimate, "n_valid_samples": 0,
               "n_rejected": n_rejected, "median_alpha": None, "intervals": {}}

    alpha_array = np.asarray(alphas)
    intervals: dict = {}
    for level in levels:
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(alpha_array, [tail, 1.0 - tail])
        intervals[str(level)] = [float(low), float(high)]

    return {"point_estimate": point_estimate, "n_valid_samples": len(alphas),
           "n_rejected": n_rejected, "median_alpha": float(np.median(alpha_array)),
           "intervals": intervals}


def real_two_point_spectral_index(ra_deg: float, dec_deg: float, vlass_flux_mjy: float,
                                  vlass_flux_err_mjy: float, *, nvss_search_radius_arcsec: float = 15.0,
                                  n_samples: int = 5000, seed: int = 42) -> dict | None:
    """A genuine VLASS (~3 GHz) + NVSS (1.4 GHz) two-point spectral index
    at a real position, via `vlass.query_nvss_flux_1_4ghz` -- `None` (not
    a fabricated index) when no NVSS counterpart is found within
    `nvss_search_radius_arcsec`.
    """
    nvss = vlass.query_nvss_flux_1_4ghz(ra_deg, dec_deg, radius_arcsec=nvss_search_radius_arcsec)
    if nvss is None:
        return None
    frequency_ghz = np.array([nvss["frequency_ghz"], VLASS_NOMINAL_FREQUENCY_GHZ])
    flux_mjy = np.array([nvss["flux_mjy"], vlass_flux_mjy])
    flux_err_mjy = np.array([nvss["flux_err_mjy"], vlass_flux_err_mjy])
    result = spectral_index_monte_carlo(frequency_ghz, flux_mjy, flux_err_mjy,
                                        n_samples=n_samples, seed=seed)
    result["nvss_name"] = nvss["nvss_name"]
    return result
