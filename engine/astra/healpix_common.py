"""Shared HEALPix pixel-membership primitives.

Three modules in this codebase each grew their own HEALPix lookup because
each was solving a locally different problem at the time it was written:

  - `gw.py` builds a DENSE per-pixel probability array by histogram-binning
    GWOSC posterior samples, and needs a cumulative-probability credible
    level at one exact point.
  - `association.py` (`_healpix_probability`) consumes a SPARSE list of
    `{"index", "probability"}` pixels straight from a generic event packet's
    `localization.pixels` field -- raw density, credible level not yet
    computed.
  - `frb.py` (`localization_membership`) consumes a SPARSE list of
    `{"index" (as "ipix"), "confidence_level" (as "CL")}` pairs -- CHIME/FRB
    publishes the credible level itself, already computed upstream; this is
    a pure lookup, no cumulative-sum needed.

All three ultimately answer the same question -- "how confident is a map
that the true position is at (ra, dec)?" -- with the same
`{pixel_probability, credible_level, in_credible_region}`-shaped answer.
This module is that one shared answer, covering all three input shapes, so
new code (the cross-event correlation work in `association.py`) has one
correct implementation to call instead of writing a fourth.

`gw.py` and `frb.py` are NOT switched over to call through this module in
this pass: both are already shipped, individually well-tested, and their
own inline implementations are proven correct by their existing test
suites. `tests/test_healpix_common.py` proves this module's
`pixel_probability` reproduces each of their computations bit-for-bit on
shared fixtures, which is the correctness bar the project's own plan sets
before a migration; the migration itself is a low-risk mechanical follow-up
deliberately left undone here to avoid unnecessary churn in two modules
that already work, matching this codebase's general preference for proven
convergence over defensive rewriting (see `docs/LIMITATIONS.md`'s repeated
"measured but not adopted" pattern for calibrated-but-unadopted work).
`association.py`'s own `_healpix_probability` DOES switch over, since it is
the shape most directly reused by the new cross-event correlation code.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

DEFAULT_CREDIBLE_THRESHOLD = 0.90
# Fraction whose enclosed area stands in for a "1-sigma-equivalent" radius
# for a 2D Gaussian-like distribution (the enclosed probability of a
# circular 1-sigma region under an isotropic 2D Gaussian is 1 - exp(-1/2)).
ONE_SIGMA_EQUIVALENT_FRACTION = 1.0 - math.exp(-0.5)


def _target_pixel(ra_deg: float, dec_deg: float, nside: int, order: str) -> int:
    import astropy.units as u
    from astropy_healpix import HEALPix

    healpix = HEALPix(nside=nside, order=order)
    return int(healpix.lonlat_to_healpix(float(ra_deg) * u.deg, float(dec_deg) * u.deg))


def pixel_probability(ra_deg: float, dec_deg: float, *, nside: int, order: str = "nested",
                      probability_map: np.ndarray | None = None,
                      sparse_pixels: list[dict[str, Any]] | None = None,
                      precomputed_credible_levels: bool = False,
                      credible_threshold: float = DEFAULT_CREDIBLE_THRESHOLD
                      ) -> dict[str, Any] | None:
    """Pixel membership and credible-region containment at one sky position.

    Exactly one of `probability_map` (dense, length `12*nside**2`) or
    `sparse_pixels` (a list of dicts) must be supplied.

    For `sparse_pixels`, each item is either `{"index", "probability"}` (raw
    density; `precomputed_credible_levels=False`, the default -- credible
    level is computed here by descending-probability cumulative sum, the
    same math `association._healpix_probability` used inline before this
    module existed) or `{"index", "confidence_level"}` (the credible level
    is already computed upstream; `precomputed_credible_levels=True` -- the
    same shape CHIME/FRB's published sparse maps carry, matching
    `frb.localization_membership`'s pure-lookup case).

    Returns `None` when no map data is usable (empty map, unusable inputs).
    A target position covered by neither a dense map's nonzero pixels nor a
    sparse map's listed pixels is the least-confident answer the map can
    give -- `credible_level: 1.0`, not `None` -- matching the existing
    "outside every listed pixel" convention in both `gw.py` and `frb.py`.
    """
    if (probability_map is None) == (sparse_pixels is None):
        raise ValueError("supply exactly one of probability_map or sparse_pixels")
    if nside < 1:
        return None
    target = _target_pixel(ra_deg, dec_deg, nside, order)

    if probability_map is not None:
        probability = np.asarray(probability_map, dtype=np.float64)
        if probability.size == 0 or not np.isfinite(probability).any():
            return None
        total = float(probability.sum())
        if total <= 0:
            return None
        density = float(probability[target]) if 0 <= target < probability.size else 0.0
        order_idx = np.argsort(probability)[::-1]
        cumulative = np.cumsum(probability[order_idx])
        position = np.where(order_idx == target)[0]
        credible_level = float(cumulative[position[0]]) if len(position) else 1.0
        return {
            "pixel_probability": density / total,
            "credible_level": credible_level,
            "in_credible_region": credible_level <= credible_threshold,
        }

    clean = [item for item in (sparse_pixels or []) if isinstance(item, dict)
             and item.get("index") is not None]
    if not clean:
        return None

    if precomputed_credible_levels:
        match = next((item for item in clean if int(item["index"]) == target), None)
        if match is None:
            return {"pixel_probability": None, "credible_level": 1.0,
                    "in_credible_region": False}
        level = float(match.get("confidence_level", match.get("CL", 1.0)))
        return {"pixel_probability": None, "credible_level": level,
                "in_credible_region": level <= credible_threshold}

    probabilities = []
    target_probability = 0.0
    for item in clean:
        probability = item.get("probability")
        if probability is None:
            continue
        probability = float(probability)
        if probability < 0:
            continue
        probabilities.append(probability)
        if int(item["index"]) == target:
            target_probability += probability
    if not probabilities:
        return None
    ordered = sorted(probabilities, reverse=True)
    cumulative = 0.0
    credible_level = 1.0
    for probability in ordered:
        cumulative += probability
        if target_probability >= probability:
            credible_level = min(1.0, cumulative)
            break
    return {
        "pixel_probability": target_probability,
        "credible_level": credible_level,
        "in_credible_region": credible_level <= credible_threshold,
    }


def effective_point_and_radius(*, nside: int, order: str = "nested",
                               probability_map: np.ndarray | None = None,
                               sparse_pixels: list[dict[str, Any]] | None = None,
                               precomputed_credible_levels: bool = False,
                               credible_fraction: float = ONE_SIGMA_EQUIVALENT_FRACTION
                               ) -> dict[str, Any] | None:
    """Collapse an extended localization map to a point-plus-radius summary.

    Used only where a caller genuinely needs a single effective position and
    a 1-sigma-equivalent radius (the cross-event Bayes-factor statistic in
    `association.event_to_event_correlation` is the one caller today) --
    NOT a substitute for `pixel_probability`'s exact membership test, which
    should always be preferred when both a map and a specific target
    position are available. The peak-probability pixel stands in for the
    point estimate; the radius is derived from the on-sky area of the
    smallest pixel set whose cumulative probability reaches
    `credible_fraction`, converted to an equivalent circular radius
    (`sqrt(area / pi)`). This is a deliberately coarse approximation --
    documented, not asserted as a rigorous shape fit -- appropriate for a
    coincidence screen, not a publication-grade localization comparison.
    """
    import astropy.units as u
    from astropy_healpix import HEALPix

    if (probability_map is None) == (sparse_pixels is None):
        raise ValueError("supply exactly one of probability_map or sparse_pixels")
    healpix = HEALPix(nside=nside, order=order)
    pixel_area_deg2 = float(healpix.pixel_area.to(u.deg ** 2).value)

    if probability_map is not None:
        probability = np.asarray(probability_map, dtype=np.float64)
        total = float(probability.sum())
        if probability.size == 0 or total <= 0:
            return None
        peak_pixel = int(np.argmax(probability))
        order_idx = np.argsort(probability)[::-1]
        cumulative = np.cumsum(probability[order_idx] / total)
        n_pixels = int(np.searchsorted(cumulative, credible_fraction) + 1)
    else:
        clean = [item for item in (sparse_pixels or []) if isinstance(item, dict)
                and item.get("index") is not None]
        if not clean:
            return None
        if precomputed_credible_levels:
            levels = [(int(item["index"]), float(item.get("confidence_level", item.get("CL", 1.0))))
                     for item in clean]
            levels.sort(key=lambda pair: pair[1])
            peak_pixel = levels[0][0]
            n_pixels = sum(1 for _, level in levels if level <= credible_fraction) or 1
        else:
            weighted = [(int(item["index"]), float(item.get("probability", 0.0)))
                       for item in clean if item.get("probability") is not None]
            if not weighted:
                return None
            weighted.sort(key=lambda pair: -pair[1])
            total = sum(probability for _, probability in weighted) or 1.0
            peak_pixel = weighted[0][0]
            cumulative = 0.0
            n_pixels = len(weighted)
            for count, (_, probability) in enumerate(weighted, start=1):
                cumulative += probability / total
                if cumulative >= credible_fraction:
                    n_pixels = count
                    break

    lon, lat = healpix.healpix_to_lonlat(peak_pixel)
    area_deg2 = n_pixels * pixel_area_deg2
    radius_deg = math.sqrt(area_deg2 / math.pi)
    return {
        "ra_deg": float(lon.to(u.deg).value), "dec_deg": float(lat.to(u.deg).value),
        "radius_arcsec": radius_deg * 3600.0, "credible_fraction": float(credible_fraction),
        "n_pixels": n_pixels,
    }
