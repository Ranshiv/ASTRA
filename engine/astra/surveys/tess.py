"""TESS — high-cadence time-series photometry via Lightkurve/MAST.

This connector deliberately requests SPOC light curves rather than target
pixel files. A TPF is 20-50 MB per target/sector against roughly 2 MB for the
corresponding light curve, and the pixels are only needed for the image
morphology features, which apply to a few hundred surviving candidates rather
than the whole search. That single choice is the difference between a Stage B
run fitting on this machine and overrunning it by an order of magnitude.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

LOGGER = logging.getLogger(__name__)

DEFAULT_RELEASE = "spoc"

# Lightkurve reports time as BTJD; BJD_TDB = BTJD + 2457000.
BTJD_OFFSET = 2457000.0

# PDCSAP has instrumental systematics removed and is the correct default for
# variability searches; SAP is kept available for artifact investigation,
# where the raw systematics are the thing under study. QLP and TGLC are
# different pipelines over TESS full-frame images rather than SPOC's own
# 2-minute cadence product, and neither publishes a "pdcsap_flux" column at
# all -- requesting it against their FITS tables raises a KeyError, which
# fetch_light_curves used to swallow as an empty result indistinguishable
# from "nothing here". This table is what fetch_light_curves keys on instead.
FLUX_COLUMNS = {
    "SPOC": "pdcsap_flux",
    "QLP": "sap_flux",
    # TGLC has no per-point flux_err column. _convert() estimates a robust
    # cadence-scale error and marks the provenance in SourceRef.extra.
    "TGLC": "cal_psf_flux",
}
DEFAULT_FLUX_COLUMN = FLUX_COLUMNS["SPOC"]


def _row_float(row, columns: set[str], candidates: tuple[str, ...],
               fallback: float) -> float:
    """Read a coordinate from the results table, falling back to the cone.

    Every TESS source used to inherit the cone centre, so all targets in one
    search shared a single position. Cross-matching then gave every TESS
    candidate an identical separation from any anchor and picked between them
    arbitrarily, which makes object-centric grouping meaningless. The archive
    supplies per-target coordinates; use them when present.
    """
    for name in candidates:
        if name not in columns:
            continue
        try:
            value = float(row[name])
        # A real bug found live: a masked/missing cell in the archive's
        # results table (astropy masked columns are common in MAST search
        # results) raises numpy.ma.MaskError on conversion, not TypeError
        # or ValueError -- an uncaught MaskError here crashed the whole
        # cone_search() for every target in the batch over one missing
        # coordinate cell, not just this one candidate column.
        except (TypeError, ValueError, np.ma.MaskError):
            continue
        if np.isfinite(value):
            return value
    return float(fallback)


def _record_sector(source: SourceRef, row, columns: set[str]) -> None:
    """Accumulate the sectors a target was observed in."""
    if "sequence_number" not in columns:
        return
    try:
        sector = int(row["sequence_number"])
    # Same masked-cell gap as _row_float above: a masked sequence_number
    # must be skipped, not crash the whole cone_search() batch.
    except (TypeError, ValueError, np.ma.MaskError):
        return
    sectors = source.extra.setdefault("sectors", [])
    if sector not in sectors:
        sectors.append(sector)


def _estimate_flux_error(value: np.ndarray) -> np.ndarray:
    """Estimate a finite per-cadence error for products without errors.

    TGLC publishes calibrated flux but explicitly omits ``FLUX_ERR``.  A
    robust MAD of first differences estimates cadence-scale noise while being
    insensitive to a small number of transits/flares.  The median-flux floor
    prevents a zero-error curve from entering the canonical store.
    """
    finite = value[np.isfinite(value)]
    if finite.size < 3:
        scale = float(np.nanstd(finite)) if finite.size else 1.0
    else:
        differences = np.diff(finite)
        centre = float(np.nanmedian(differences))
        scale = float(1.4826 * np.nanmedian(np.abs(differences - centre)) / np.sqrt(2.0))
    median = float(np.nanmedian(np.abs(finite))) if finite.size else 1.0
    floor = max(np.finfo(np.float32).eps * max(median, 1.0), 1e-8)
    return np.full(value.shape, max(scale, floor), dtype=np.float32)


class TESSConnector(SurveyConnector):
    name = "TESS"
    capabilities = ("catalogue", "light_curve", "target_pixel_file")
    resolution_arcsec = 21.0

    def __init__(self, release: str | None = None,
                 author: str = "SPOC", max_sectors: int = 4) -> None:
        # release defaults from author rather than a fixed constant. SPOC
        # stays "spoc" (identical to the old hardcoded default, so existing
        # stored data and tests are untouched), but an author change now
        # carries its release along automatically. This matters because
        # store.curve_path() keys purely on (survey, release, object_id,
        # band): leaving release at a stale default would make a QLP fetch
        # silently overwrite a SPOC parquet for the same TIC/sector, and
        # acquire.py's resumable-fetch cursor keys on release too, so it
        # would treat "fetched under SPOC" as "already fetched" and skip a
        # QLP attempt entirely under the default skip_existing=True.
        self.author = author
        self.release = release or self.author.lower()
        # Every extra sector is another download and another few MB stored;
        # the default keeps an exploratory search cheap.
        self.max_sectors = max_sectors

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        import lightkurve as lk

        search = lk.search_lightcurve(
            f"{query.ra_deg} {query.dec_deg}",
            mission="TESS",
            author=self.author,
            radius=query.radius_arcsec,
        )
        if search is None or len(search) == 0:
            return []

        # Dedupe by target BEFORE applying the limit. lk.search_lightcurve
        # returns one row per target x sector, so slicing the table first
        # spends the whole budget on one well-observed star: a 20-sector
        # target consumed 20 of a 25-row limit and yielded a single source.
        # That is why only 3 TESS targets existed against 404 ZTF.
        columns = set(search.table.colnames)
        seen: dict[str, SourceRef] = {}

        for row in search.table:
            target = str(row["target_name"])
            if target not in seen:
                if len(seen) >= limit:
                    continue
                seen[target] = SourceRef(
                    survey=self.name,
                    object_id=f"TIC {target}" if target.isdigit() else target,
                    ra_deg=_row_float(row, columns, ("s_ra", "ra"),
                                      query.ra_deg),
                    dec_deg=_row_float(row, columns, ("s_dec", "dec"),
                                       query.dec_deg),
                    extra={"sectors": []},
                )
            _record_sector(seen[target], row, columns)

        return list(seen.values())

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        """One curve per sector, capped by `max_sectors`."""
        import lightkurve as lk

        search = lk.search_lightcurve(
            source.object_id, mission="TESS", author=self.author
        )
        if search is None or len(search) == 0:
            return []

        flux_column = FLUX_COLUMNS.get(self.author, DEFAULT_FLUX_COLUMN)

        curves: list[LightCurve] = []
        for index in range(min(len(search), self.max_sectors)):
            try:
                downloaded = search[index].download(flux_column=flux_column)
            except Exception as exc:  # noqa: BLE001 - one bad sector must not fail the object
                # A per-sector download failure (network, missing file) is
                # expected and handled by skipping the sector. A flux-column
                # KeyError from a wrong author/column pairing is a different,
                # silent kind of failure -- log it so a future mismatch is
                # visible instead of looking identical to "no data here".
                LOGGER.warning(
                    "TESS sector download failed author=%s flux_column=%s "
                    "object=%s sector_index=%s error=%s: %s",
                    self.author, flux_column, source.object_id, index,
                    type(exc).__name__, exc,
                )
                continue
            if downloaded is None:
                continue
            curve = self._convert(downloaded, source)
            if curve is not None and len(curve):
                curves.append(curve)
        return curves

    def _convert(self, downloaded, source: SourceRef) -> LightCurve | None:
        """Extract the three columns worth keeping and drop the rest."""
        try:
            time = np.asarray(downloaded.time.value, dtype=np.float64) + BTJD_OFFSET
            value = np.asarray(downloaded.flux.value, dtype=np.float32)
        except AttributeError:
            return None

        estimated_error = False
        try:
            err = np.asarray(downloaded.flux_err.value, dtype=np.float32)
        except AttributeError:
            err = _estimate_flux_error(value)
            estimated_error = True
        if err.shape != value.shape or not np.isfinite(err).any():
            err = _estimate_flux_error(value)
            estimated_error = True
        else:
            # Preserve archive errors where valid, filling isolated missing
            # cadences with the robust curve-level estimate.
            invalid = ~np.isfinite(err) | (err <= 0)
            if invalid.any():
                fallback = _estimate_flux_error(value)
                err = np.where(invalid, fallback, err).astype(np.float32)
                estimated_error = True

        sector = getattr(downloaded, "meta", {}).get("SECTOR", "unknown")
        return LightCurve(
            source=SourceRef(
                survey=source.survey,
                object_id=source.object_id,
                ra_deg=source.ra_deg,
                dec_deg=source.dec_deg,
            extra={**source.extra, "sector": sector,
                   "flux_error": "estimated_mad_differences" if estimated_error else "archive"},
            ),
            release=f"{self.release}-s{sector}",
            band="TESS",
            value_kind="flux",
            time=time,
            value=value,
            value_err=err,
            time_system="BJD_TDB",
        )
