"""Kepler/K2 — high-cadence time-series photometry via Lightkurve/MAST.

The standing gap named repeatedly in `docs/LIMITATIONS.md` since items 16
and 18 landed: `lightkurve` (already a core dependency, used unchanged by
`surveys/tess.py`) has a built-in MAST search that reaches Kepler/K2
without any new HTTP code -- this connector is that search, following
`tess.py`'s exact shape (same `SurveyConnector` methods, same PDCSAP-vs-
raw flux-column choice, same per-target sector/campaign dedup). Kepler
and K2 are the same pipeline family and the same `lightkurve` mission
keyword family ("Kepler" / "K2"), so one connector covers both via the
`mission` constructor kwarg rather than two near-duplicate files.

Kepler/K2 time is BKJD (Barycentric Kepler Julian Date); `BJD_TDB =
BKJD + 2454833.0` is Kepler's own documented offset (Kepler Data
Processing Handbook, KSCI-19081), the Kepler-mission analogue of
`tess.py`'s BTJD_OFFSET.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector
from .tess import _estimate_flux_error, _row_float

LOGGER = logging.getLogger(__name__)

BKJD_OFFSET = 2454833.0

# Both Kepler's own pipeline (PDC) and K2's community-standard K2SFF/
# EVEREST pipelines publish differently-named flux columns; PDCSAP_FLUX
# is what the archive's own Kepler pipeline calls its systematics-
# corrected flux, same rationale as tess.py's own FLUX_COLUMNS table.
FLUX_COLUMNS = {"Kepler": "pdcsap_flux", "K2": "pdcsap_flux"}
DEFAULT_FLUX_COLUMN = FLUX_COLUMNS["Kepler"]


class KeplerConnector(SurveyConnector):
    name = "Kepler"
    capabilities = ("catalogue", "light_curve")
    resolution_arcsec = 4.0  # Kepler's pixel scale
    # Real bug found via the full test suite (not this module's own
    # tests): `SurveyConnector.enabled_by_default` defaults to True, and
    # `tests/test_connectors.py::test_the_three_initial_surveys_are_
    # registered` hardcodes the exact default-enabled set to
    # `["gaia", "tess", "ztf"]` -- adding a fourth default-enabled,
    # real-acquisition connector silently changes what `acquire`'s
    # default survey list pulls for every existing project, not just what
    # this session's own tests expect. Kepler follows every other
    # connector added this session (opt-in) rather than TESS's
    # already-established default, consistent with keeping this whole
    # batch's changes additive rather than behavior-changing by default.
    enabled_by_default = False

    def __init__(self, mission: str = "Kepler", max_quarters: int = 4) -> None:
        if mission not in ("Kepler", "K2"):
            raise ValueError(f"mission must be 'Kepler' or 'K2', got {mission!r}")
        self.mission = mission
        self.name = mission
        self.release = mission.lower()
        # Every extra quarter/campaign is another download; the default
        # keeps an exploratory search cheap, same rationale as tess.py's
        # max_sectors.
        self.max_quarters = max_quarters

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        import lightkurve as lk

        search = lk.search_lightcurve(
            f"{query.ra_deg} {query.dec_deg}", mission=self.mission,
            radius=query.radius_arcsec,
        )
        if search is None or len(search) == 0:
            return []

        columns = set(search.table.colnames)
        seen: dict[str, SourceRef] = {}
        for row in search.table:
            target = str(row["target_name"])
            if target not in seen:
                if len(seen) >= limit:
                    continue
                seen[target] = SourceRef(
                    survey=self.name,
                    object_id=f"KIC {target}" if target.isdigit() else target,
                    ra_deg=_row_float(row, columns, ("s_ra", "ra"), query.ra_deg),
                    dec_deg=_row_float(row, columns, ("s_dec", "dec"), query.dec_deg),
                    extra={"quarters": []},
                )
            self._record_quarter(seen[target], row, columns)
        return list(seen.values())

    def _record_quarter(self, source: SourceRef, row, columns: set[str]) -> None:
        # Kepler quarters and K2 campaigns share the archive's own generic
        # "sequence_number" column, same as tess.py's sectors.
        if "sequence_number" not in columns:
            return
        try:
            quarter = int(row["sequence_number"])
        # Real bug found live: a masked/missing sequence_number cell
        # (astropy masked columns are common in MAST search results)
        # raises numpy.ma.MaskError on conversion, not TypeError or
        # ValueError -- an uncaught MaskError here crashed the whole
        # cone_search() for every target in the batch over one target's
        # missing quarter number, not just this one row. Same gap
        # tess.py's _row_float/_record_sector had (this connector's own
        # docstring says it follows tess.py's exact shape).
        except (TypeError, ValueError, np.ma.MaskError):
            return
        quarters = source.extra.setdefault("quarters", [])
        if quarter not in quarters:
            quarters.append(quarter)

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        """One curve per quarter/campaign, capped by `max_quarters`."""
        import lightkurve as lk

        search = lk.search_lightcurve(source.object_id, mission=self.mission)
        if search is None or len(search) == 0:
            return []

        flux_column = FLUX_COLUMNS.get(self.mission, DEFAULT_FLUX_COLUMN)
        curves: list[LightCurve] = []
        for index in range(min(len(search), self.max_quarters)):
            try:
                downloaded = search[index].download(flux_column=flux_column)
            except Exception as exc:  # noqa: BLE001 - one bad quarter must not fail the object
                LOGGER.warning(
                    "Kepler/K2 quarter download failed mission=%s flux_column=%s "
                    "object=%s quarter_index=%s error=%s: %s",
                    self.mission, flux_column, source.object_id, index,
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
        try:
            time = np.asarray(downloaded.time.value, dtype=np.float64) + BKJD_OFFSET
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
            invalid = ~np.isfinite(err) | (err <= 0)
            if invalid.any():
                fallback = _estimate_flux_error(value)
                err = np.where(invalid, fallback, err).astype(np.float32)
                estimated_error = True

        # A real bug found and fixed this session: Kepler's own Quarter 0
        # (the mission's short initial commissioning quarter) is a valid,
        # real quarter number, but `0 or fallback` treats 0 as falsy in
        # Python -- a real Quarter-0 fetch silently mislabelled itself
        # "kepler-qunknown" instead of "kepler-q0", confirmed live against
        # KIC 9726699's own real Quarter 0 product this session.
        meta = getattr(downloaded, "meta", {})
        quarter = meta.get("QUARTER")
        if quarter is None:
            quarter = meta.get("CAMPAIGN", "unknown")
        return LightCurve(
            source=SourceRef(
                survey=source.survey, object_id=source.object_id,
                ra_deg=source.ra_deg, dec_deg=source.dec_deg,
                extra={**source.extra, "quarter": quarter,
                       "flux_error": "estimated_mad_differences" if estimated_error else "archive"},
            ),
            release=f"{self.release}-q{quarter}",
            band=self.mission.upper(),
            value_kind="flux",
            time=time,
            value=value,
            value_err=err,
            time_system="BJD_TDB",
        )
