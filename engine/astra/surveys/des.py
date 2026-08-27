"""Dark Energy Survey DR2 catalogue connector via NOIRLab Astro Data Lab.

DES is a co-add survey, not a time-domain one: the public DR2 release exposes
stacked photometry per object rather than per-epoch measurements.  The
connector therefore surfaces catalogue metadata only, so a DES counterpart can
be discovered and its identifier preserved for a later bounded product job.

Access is through Astro Data Lab's anonymous TAP endpoint.  The ADQL text is
fixed and parameterised by ASTRA -- callers cannot supply their own query.

A real, live-confirmed service regression and its fix, found this session:
the standard ADQL geometry cone search (`CONTAINS(POINT('ICRS',...),
CIRCLE('ICRS',...))=1`) this file used to emit now fails server-side against
the real, current NOIRLab Data Lab TAP backend with `PSQLException: ERROR:
function point(unknown, double precision, double precision) does not exist`
-- confirmed to be a SERVICE-WIDE issue, not specific to `des_dr2.main`: the
identical error reproduces against a completely different table
(`ls_dr9.tractor`) with the same query shape, and `LANG=SQL`/`Postgres`/
`postgresql` are all rejected (`IllegalArgumentException: unknown LANG: ...`)
-- only `LANG=ADQL` is accepted, and its own geometry-function translation
layer is currently missing whatever Postgres extension (e.g. pg_sphere)
would define `point()`/`circle()`. This is not fixable by changing query
SYNTAX within ADQL; the real, tested-live workaround is a plain rectangular
bounding-box filter (`ra`/`dec` `BETWEEN` comparisons, no geometry functions
at all) confirmed live this session to return real DES DR2 rows in ~4
seconds for a small search box. A bounding box is a SQUARE region, not a
circle, so `cone_search` now post-filters to the real requested radius using
`crossmatch.angular_separation_arcsec` (reused unchanged) rather than
returning the box's corners as false positives.

A second, related bug found and fixed this session: a TAP error response is
itself a VOTable/XML document, not CSV -- `parse_csv`'s `csv.DictReader`
silently misparsed that XML as a single-column CSV, so the resulting "row"
had no `coadd_object_id` key and `cone_search` crashed with a bare
`KeyError` deep in its pagination loop instead of surfacing the real error.
`_is_error_response` now detects a TAP error body before parsing and
`cone_search` raises a clear `DESQueryError` instead.

A galaxy angular-size proxy was added for roadmap item 31
(`host_association.py`), which needs one to normalize a transient's offset
from a candidate host. The first attempt (a bare `flux_radius` column,
assumed in arcsec from documentation) was WRONG on both counts, confirmed
by a live `TAP_SCHEMA.columns` query this session:
`des_dr2.main` has no bare `flux_radius` column at all -- only per-band
`flux_radius_{g,i,r,y,z}` and `kron_radius`/`kron_radius_{g,i,r,y,z}` --
and the real column's `unit` field, also read live, is `pixel`, not arcsec
(confirmed sane by a live sample query against RA=45.0/Dec=-40.0: values
of ~1.5-4.2, consistent with a few-pixel half-light radius, not an arcsec
one). Fixed to select `flux_radius_r` (r-band, matching this module's own
r-band magnitude convention and `host_association.py`'s r-band Schechter
parameters) and convert to arcsec via `DES_PIXEL_SCALE_ARCSEC` (0.263
arcsec/pixel, the standard, widely published DECam pixel scale -- e.g.
Abbott et al. 2018, ApJS 239, 18, DES DR1). `cone_search` degrades a
missing/malformed value to `None` rather than crashing or fabricating a
size, so any future column-name drift fails soft (falling back to
`host_association.py`'s `DEFAULT_R_E_ARCSEC`), not hard.
"""

from __future__ import annotations

import csv
import io

from .. import netclient
from ..crossmatch import angular_separation_arcsec
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

DEFAULT_RELEASE = "dr2"
TAP_URL = "https://datalab.noirlab.edu/tap/sync"
PAGE_SIZE = 200
MAX_PAGES = 20
# DECam's native pixel scale, confirmed against des_dr2.main's own
# TAP_SCHEMA.columns.unit this session ("pixel", not arcsec) -- see this
# module's own docstring. Widely published (e.g. Abbott et al. 2018, ApJS
# 239, 18, DES DR1), not this codebase's own measurement.
DES_PIXEL_SCALE_ARCSEC = 0.263


class DESQueryError(RuntimeError):
    """The Data Lab TAP service returned an error instead of results."""


def _is_error_response(payload: str) -> bool:
    """A TAP error is delivered as a VOTable with `QUERY_STATUS="ERROR"`,
    not as CSV -- detected here before `parse_csv` ever sees it, so an
    error body degrades to a clear exception instead of silently
    misparsed rows missing every expected column."""
    return "QUERY_STATUS" in payload and 'value="ERROR"' in payload


def parse_csv(payload: str, limit: int = 100) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload))
    return [dict(row) for row in list(reader)[:limit]]


class DESConnector(SurveyConnector):
    name = "DES"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 0.9
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    def build_query(self, query: ConeQuery, top: int, after_id: str | None = None) -> str:
        """Fixed ADQL for one cone, keyset-paginated by coadd_object_id.

        No user SQL reaches the service. Standard ADQL has no portable
        OFFSET, so pages beyond the first are requested with
        `coadd_object_id > after_id` instead, ordered ascending so every
        page picks up exactly where the previous one stopped.

        Uses a plain rectangular `ra`/`dec` bounding box rather than
        `CONTAINS(POINT(...), CIRCLE(...))` -- see this module's own
        docstring for why the latter currently fails service-wide.
        `cone_search` post-filters the box down to the true circular
        radius, so this box is deliberately allowed to be a superset.
        """
        import math

        cos_dec = math.cos(math.radians(query.dec_deg))
        # A pole-adjacent cone (|dec| near 90) would need RA wraparound
        # handling this simple box does not attempt; DES DR2's own real
        # sky coverage (|dec| below about -15) never approaches a pole,
        # so this is a real, not merely theoretical, non-issue for this
        # connector -- stated rather than silently assumed.
        ra_half_width = min(query.radius_deg / max(cos_dec, 1e-6), 180.0)
        clause = (f"WHERE dec BETWEEN {query.dec_deg - query.radius_deg:.8f} "
                 f"AND {query.dec_deg + query.radius_deg:.8f} "
                 f"AND ra BETWEEN {query.ra_deg - ra_half_width:.8f} "
                 f"AND {query.ra_deg + ra_half_width:.8f}")
        if after_id is not None:
            clause += f" AND coadd_object_id > {after_id}"
        return (
            f"SELECT TOP {top} coadd_object_id, ra, dec, "
            "mag_auto_g, mag_auto_r, mag_auto_i, mag_auto_z, flux_radius_r "
            f"FROM des_{self.release}.main "
            f"{clause} ORDER BY coadd_object_id ASC"
        )

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        remaining = max(1, min(int(limit), 2000))
        sources: list[SourceRef] = []
        after_id: str | None = None
        for _ in range(MAX_PAGES):
            page_size = max(1, min(remaining, PAGE_SIZE))
            response = netclient.get(
                TAP_URL,
                {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                 "MAXREC": page_size,
                 "QUERY": self.build_query(query, page_size, after_id)},
                timeout=60, provider="datalab",
            )
            if _is_error_response(response.text):
                raise DESQueryError(f"DES TAP query failed: {response.text[:500]}")
            rows = parse_csv(response.text, page_size)
            if not rows:
                break
            for row in rows:
                try:
                    object_id = str(row["coadd_object_id"])
                    ra_deg, dec_deg = float(row["ra"]), float(row["dec"])
                except (KeyError, TypeError, ValueError):
                    continue
                if angular_separation_arcsec(query.ra_deg, query.dec_deg,
                                            ra_deg, dec_deg) > query.radius_arcsec:
                    continue
                try:
                    flux_radius_r_arcsec = float(row["flux_radius_r"]) * DES_PIXEL_SCALE_ARCSEC
                except (KeyError, TypeError, ValueError):
                    flux_radius_r_arcsec = None
                sources.append(SourceRef(
                    survey=self.name, object_id=object_id,
                    ra_deg=ra_deg, dec_deg=dec_deg,
                    extra={"g_mean": row.get("mag_auto_g"),
                           "r_mean": row.get("mag_auto_r"),
                           "i_mean": row.get("mag_auto_i"),
                           "z_mean": row.get("mag_auto_z"),
                           # Galaxy half-light-radius proxy for
                           # host_association.py, already converted from
                           # the real live-confirmed pixel unit to arcsec
                           # -- see this module's own docstring.
                           "flux_radius_r_arcsec": flux_radius_r_arcsec},
                ))
            after_id = str(rows[-1]["coadd_object_id"])
            remaining -= len(rows)
            if len(rows) < page_size or remaining <= 0:
                break
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # DR2 publishes co-added photometry only.  Single-epoch extraction is a
        # separate bounded job, not something this adapter fabricates.
        return []
