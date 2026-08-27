"""DESI redshift-catalog connector via NOIRLab Astro Data Lab (roadmap item 24).

Mirrors `des.py`'s DataLab TAP shape exactly (same `TAP_URL`, same anonymous
sync-query contract, same fixed/parameterised ADQL -- no user SQL reaches the
service). The live schema was checked directly while building this connector,
not assumed:

- `desi_dr1.zpix` / `desi_edr.zpix` are real, queryable tables (confirmed via
  a live `SELECT *` against `desi_dr1.zpix`, which returned real rows). They
  carry the DESI HEALPix-COADD redshift catalog: `targetid`, `healpix`,
  `survey`, `program`, `spectype`, `z`, `zerr`, `zwarn`, `deltachi2`,
  `desiname`.
- There is NO plain `ra`/`dec` column on `zpix` -- only `mean_fiber_ra`/
  `mean_fiber_dec` (confirmed live; a query for a bare `ra` column against
  this table would misbehave the same way `sdss.py`'s docstring already
  documents for a wrong column assumption elsewhere in this codebase). This
  connector's cone search therefore filters on `mean_fiber_ra`/
  `mean_fiber_dec`, not `ra`/`dec`.
- Per NOIRLab's own DESI-at-Data-Lab documentation (checked live while
  building this module): the HEALPix-coadd *spectra* themselves are NOT
  downloadable through Data Lab -- they are "searchable and retrievable from
  SPARCL" instead; per-exposure/per-night/per-tile spectra are file-only,
  reachable only via NERSC/Globus. This connector is therefore metadata-only,
  matching `des.py`'s own "co-add survey, catalogue metadata only" shape --
  real spectrum acquisition for DESI is a separate, SPARCL-based job, not
  something this adapter fabricates. `targetid`/`healpix`/`survey`/`program`
  are preserved in `SourceRef.extra` specifically so that future SPARCL job
  has everything it needs to look the spectrum up.

FIXED in a follow-up session (2026-08-25): `desi_dr1.zpix` is real and a bare
`SELECT * ... TOP N` (no WHERE clause) returns real rows quickly, but every
WHERE-filtered query tried against it via `https://datalab.noirlab.edu/
tap/sync` -- ADQL `CONTAINS(POINT(...), CIRCLE(...))` (des.py's own working
pattern, here failing with a Postgres `function point(unknown, double
precision, double precision) does not exist` error), a plain numeric
`BETWEEN` range filter on `mean_fiber_ra`/`mean_fiber_dec`, and even
`WHERE targetid = <a real value read back from an unfiltered row>` -- either
errored or exceeded a 60-90s synchronous timeout.

Live testing in the follow-up session isolated TWO separate, unrelated
problems previously conflated under one "sync times out" finding:
(1) `WHERE targetid = <value>` and a plain `BETWEEN` box both time out via
`/tap/sync` but complete in under a second (the exact-match case) to ~20s
(the box case) via `/tap/async` (`tap.async_query`, added this session) --
the sync endpoint's connection-timeout window, not query execution speed,
was the actual blocker there. (2) `CONTAINS(POINT(...), CIRCLE(...))`
fails with the IDENTICAL `PSQLException: function point(...) does not
exist` via BOTH sync and async -- a real, permanent Postgres schema gap on
this specific table (`des_dr2.main` supports the identical ADQL shape
fine), NOT a timeout, and async cannot fix it. `cone_search` below
therefore uses `tap.async_query` (not `netclient.get`) with a bounding-box
`BETWEEN` prefilter on `mean_fiber_ra`/`mean_fiber_dec` instead of
`CONTAINS`/`CIRCLE`, then applies an EXACT great-circle-distance filter
(`crossmatch.angular_separation_arcsec`, reused unchanged) to the returned
rows before building `SourceRef`s -- so the box's corner-overshoot never
leaks an out-of-radius source into the result. RA-range wraparound across
0/360 degrees, and the near-pole case where a single `cos(dec)`-scaled RA
half-width would blow up, are both handled explicitly (see `_bounding_box`),
not silently mishandled.

A SPARCL pivot was attempted in a follow-up session (2026-08-25) and reverted:
`pip install sparclclient` (v1.3.0) force-downgrades `numpy` to `1.26.4` via
its `specutils`/`asdf-astropy`/`gwcs` dependency chain, which is a HARD,
real conflict with this project's pinned `numpy==2.5.2` (`engine/
requirements.lock`) -- confirmed live: `astropy_healpix` (this codebase's
own `gw.py`/`kilonova_eval.py` dependency) failed to import outright after
the install with a binary C-API mismatch (`numpy._core.multiarray failed to
import`), and `photutils`/`cupy` reported the identical incompatibility.
This is not a version-pinning nuisance to route around casually: `pip`'s own
resolver flagged the conflict and proceeded anyway (no hard failure at
install time), so the breakage is silent until something that needs the
newer numpy is actually imported -- exactly what happened here. The
installation was fully reverted this session (`sparclclient` and its forced
dependency chain uninstalled, `numpy==2.5.2`/`scipy==1.18.0`/`astropy==8.0.1`
reinstalled from the pinned versions, `pandas` restored to `2.3.3`) and
verified via the full test suite (1776 passed, 23 skipped, 0 failed --
identical to the pre-attempt count). SPARCL is therefore NOT a viable path
for this codebase as it stands, without either (a) `sparclclient` releasing
a numpy-2-compatible version, or (b) installing it into a separate,
isolated environment from this project's own dependencies -- neither
attempted here. DESI real spectrum acquisition remains an open [GAP] --
catalog/redshift access (this connector's `cone_search`) is fixed via
`/tap/async` (see above), which stays within this project's existing
dependency set, but Data Lab still does not serve DESI spectra at all, and
SPARCL remains the only documented route to them.
"""

from __future__ import annotations

import math

from .. import tap
from ..crossmatch import angular_separation_arcsec
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

DEFAULT_RELEASE = "dr1"
TAP_SYNC_URL = "https://datalab.noirlab.edu/tap/sync"
PAGE_SIZE = 200
MAX_PAGES = 20
# Above this |declination|, a cos(dec)-scaled RA half-width blows up
# (cos(dec) -> 0); every RA value is within any reasonable search radius
# of a pole anyway, so the bounding box falls back to the full RA range.
POLE_GUARD_DEC_DEG = 89.0


def _bounding_box(ra_deg: float, dec_deg: float, radius_deg: float) -> dict:
    """A rectangular RA/Dec box covering a cone, wide enough that the
    circle is fully inside it -- `cone_search` filters the box's results
    down to the exact circle afterward (`angular_separation_arcsec`), so
    this only needs to never be too NARROW, unlike a final answer.

    Two real edge cases handled explicitly rather than left to silently
    misbehave: RA wraparound across the 0/360 degree boundary (returns
    two RA ranges instead of one), and a box that reaches within
    `POLE_GUARD_DEC_DEG` of a pole (falls back to the full RA range,
    since `1/cos(dec)` diverges there).
    """
    dec_lo = max(-90.0, dec_deg - radius_deg)
    dec_hi = min(90.0, dec_deg + radius_deg)
    if dec_hi >= POLE_GUARD_DEC_DEG or dec_lo <= -POLE_GUARD_DEC_DEG:
        return {"dec_lo": dec_lo, "dec_hi": dec_hi, "ra_ranges": [(0.0, 360.0)]}

    cos_dec = math.cos(math.radians(max(abs(dec_lo), abs(dec_hi))))
    ra_half_width = min(radius_deg / max(cos_dec, 1e-6), 180.0)
    lo, hi = ra_deg - ra_half_width, ra_deg + ra_half_width
    if lo < 0.0:
        ra_ranges = [(0.0, hi), (360.0 + lo, 360.0)]
    elif hi > 360.0:
        ra_ranges = [(lo, 360.0), (0.0, hi - 360.0)]
    else:
        ra_ranges = [(lo, hi)]
    return {"dec_lo": dec_lo, "dec_hi": dec_hi, "ra_ranges": ra_ranges}


class DESIConnector(SurveyConnector):
    name = "DESI"
    capabilities = ("catalogue", "spectroscopic_redshift")
    resolution_arcsec = 1.0
    # New TAP-backed catalogues launch opt-in, matching des.py/sdss.py
    # precedent, until broader use validates the provider contract.
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    def build_query(self, query: ConeQuery, top: int, after_id: str | None = None) -> str:
        """Fixed ADQL for one bounding-box page against `desi_{release}.zpix`,
        keyset-paginated by `targetid` -- same no-OFFSET-in-ADQL pagination
        `des.py` already uses, since `targetid` (unlike DES's
        `coadd_object_id`) is not guaranteed positive, so pagination
        compares as a plain `>`. A bounding box, not `CONTAINS`/`CIRCLE`
        -- see module docstring for why.
        """
        box = _bounding_box(query.ra_deg, query.dec_deg, query.radius_deg)
        ra_clause = " OR ".join(
            f"mean_fiber_ra BETWEEN {lo:.8f} AND {hi:.8f}" for lo, hi in box["ra_ranges"])
        clause = (f"WHERE ({ra_clause}) "
                  f"AND mean_fiber_dec BETWEEN {box['dec_lo']:.8f} AND {box['dec_hi']:.8f}")
        if after_id is not None:
            clause += f" AND targetid > {after_id}"
        return (
            f"SELECT TOP {top} targetid, mean_fiber_ra, mean_fiber_dec, "
            "healpix, survey, program, spectype, z, zerr, zwarn, deltachi2, desiname "
            f"FROM desi_{self.release}.zpix "
            f"{clause} ORDER BY targetid ASC"
        )

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        remaining = max(1, min(int(limit), 2000))
        sources: list[SourceRef] = []
        after_id: str | None = None
        for _ in range(MAX_PAGES):
            page_size = max(1, min(remaining, PAGE_SIZE))
            result = tap.async_query(
                TAP_SYNC_URL, self.build_query(query, page_size, after_id),
                release=self.release, max_rows=page_size, fmt="csv", provider="datalab")
            rows = result["rows"]
            if not rows:
                break
            for row in rows:
                try:
                    object_id = str(row["targetid"])
                    ra_deg = float(row["mean_fiber_ra"])
                    dec_deg = float(row["mean_fiber_dec"])
                except (KeyError, TypeError, ValueError):
                    continue
                # The bounding box is deliberately wider than the requested
                # cone (see `_bounding_box`); this is where it narrows back
                # down to an exact circle.
                if angular_separation_arcsec(query.ra_deg, query.dec_deg,
                                             ra_deg, dec_deg) > query.radius_arcsec:
                    continue
                try:
                    z = float(row["z"])
                except (KeyError, TypeError, ValueError):
                    z = None
                try:
                    z_err = float(row["zerr"])
                except (KeyError, TypeError, ValueError):
                    z_err = None
                sources.append(SourceRef(
                    survey=self.name, object_id=object_id,
                    ra_deg=ra_deg, dec_deg=dec_deg,
                    extra={"healpix": row.get("healpix"), "survey": row.get("survey"),
                           "program": row.get("program"), "spectype": row.get("spectype"),
                           "z": z, "z_err": z_err, "zwarn": row.get("zwarn"),
                           "deltachi2": row.get("deltachi2"), "desiname": row.get("desiname")},
                ))
            after_id = str(rows[-1]["targetid"])
            remaining -= len(rows)
            if len(rows) < page_size or remaining <= 0:
                break
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # DESI is not a time-domain survey; HEALPix-coadd spectra are
        # retrievable only via SPARCL (see module docstring), not a light
        # curve this adapter could fabricate.
        return []
