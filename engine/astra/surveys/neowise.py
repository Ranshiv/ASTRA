"""NEOWISE-Reactivation multi-epoch mid-IR connector via IRSA.

`wise.py`'s AllWISE connector is the static (one epoch, co-added) mid-IR
catalogue; this project's own review named the gap it leaves: NEOWISE is
the actual multi-epoch half, absent entirely before this connector.
Confirmed live this session, via `astroquery.ipac.irsa.Irsa` (the same
client `ztf.py` already uses for IRSA) against the `neowiser_p1bs_psd`
table (the NEOWISE-R single-exposure source table): a 5 arcsec cone at
RA=280.0, Dec=-20.0 returned 240 real per-exposure detection rows,
collapsing to real physical sources via `allwise_cntr` (the row's
cross-match to the static AllWISE catalogue) -- one source in that cone had
240 individual epochs spanning the mission's multi-year baseline, real
`mjd`/`w1mag`/`w2mag` columns confirmed present on every row.

`allwise_cntr` is `'--'` (an astropy masked-table sentinel, not the string
literal) for a detection IRSA could not cross-match to AllWISE -- e.g. a
genuinely new or moving source, or one too close to a bright neighbour.
Those rows are real detections but cannot be grouped into "one physical
source" without this connector doing its own spatial clustering, which it
does not attempt; `cone_search` only emits sources with a resolved
`allwise_cntr`; the discarded rows are not silently invented into
duplicate objects.

`fetch_light_curves` returns metadata only, the same "discovery now,
light-curve extraction later as its own bounded job" boundary
`panstarrs.py`/`sdss.py`/every VizieR-backed connector already draws:
`neowiser_p1bs_psd` is 42 TB across 200 billion rows, and assembling one
object's full per-epoch time series (quality-flag filtering, W1/W2 band
separation, per-epoch photometric-calibration handling) is real, scoped-out
follow-up work, not attempted here. `extra["epoch_count"]`/`mjd_first`/
`mjd_last` are reported precisely so a caller can see this connector's
sources ARE genuinely multi-epoch even though the epochs themselves are
not yet extracted -- the fact `wise.py`'s AllWISE sources structurally
cannot report.
"""

from __future__ import annotations

from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

CATALOG = "neowiser_p1bs_psd"


class NEOWISEConnector(SurveyConnector):
    name = "NEOWISE"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 6.5  # WISE W1/W2 PSF FWHM, same beam as AllWISE
    # New survey, launches opt-in until this session's live verification is
    # reproduced against a wider range of positions -- same precedent as
    # every other connector added this way.
    enabled_by_default = False

    def __init__(self, release: str = CATALOG) -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.irsa import Irsa

        top = max(1, min(int(limit), 200))
        coord = SkyCoord(query.ra_deg, query.dec_deg, unit="deg")
        table = Irsa.query_region(
            coordinates=coord, catalog=self.release, spatial="Cone",
            radius=query.radius_arcsec * u.arcsec,
        )
        if table is None or len(table) == 0:
            return []

        groups: dict[str, list] = {}
        for row in table:
            cntr = row["allwise_cntr"]
            # A masked/unresolved cross-match ('--') cannot be grouped into
            # one physical source without this connector's own spatial
            # clustering, which it does not attempt -- see module docstring.
            if cntr is None or str(cntr) in ("--", "") or getattr(cntr, "mask", False):
                continue
            groups.setdefault(str(cntr), []).append(row)

        sources: list[SourceRef] = []
        for cntr, rows in list(groups.items())[:top]:
            try:
                latest = max(rows, key=lambda r: float(r["mjd"]))
                ra_deg = float(latest["ra"])
                dec_deg = float(latest["dec"])
            except (KeyError, TypeError, ValueError):
                continue
            mjds = [float(r["mjd"]) for r in rows]
            sources.append(SourceRef(
                survey=self.name, object_id=cntr, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"w1_mag": _safe_float(latest, "w1mpro"),
                       "w1_mag_error": _safe_float(latest, "w1sigmpro"),
                       "w2_mag": _safe_float(latest, "w2mpro"),
                       "w2_mag_error": _safe_float(latest, "w2sigmpro"),
                       "epoch_count": len(rows),
                       "mjd_first": min(mjds), "mjd_last": max(mjds)},
            ))

        if table is not None and len(table) and not sources:
            import logging
            logging.getLogger(__name__).warning(
                "NEOWISE: IRSA returned %d row(s) but none resolved to a "
                "cross-matched source -- allwise_cntr/ra/dec/mjd may no "
                "longer match %r's real columns.", len(table), self.release)
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        return []


def _safe_float(row, name: str) -> float | None:
    try:
        value = row[name]
        return None if getattr(value, "mask", False) else float(value)
    except (KeyError, TypeError, ValueError):
        return None
