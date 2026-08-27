"""Swift-XRT point-source catalog (SXPS) metadata connector.

Metadata-only, matching the Chandra/SDSS/Pan-STARRS shape: this connector
discovers X-ray counterparts detected by Swift's XRT instrument and their
basic properties, and deliberately returns no light curves. Swift XRT does
publish per-source time series (it is a genuinely time-domain X-ray mission),
but extracting and normalising them is a separate, bounded job -- not
something this adapter builds implicitly.

`query_hardness_ratios`, added for roadmap item 23, is a new additive
function -- `cone_search` above is unchanged. It targets VizieR (CDS)
rather than this module's own `CONE_URL`, mirroring `chandra.py`'s
`query_band_fluxes`: 2SXPS is real and live on VizieR as `IX/58`
(Evans et al. 2020), confirmed this session with a real cone search
returning `CR0` (total 0.3-10 keV count rate) and 2SXPS's own pre-computed
`HR1`/`HR2` -- per that paper's own stated definitions,
`HR1 = (M-S)/(M+S)` and `HR2 = (H-M)/(H+M)` for Soft=0.3-1, Medium=1-2,
Hard=2-10 keV count rates, the same `(x-y)/(x+y)` convention
`xray_hardness.hardness_ratio` implements generically. 2SXPS additionally
publishes real per-source, multi-epoch light curves in four energy bands
plus these two hardness ratios (per that paper's own abstract) -- a real,
better-than-single-epoch data source for `xray_hardness_eval.py`'s state-
transition-detection validation, though extracting that per-epoch time
series (as opposed to this catalogue-level summary row) was not attempted
this session -- a real, stated [GAP].
"""

from __future__ import annotations

from .. import netclient
from ..tap import parse_votable
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

VIZIER_SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
VIZIER_CATALOG = "IX/58"

DEFAULT_RELEASE = "sxps2"
CONE_URL = "https://www.swift.ac.uk/xrt_products/sxpsCone"


def parse_rows(payload: object, limit: int = 100) -> list[dict]:
    """The cone endpoint returns a list of row dicts; keep only real rows."""
    if not isinstance(payload, list):
        return []
    return [row for row in payload[:limit] if isinstance(row, dict)]


class SwiftConnector(SurveyConnector):
    name = "Swift"
    capabilities = ("catalogue", "xray_metadata")
    resolution_arcsec = 2.5
    # New X-ray surveys launch opt-in, matching Chandra/SDSS/Pan-STARRS
    # precedent, until the provider contract (endpoint, auth, rate limits)
    # is validated against the real service.
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        response = netclient.get(
            CONE_URL,
            {"ra": query.ra_deg, "dec": query.dec_deg,
             "radius": query.radius_arcsec / 60.0, "catalog": self.release,
             "limit": top, "format": "json"},
            timeout=60, provider="swift",
        )
        try:
            rows = parse_rows(response.json(), top)
        except ValueError:
            rows = []
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row.get("IAUName") or row["source_id"])
                ra_deg = float(row.get("RA") if row.get("RA") is not None else row["ra_deg"])
                dec_deg = float(row.get("Decl") if row.get("Decl") is not None else row["dec_deg"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id,
                ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"obsid": row.get("ObsID"),
                       "num_obs": row.get("NumObs"),
                       "rate": row.get("Rate")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # SXPS per-source time-series extraction is a separate bounded job,
        # same rationale as chandra.py / sdss.py: this connector is discovery
        # and catalogue metadata only.
        return []


def query_hardness_ratios(ra_deg: float, dec_deg: float, radius_arcsec: float, limit: int = 100
                          ) -> list[dict]:
    """Real 2SXPS total count rate and pre-computed hardness ratios near a
    position, via VizieR `IX/58` -- see this module's docstring.
    """
    top = max(1, min(int(limit), 200))
    response = netclient.get(
        VIZIER_SCS_URL,
        {"-source": VIZIER_CATALOG, "RA": ra_deg, "DEC": dec_deg,
         "SR": radius_arcsec / 3600.0, "-out.max": top},
        timeout=60, provider="vizier",
    )
    rows = parse_votable(response.text, top)
    results: list[dict] = []
    for row in rows:
        try:
            object_id = str(row["IAUName"])
            ra_val, dec_val = float(row["RAJ2000"]), float(row["DEJ2000"])
        except (KeyError, TypeError, ValueError):
            continue
        results.append({
            "object_id": object_id, "ra_deg": ra_val, "dec_deg": dec_val,
            "count_rate": row.get("CR0"), "hr1": row.get("HR1"), "hr2": row.get("HR2"),
        })
    return results
