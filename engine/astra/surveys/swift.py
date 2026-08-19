"""Swift-XRT point-source catalog (SXPS) metadata connector.

Metadata-only, matching the Chandra/SDSS/Pan-STARRS shape: this connector
discovers X-ray counterparts detected by Swift's XRT instrument and their
basic properties, and deliberately returns no light curves. Swift XRT does
publish per-source time series (it is a genuinely time-domain X-ray mission),
but extracting and normalising them is a separate, bounded job -- not
something this adapter builds implicitly.
"""

from __future__ import annotations

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

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
