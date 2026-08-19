"""Chandra Source Catalog (CSC) X-ray metadata connector.

Metadata-only, matching the SDSS/Pan-STARRS shape: this connector discovers
X-ray counterparts and their basic properties, and deliberately returns no
light curves. CSC per-source time-series extraction (from event files) is a
separate, bounded job, not something this adapter builds implicitly.
"""

from __future__ import annotations

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

DEFAULT_RELEASE = "csc2.1"
CONE_URL = "https://cda.cfa.harvard.edu/csccli/browse"


def parse_rows(payload: object, limit: int = 100) -> list[dict]:
    """CSC's browse endpoint returns a list of row dicts; keep only real rows."""
    if not isinstance(payload, list):
        return []
    return [row for row in payload[:limit] if isinstance(row, dict)]


class ChandraConnector(SurveyConnector):
    name = "Chandra"
    capabilities = ("catalogue", "xray_metadata")
    resolution_arcsec = 0.5
    # New X-ray surveys launch opt-in, matching SDSS/Pan-STARRS precedent,
    # until the provider contract (endpoint, auth, rate limits) is validated.
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
            timeout=60, provider="chandra",
        )
        try:
            rows = parse_rows(response.json(), top)
        except ValueError:
            rows = []
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row.get("name") or row["src_id"])
                ra_deg = float(row.get("ra") if row.get("ra") is not None else row["ra_deg"])
                dec_deg = float(row.get("dec") if row.get("dec") is not None else row["dec_deg"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id,
                ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"obsid": row.get("obsid"),
                       "instrument": row.get("instrument"),
                       "flux_aper_b": row.get("flux_aper_b")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # CSC event-file/lightcurve extraction is a separate bounded job, same
        # rationale as sdss.py / panstarrs.py: this connector is discovery and
        # catalogue metadata only.
        return []
