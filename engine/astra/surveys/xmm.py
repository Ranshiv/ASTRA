"""XMM-Newton Serendipitous Source Catalog (4XMM) metadata connector.

Metadata-only, matching the Chandra/Swift/SDSS/Pan-STARRS shape: this
connector discovers X-ray counterparts from the 4XMM catalogue and their
basic properties, and deliberately returns no light curves. XMM's own
per-source time-series products exist but extracting and normalising them
is a separate, bounded job -- not something this adapter builds implicitly.
"""

from __future__ import annotations

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

DEFAULT_RELEASE = "4xmm-dr13"
CONE_URL = "https://xmmssc-www.star.le.ac.uk/newpages/xsa_cone"


def parse_rows(payload: object, limit: int = 100) -> list[dict]:
    """The cone endpoint returns a list of row dicts; keep only real rows."""
    if not isinstance(payload, list):
        return []
    return [row for row in payload[:limit] if isinstance(row, dict)]


class XMMConnector(SurveyConnector):
    name = "XMM-Newton"
    capabilities = ("catalogue", "xray_metadata")
    resolution_arcsec = 6.0
    # New X-ray surveys launch opt-in, matching Chandra/Swift/SDSS/Pan-STARRS
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
            timeout=60, provider="xmm",
        )
        try:
            rows = parse_rows(response.json(), top)
        except ValueError:
            rows = []
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row.get("SRCID") or row["iauname"])
                ra_deg = float(row.get("SC_RA") if row.get("SC_RA") is not None else row["ra_deg"])
                dec_deg = float(row.get("SC_DEC") if row.get("SC_DEC") is not None else row["dec_deg"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id,
                ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"obs_id": row.get("OBS_ID"),
                       "ep_flux": row.get("SC_EP_8_FLUX"),
                       "n_detections": row.get("N_DETECTIONS")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # 4XMM per-source time-series extraction is a separate bounded job,
        # same rationale as chandra.py / swift.py: this connector is
        # discovery and catalogue metadata only.
        return []
