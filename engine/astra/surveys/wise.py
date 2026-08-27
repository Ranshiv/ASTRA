"""AllWISE mid-IR catalogue connector via VizieR's Simple Cone Search
(roadmap item 26, panchromatic energy-balance SED).

VizieR catalogue `II/328/allwise` ("AllWISE Data Release", Cutri+ 2013)
confirmed live this session, same SCS pattern as `galex.py`/`twomass.py`:
a real cone search around RA=180.0, Dec=0.0 returned real
`AllWISE`/`RAJ2000`/`DEJ2000`/`W1mag`/`e_W1mag`/.../`W4mag`/`e_W4mag`
rows. Static point-source catalogue, metadata-only `fetch_light_curves`.
"""

from __future__ import annotations

from .. import netclient
from ..tap import parse_votable
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
DEFAULT_CATALOG = "II/328/allwise"


class WISEConnector(SurveyConnector):
    name = "WISE"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 6.0
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_CATALOG) -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        response = netclient.get(
            SCS_URL,
            {"-source": self.release, "RA": query.ra_deg, "DEC": query.dec_deg,
             "SR": query.radius_arcsec / 3600.0, "-out.max": top},
            timeout=60, provider="vizier",
        )
        rows = parse_votable(response.text, top)
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row["AllWISE"])
                ra_deg = float(row["RAJ2000"])
                dec_deg = float(row["DEJ2000"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"w1_mag": row.get("W1mag"), "w1_mag_error": row.get("e_W1mag"),
                       "w2_mag": row.get("W2mag"), "w2_mag_error": row.get("e_W2mag"),
                       "w3_mag": row.get("W3mag"), "w3_mag_error": row.get("e_W3mag"),
                       "w4_mag": row.get("W4mag"), "w4_mag_error": row.get("e_W4mag")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        return []
