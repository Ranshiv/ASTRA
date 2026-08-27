"""2MASS near-IR catalogue connector via VizieR's Simple Cone Search
(roadmap item 26, panchromatic energy-balance SED).

VizieR catalogue `II/246/out` ("2MASS All-Sky Catalog of Point Sources",
Cutri+ 2003) confirmed live this session, same `vizier.cds.unistra.fr` SCS
endpoint `vlass.py`/`galex.py` already use: a real cone search around
RA=180.0, Dec=0.0 returned real `2MASS`/`RAJ2000`/`DEJ2000`/`Jmag`/
`e_Jmag` rows (and the analogous H/K columns). Static point-source
catalogue, so `fetch_light_curves` is the same metadata-only shape as
`panstarrs.py`.
"""

from __future__ import annotations

from .. import netclient
from ..tap import parse_votable
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
DEFAULT_CATALOG = "II/246/out"


class TwoMASSConnector(SurveyConnector):
    name = "2MASS"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 2.0
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
                object_id = str(row["2MASS"])
                ra_deg = float(row["RAJ2000"])
                dec_deg = float(row["DEJ2000"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"j_mag": row.get("Jmag"), "j_mag_error": row.get("e_Jmag"),
                       "h_mag": row.get("Hmag"), "h_mag_error": row.get("e_Hmag"),
                       "k_mag": row.get("Kmag"), "k_mag_error": row.get("e_Kmag")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        return []
