"""GALEX UV catalogue connector via VizieR's Simple Cone Search (roadmap
item 26, panchromatic energy-balance SED).

`sed.py` already covers Gaia/ZTF/TESS optical bands; the panchromatic SED
work needs UV and IR anchors it does not have. VizieR catalogue
`II/335/galex_ais` ("Revised catalog of GALEX UV sources (GUVcat_AIS
GR6+7)", Bianchi+ 2017) was confirmed live this session via the same
`vizier.cds.unistra.fr` Simple Cone Search endpoint `vlass.py` already
uses: a real cone search around RA=180.0, Dec=0.0 returned real
`Name`/`RAJ2000`/`DEJ2000`/`FUV`/`e_FUV`/`NUV`/`e_NUV` rows. This is a
static-catalogue survey, not a time-domain one, so `fetch_light_curves`
follows the same metadata-only shape as `panstarrs.py`/`des.py`.
"""

from __future__ import annotations

from .. import netclient
from ..tap import parse_votable
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
DEFAULT_CATALOG = "II/335/galex_ais"


class GALEXConnector(SurveyConnector):
    name = "GALEX"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 5.0  # GALEX PSF FWHM is a few arcsec; conservative
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
                object_id = str(row["Name"])
                ra_deg = float(row["RAJ2000"])
                dec_deg = float(row["DEJ2000"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"fuv_mag": row.get("FUV"), "fuv_mag_error": row.get("e_FUV"),
                       "nuv_mag": row.get("NUV"), "nuv_mag_error": row.get("e_NUV")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # GUVcat is a co-added source list, not per-visit photometry -- same
        # metadata-only shape as panstarrs.py/des.py.
        return []
