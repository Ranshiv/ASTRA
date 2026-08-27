"""Herschel/PACS far-IR catalogue connector via VizieR's Simple Cone Search
(roadmap item 26, panchromatic energy-balance SED).

VizieR catalogue `VIII/106` ("Herschel/PACS Point Source Catalogs",
Herschel team 2017) confirmed live this session, same SCS pattern as
`galex.py`/`twomass.py`/`wise.py`: a real cone search around RA=180.0,
Dec=0.0 returned real `Name`/`Band`/`RAJ2000`/`DEJ2000`/`Flux`/`snr` rows
(flux in mJy, `snr` the signal-to-noise ratio -- there is no separate
tabulated error column, so `flux_error_mjy` is derived as `Flux / snr`
when `snr` is positive). PACS observed at 70/100/160um in different
pointed programmes rather than one uniform all-sky pass, so a given cone
may return zero, one, or several band rows for the same physical source;
`extra["band"]` carries which wavelength each row is. Pointed/targeted
coverage (not all-sky), metadata-only `fetch_light_curves`.
"""

from __future__ import annotations

from .. import netclient
from ..tap import parse_votable
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
DEFAULT_CATALOG = "VIII/106"


def _flux_error_mjy(flux: object, snr: object) -> float | None:
    try:
        flux_value, snr_value = float(flux), float(snr)
    except (TypeError, ValueError):
        return None
    if snr_value <= 0:
        return None
    return abs(flux_value) / snr_value


class HerschelConnector(SurveyConnector):
    name = "Herschel"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 6.0  # PACS 70/100um beam; coarser than WISE, finer than SPIRE
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
                extra={"band": row.get("Band"), "flux_mjy": row.get("Flux"),
                       "flux_error_mjy": _flux_error_mjy(row.get("Flux"), row.get("snr")),
                       "snr": row.get("snr")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        return []
