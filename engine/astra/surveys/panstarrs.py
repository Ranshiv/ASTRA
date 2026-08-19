"""Pan-STARRS DR2 mean-photometry metadata connector via MAST."""

from __future__ import annotations

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

DEFAULT_RELEASE = "dr2"
MEAN_URL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/{release}/mean"


def parse_rows(payload: object, limit: int = 100) -> list[dict]:
    if not isinstance(payload, list):
        return []
    return [row for row in payload[:limit] if isinstance(row, dict)]


class PanSTARRSConnector(SurveyConnector):
    name = "Pan-STARRS"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 1.0
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        response = netclient.get(
            MEAN_URL.format(release=self.release),
            {"ra": query.ra_deg, "dec": query.dec_deg,
             "radius": query.radius_deg, "pagesize": top, "format": "json"},
            timeout=60, provider="mast",
        )
        try:
            rows = parse_rows(response.json(), top)
        except ValueError:
            rows = []
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row.get("objID") or row["objid"])
                ra_deg = float(row.get("raMean", row.get("ra")))
                dec_deg = float(row.get("decMean", row.get("dec")))
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id,
                ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"g_mean": row.get("gMeanPSFMag"),
                       "r_mean": row.get("rMeanPSFMag"),
                       "i_mean": row.get("iMeanPSFMag"),
                       "z_mean": row.get("zMeanPSFMag"),
                       "y_mean": row.get("yMeanPSFMag")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        return []
