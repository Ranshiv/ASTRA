"""SDSS optical catalogue/spectroscopy connector.

The connector deliberately exposes catalogue metadata only until a bounded
spectral product contract is selected.  It uses SkyServer's public SQL endpoint
for small cone queries and keeps the query text fixed/parameterized by ASTRA.
No raw spectra are downloaded by this adapter yet; callers can still discover
SDSS counterparts and preserve their identifiers for a later spectrum job.
"""

from __future__ import annotations

import csv
import io
from urllib.parse import quote

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

DEFAULT_RELEASE = "dr18"
SQL_URL = "https://skyserver.sdss.org/{release}/SkyServerWS/SearchTools/SqlSearch"


def parse_csv(payload: str, limit: int = 100) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload))
    return [dict(row) for row in list(reader)[:limit]]


class SDSSConnector(SurveyConnector):
    name = "SDSS"
    capabilities = ("catalogue", "spectrum_metadata")
    resolution_arcsec = 1.0
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        # The fixed query returns only identifiers/positions and does not allow
        # user-provided SQL, which keeps the connector injection-safe.
        sql = (
            f"SELECT TOP {top} objID, ra, dec, plate, mjd, fiberID "
            "FROM PhotoObj "
            f"WHERE dbo.fDistanceArcMinEq(ra, dec, {query.ra_deg}, {query.dec_deg}) "
            f"<= {query.radius_arcsec / 60.0:.8f}"
        )
        response = netclient.get(
            SQL_URL.format(release=self.release),
            {"cmd": sql, "format": "csv"}, timeout=60, provider="sdss",
        )
        sources: list[SourceRef] = []
        for row in parse_csv(response.text, top):
            try:
                object_id = str(row["objID"])
                ra_deg, dec_deg = float(row["ra"]), float(row["dec"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id,
                ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"plate": row.get("plate"), "mjd": row.get("mjd"),
                       "fiber_id": row.get("fiberID"),
                       "spectrum_ready": bool(row.get("plate"))},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # SDSS is not a time-series connector.  A future spectrum job consumes
        # source.extra and writes spectral products, not a fake light curve.
        return []
