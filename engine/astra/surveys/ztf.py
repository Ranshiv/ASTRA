"""Zwicky Transient Facility — time-domain photometry via IRSA.

IRSA exposes ZTF light curves through a plain HTTP endpoint rather than an
astroquery wrapper, so the request is made directly. Object discovery uses
the `ztf_objects_*` catalogues through astroquery.
"""

from __future__ import annotations

import csv
import io

from .. import netclient
from .base import (ConeQuery, LightCurve, SourceRef, SurveyConnector,
                   normalise_band, to_arrays)

LIGHTCURVE_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

# Bit 32768 marks a suspect or contaminated epoch. Excluding it at the archive
# is cheaper than downloading the points and filtering them locally, and keeps
# the "artifact rejection" stage from re-litigating known-bad data.
DEFAULT_CATFLAGS_MASK = 32768

DEFAULT_RELEASE = "dr24"
REQUEST_TIMEOUT_S = 180


class ZTFConnector(SurveyConnector):
    name = "ZTF"
    capabilities = ("catalogue", "light_curve", "image")
    resolution_arcsec = 1.0

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    @property
    def catalog(self) -> str:
        return f"ztf_objects_{self.release}"

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.irsa import Irsa

        coord = SkyCoord(query.ra_deg, query.dec_deg, unit="deg")
        table = Irsa.query_region(
            coordinates=coord,
            catalog=self.catalog,
            spatial="Cone",
            radius=query.radius_arcsec * u.arcsec,
        )
        if table is None or len(table) == 0:
            return []

        sources: list[SourceRef] = []
        for row in table[:limit]:
            oid = str(row["oid"])
            sources.append(SourceRef(
                survey=self.name,
                object_id=oid,
                ra_deg=float(row["ra"]),
                dec_deg=float(row["dec"]),
                extra={
                    "filtercode": str(row["filtercode"])
                    if "filtercode" in table.colnames else "",
                    "field": int(row["field"]) if "field" in table.colnames else -1,
                },
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        """One curve per filter. ZTF returns all bands in a single response."""
        rows = self._request({"ID": source.object_id, "FORMAT": "CSV",
                              "BAD_CATFLAGS_MASK": DEFAULT_CATFLAGS_MASK})
        return self._rows_to_curves(rows, source)

    def fetch_light_curves_at(self, query: ConeQuery) -> list[LightCurve]:
        """Positional variant, for when no object id is known yet."""
        pos = f"CIRCLE {query.ra_deg} {query.dec_deg} {query.radius_deg}"
        rows = self._request({"POS": pos, "FORMAT": "CSV",
                              "BAD_CATFLAGS_MASK": DEFAULT_CATFLAGS_MASK})
        if not rows:
            return []
        source = SourceRef(
            survey=self.name,
            object_id=str(rows[0].get("oid", "unknown")),
            ra_deg=float(rows[0].get("ra", query.ra_deg)),
            dec_deg=float(rows[0].get("dec", query.dec_deg)),
        )
        return self._rows_to_curves(rows, source)

    def _request(self, params: dict) -> list[dict]:
        """Throttled and retrying, so IRSA throttling does not drop objects."""
        response = netclient.get(LIGHTCURVE_URL, params=params,
                                 timeout=REQUEST_TIMEOUT_S, provider="irsa")
        return parse_csv(response.text)

    def _rows_to_curves(self, rows: list[dict], source: SourceRef) -> list[LightCurve]:
        by_band: dict[str, list[tuple[float, float, float]]] = {}
        for row in rows:
            try:
                point = (float(row["hjd"]), float(row["mag"]), float(row["magerr"]))
            except (KeyError, TypeError, ValueError):
                continue  # a malformed epoch must not discard the whole curve
            band = normalise_band(self.name, row.get("filtercode", ""))
            by_band.setdefault(band, []).append(point)

        curves = []
        for band, points in sorted(by_band.items()):
            time, value, value_err = to_arrays(points)
            curves.append(LightCurve(
                source=source,
                release=self.release,
                band=band,
                value_kind="mag",
                time=time,
                value=value,
                value_err=value_err,
                time_system="HJD_UTC",
            ))
        return curves


def parse_csv(text: str) -> list[dict]:
    """Parse an IRSA CSV response, tolerating an empty result set."""
    stripped = text.strip()
    if not stripped:
        return []
    return list(csv.DictReader(io.StringIO(stripped)))
