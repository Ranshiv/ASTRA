"""Pan-STARRS DR2 mean-photometry metadata connector via MAST.

A galaxy angular-size proxy (`rKronRad`) was attempted for roadmap item 31
(`host_association.py`), which needs one. Confirmed live this session,
via the real `mean` endpoint's own column-metadata response (the
`info` list `cone_search` gets back alongside `data`): the `mean` object
view this connector queries has NO Kron-radius or any other size/shape
column at all -- only per-band `MeanKronMag` (a Kron *magnitude*, not a
radius). A real size column (`{band}KronRad`) exists only on the separate
`stack` endpoint (confirmed live the same session), which this connector
does not query. This is therefore a real, confirmed data-source gap, not
an unverified assumption: `host_association.py` gets no size proxy from
Pan-STARRS and falls back to its own `DEFAULT_R_E_ARCSEC` for any
Pan-STARRS-only candidate. Switching this connector to `stack` (a
differently shaped, per-detection rather than per-object table) would be
a real scope change to its existing `mean_photometry` contract, not
attempted here.
"""

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
                       "y_mean": row.get("yMeanPSFMag"),
                       # Pan-STARRS DR2's mean table publishes these
                       # alongside the magnitudes above; not previously
                       # requested here. photometric_calibration.py needs
                       # them to weight matched pairs by uncertainty.
                       "g_mean_error": row.get("gMeanPSFMagErr"),
                       "r_mean_error": row.get("rMeanPSFMagErr"),
                       "i_mean_error": row.get("iMeanPSFMagErr"),
                       "z_mean_error": row.get("zMeanPSFMagErr"),
                       "y_mean_error": row.get("yMeanPSFMagErr")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        return []
