"""JWST observation-metadata connector via MAST CAOM.

Same contract and same service as the Hubble connector, filtered to the JWST
collection: observation metadata only, with product retrieval left to a
separate bounded job.
"""

from __future__ import annotations

from ._mast_caom import fetch_all_pages, parse_rows
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

DEFAULT_RELEASE = "caom"
COLLECTION = "JWST"

__all__ = ["JWSTConnector", "parse_rows"]


class JWSTConnector(SurveyConnector):
    name = "JWST"
    capabilities = ("catalogue", "image_metadata")
    resolution_arcsec = 0.07
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 2000))
        rows = fetch_all_pages(query.ra_deg, query.dec_deg, query.radius_deg,
                               COLLECTION, top, provider="mast")
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row.get("obsid") or row["obs_id"])
                ra_deg = float(row.get("s_ra") if row.get("s_ra") is not None
                               else row["ra"])
                dec_deg = float(row.get("s_dec") if row.get("s_dec") is not None
                                else row["dec"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id,
                ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"instrument": row.get("instrument_name"),
                       "filters": row.get("filters"),
                       "t_min": row.get("t_min"),
                       "exptime": row.get("t_exptime"),
                       "product_type": row.get("dataproduct_type")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # As with HST: imaging and spectroscopy, not a survey light-curve feed.
        return []
