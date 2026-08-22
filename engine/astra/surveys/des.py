"""Dark Energy Survey DR2 catalogue connector via NOIRLab Astro Data Lab.

DES is a co-add survey, not a time-domain one: the public DR2 release exposes
stacked photometry per object rather than per-epoch measurements.  The
connector therefore surfaces catalogue metadata only, so a DES counterpart can
be discovered and its identifier preserved for a later bounded product job.

Access is through Astro Data Lab's anonymous TAP endpoint.  The ADQL text is
fixed and parameterised by ASTRA -- callers cannot supply their own query.
"""

from __future__ import annotations

import csv
import io

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

DEFAULT_RELEASE = "dr2"
TAP_URL = "https://datalab.noirlab.edu/tap/sync"
PAGE_SIZE = 200
MAX_PAGES = 20


def parse_csv(payload: str, limit: int = 100) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload))
    return [dict(row) for row in list(reader)[:limit]]


class DESConnector(SurveyConnector):
    name = "DES"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 0.9
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    def build_query(self, query: ConeQuery, top: int, after_id: str | None = None) -> str:
        """Fixed ADQL for one cone, keyset-paginated by coadd_object_id.

        No user SQL reaches the service. Standard ADQL has no portable
        OFFSET, so pages beyond the first are requested with
        `coadd_object_id > after_id` instead, ordered ascending so every
        page picks up exactly where the previous one stopped.
        """
        clause = ("WHERE CONTAINS(POINT('ICRS', ra, dec), "
                  f"CIRCLE('ICRS', {query.ra_deg}, {query.dec_deg}, "
                  f"{query.radius_deg:.8f})) = 1")
        if after_id is not None:
            clause += f" AND coadd_object_id > {after_id}"
        return (
            f"SELECT TOP {top} coadd_object_id, ra, dec, "
            "mag_auto_g, mag_auto_r, mag_auto_i, mag_auto_z "
            f"FROM des_{self.release}.main "
            f"{clause} ORDER BY coadd_object_id ASC"
        )

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        remaining = max(1, min(int(limit), 2000))
        sources: list[SourceRef] = []
        after_id: str | None = None
        for _ in range(MAX_PAGES):
            page_size = max(1, min(remaining, PAGE_SIZE))
            response = netclient.get(
                TAP_URL,
                {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                 "MAXREC": page_size,
                 "QUERY": self.build_query(query, page_size, after_id)},
                timeout=60, provider="datalab",
            )
            rows = parse_csv(response.text, page_size)
            if not rows:
                break
            for row in rows:
                try:
                    object_id = str(row["coadd_object_id"])
                    ra_deg, dec_deg = float(row["ra"]), float(row["dec"])
                except (KeyError, TypeError, ValueError):
                    continue
                sources.append(SourceRef(
                    survey=self.name, object_id=object_id,
                    ra_deg=ra_deg, dec_deg=dec_deg,
                    extra={"g_mean": row.get("mag_auto_g"),
                           "r_mean": row.get("mag_auto_r"),
                           "i_mean": row.get("mag_auto_i"),
                           "z_mean": row.get("mag_auto_z")},
                ))
            after_id = str(rows[-1]["coadd_object_id"])
            remaining -= len(rows)
            if len(rows) < page_size or remaining <= 0:
                break
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # DR2 publishes co-added photometry only.  Single-epoch extraction is a
        # separate bounded job, not something this adapter fabricates.
        return []
