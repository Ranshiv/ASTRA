"""ASAS-SN (All-Sky Automated Survey for Supernovae) metadata connector.

The clearest gap this project's own review named: a free, no-auth, all-sky
survey with a long (~2012-present, over a decade for the original ASAS-SN
footprint) multi-epoch g/V baseline, absent from this codebase entirely
before this connector. Confirmed live this session: ASAS-SN Sky Patrol's
own cone-search server (not VizieR -- ASAS-SN publishes through its own
infrastructure at the University of Hawaii IfA, `asassn-lb01.ifa.hawaii.edu`,
port 9006, plain HTTP) answers a `POST /lookup_cone/radius{deg}_ra{deg}_
dec{deg}` with a JSON body `{"catalog": ..., "cols": [...], "format":
"arrow", "download": false}` -- despite the `"format": "arrow"` field name,
the response body is real Parquet bytes (`pyarrow.parquet` reads it
directly), not Arrow IPC.

`catalog: "stellar_main"` is ASAS-SN's REFCAT2-derived master source list;
confirmed live returning real `asas_sn_id`/`ra_deg`/`dec_deg`/`gaia_mag`/
`gaia_b_mag`/`gaia_r_mag` rows (e.g. a cone at RA=280.0, Dec=-20.0, 0.05 deg
returned 17-24 real rows across three live checks this session). This is
mean photometry (Gaia-band magnitudes ASAS-SN's own pipeline carries
alongside its source list), not ASAS-SN's own g/V time series.

The actual multi-epoch g/V light curves live behind a materially different,
undocumented protocol (`download=true`): the server returns a query hash
and a set of block-server assignments, and the client must then GET each
block separately and reassemble them -- a stateful, multi-request streaming
protocol built around the `pyasassn` reference client
(https://github.com/asas-sn/skypatrol), not the single-request contract
every other connector in this codebase relies on. Implementing that
block-streaming protocol is real, scoped-out follow-up work, the same
"discovery now, light-curve extraction later as its own bounded job"
boundary `panstarrs.py`/`sdss.py`/the VizieR-backed connectors already draw
-- `fetch_light_curves` here is metadata-only for the same reason theirs is.
"""

from __future__ import annotations

import io

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

LOOKUP_CONE_URL = "http://asassn-lb01.ifa.hawaii.edu:9006/lookup_cone/radius{radius_deg}_ra{ra_deg}_dec{dec_deg}"
DEFAULT_CATALOG = "stellar_main"
COLUMNS = ("asas_sn_id", "ra_deg", "dec_deg", "gaia_mag", "gaia_b_mag", "gaia_r_mag")


class ASASSNConnector(SurveyConnector):
    name = "ASAS-SN"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 2.0  # REFCAT2-derived astrometry; a few arcsec class
    # New survey, launches opt-in until this session's live verification is
    # reproduced against a wider range of positions -- same precedent as
    # every other connector added this way (chandra.py/swift.py/xmm.py/...).
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_CATALOG) -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        import logging

        response = netclient.post(
            LOOKUP_CONE_URL.format(
                radius_deg=query.radius_deg, ra_deg=query.ra_deg, dec_deg=query.dec_deg),
            json={"catalog": self.release, "cols": list(COLUMNS),
                 "format": "arrow", "download": False},
            timeout=60, provider="asassn",
            headers={"Content-Type": "application/json"},
        )
        rows = _parse_rows(response.content)[:top]
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row["asas_sn_id"])
                ra_deg = float(row["ra_deg"])
                dec_deg = float(row["dec_deg"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"gaia_mag": row.get("gaia_mag"),
                       "gaia_b_mag": row.get("gaia_b_mag"),
                       "gaia_r_mag": row.get("gaia_r_mag")},
            ))
        if rows and not sources:
            logging.getLogger(__name__).warning(
                "ASAS-SN: lookup_cone returned %d row(s) but none parsed as "
                "a source -- asas_sn_id/ra_deg/dec_deg may no longer match "
                "the %r catalog's real columns.", len(rows), self.release)
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # ASAS-SN's own multi-epoch g/V photometry requires the block-
        # streaming download protocol -- see this module's docstring.
        return []


def _parse_rows(content: bytes) -> list[dict]:
    """Parquet bytes (mislabeled "arrow" by the server) -> plain row dicts."""
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(content))
    columns = [name for name in table.column_names if not name.startswith("__index_level")]
    return [dict(zip(columns, values)) for values in zip(*(
        table.column(name).to_pylist() for name in columns))]
