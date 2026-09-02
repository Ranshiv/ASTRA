"""ANTARES alert-broker connector — another credential-free route to real
ZTF (and, eventually, LSST) alert-stream detections, alongside ALeRCE.

ANTARES (NOIRLab) does not offer a simple flat `?ra=&dec=&radius=` cone
search -- confirmed live this session: those params are silently accepted
and IGNORED by `GET /v1/loci`, which returns an unrelated default listing
with an HTTP 200 rather than an error. Real spatial filtering requires the
Elasticsearch query the `antares_client` reference library
(https://github.com/nsf-noirlab/csdc-antares-client) builds internally: a
`sky_distance` filter keyed on `htm16` (a hierarchical triangular mesh
spatial index), passed as a JSON-encoded `elasticsearch_query[locus_listing]`
query parameter -- reverse-engineered from that client's `search.py` and
then confirmed live against a real position (RA=37.284397, Dec=9.258595,
0.05 deg radius): the naive flat-param request returned an unrelated
object at that same URL, while the real ES-filtered query correctly
returned the true nearby loci, including the exact object queried for
(`ANT2021y65ce`, ZTF object `ZTF21abxxjrh`).

`GET /v1/loci/{locus_id}/alerts` (confirmed live against that same locus)
returns two real alert kinds: `ztf_candidate:*` (an actual PSF-fit
detection, carrying `ztf_magpsf`/`ztf_sigmapsf`/`ztf_fid`) and
`ztf_upper_limit:*` (a non-detection limiting magnitude, no flux measured
-- confirmed live to lack `ztf_magpsf` entirely). `fetch_light_curves`
below keeps only `ztf_candidate:*` alerts for exactly this reason: an
upper limit is not a measurement and must not be plotted as one.
"""

from __future__ import annotations

import json

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector, to_arrays

BASE_URL = "https://api.antares.noirlab.edu/v1"
LOCI_URL = f"{BASE_URL}/loci"
ALERTS_URL_TMPL = f"{BASE_URL}/loci/{{locus_id}}/alerts"

ZTF_FID_TO_BAND = {"1": "g", "2": "r", "3": "i"}


def parse_rows(payload: object) -> list[dict]:
    """ANTARES' JSON:API envelope: `{"data": [...]}`. Keep only real row dicts."""
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


class ANTARESConnector(SurveyConnector):
    name = "ANTARES"
    capabilities = ("catalogue", "light_curve")
    # Genuinely public, no-auth API, same rationale as alerce.py.
    credential_required = False
    resolution_arcsec = 1.0  # ZTF PSF-fit astrometry, same as alerce.py's ztf release
    # New broker, launches opt-in, matching every connector added since
    # sdss.py, until the provider contract is validated more widely.
    enabled_by_default = False

    def __init__(self, release: str = "ztf") -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        es_query = {"query": {"bool": {"filter": {"sky_distance": {
            "distance": f"{query.radius_deg} degree",
            "htm16": {"center": f"{query.ra_deg} {query.dec_deg}"},
        }}}}}
        response = netclient.get(
            LOCI_URL,
            {"elasticsearch_query[locus_listing]": json.dumps(es_query),
             "sort": "-properties.newest_alert_observation_time"},
            timeout=60, provider="antares",
        )
        try:
            rows = parse_rows(response.json())[:top]
        except ValueError:
            rows = []

        sources: list[SourceRef] = []
        for row in rows:
            attrs = row.get("attributes")
            if not isinstance(attrs, dict):
                continue
            try:
                object_id = str(row["id"])
                ra_deg = float(attrs["ra"])
                dec_deg = float(attrs["dec"])
            except (KeyError, TypeError, ValueError):
                continue
            props = attrs.get("properties")
            props = props if isinstance(props, dict) else {}
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"ztf_object_id": props.get("ztf_object_id"),
                       "num_alerts": props.get("num_alerts"),
                       "newest_alert_magnitude": props.get("newest_alert_magnitude"),
                       "anomaly_type": props.get("anomaly_type")},
            ))

        if rows and not sources:
            import logging
            logging.getLogger(__name__).warning(
                "ANTARES: /v1/loci returned %d row(s) but none parsed as a "
                "source -- the JSON:API response shape (attributes.ra/dec, "
                "top-level id) may have changed.", len(rows))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        response = netclient.get(
            ALERTS_URL_TMPL.format(locus_id=source.object_id), {},
            timeout=60, provider="antares",
        )
        try:
            rows = parse_rows(response.json())
        except ValueError:
            rows = []

        by_band: dict[str, list[tuple[float, float, float]]] = {}
        for row in rows:
            # Only a real PSF-fit detection carries ztf_magpsf; a
            # ztf_upper_limit alert is a non-detection and must not be
            # plotted as a measured point -- see module docstring.
            if not str(row.get("id", "")).startswith("ztf_candidate:"):
                continue
            attrs = row.get("attributes")
            if not isinstance(attrs, dict):
                continue
            props = attrs.get("properties")
            if not isinstance(props, dict):
                continue
            try:
                point = (float(attrs["mjd"]), float(props["ztf_magpsf"]),
                        float(props["ztf_sigmapsf"]))
            except (KeyError, TypeError, ValueError):
                continue
            band = ZTF_FID_TO_BAND.get(str(props.get("ztf_fid")), "unknown")
            by_band.setdefault(band, []).append(point)

        curves: list[LightCurve] = []
        for band, points in sorted(by_band.items()):
            time, value, value_err = to_arrays(points)
            curves.append(LightCurve(
                source=source, release=self.release, band=band, value_kind="mag",
                time=time, value=value, value_err=value_err, time_system="MJD_UTC",
            ))
        return curves
