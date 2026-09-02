"""ALeRCE alert-broker connector — a credential-free route to LSST photometry.

ALeRCE (Automatic Learning for the Rapid Classification of Events) is one of
NOIRLab's official community alert brokers for the Rubin Observatory Legacy
Survey of Space and Time (LSST), alongside ANTARES/Fink/Lasair/etc. Its REST
API (https://api.alerce.online/) is genuinely public and unauthenticated, and
its query methods accept a `survey` parameter supporting both "ztf" and
"lsst" — i.e. it already brokers real LSST alerts, with no data-rights
credential wall. That distinguishes this connector from the direct Rubin/LSST
TAP endpoint (https://data.lsst.cloud/api/tap), which does require one; see
docs/LIMITATIONS.md for the full history of that blocker.

Unlike the metadata-only opt-in connectors (sdss.py, panstarrs.py, chandra.py,
swift.py, xmm.py, des.py, hubble.py, jwst.py), ALeRCE serves real per-object
detections (mjd/magpsf/sigmapsf/fid or band), so `fetch_light_curves()` here
is a genuine implementation, not a `[]` stub.

`cone_search`/`fetch_light_curves` for `survey="ztf"` were confirmed live
this session (roadmap item 21): a real cone search around RA=180.122,
Dec=22.411 returns real `{"items": [...]}`-wrapped objects with populated
`oid`/`meanra`/`meandec` (note: `GET /ztf/v1/objects` without a trailing
slash 302-redirects to the trailing-slash form -- `requests` follows this
by default, costing one extra round trip, not a correctness problem), and
`fetch_light_curves` against a real returned `oid` returns a real bare-list
detections response with populated `mjd`/`magpsf`/`sigmapsf`/`fid`. Both
match this module's existing field-name assumptions exactly -- no bug
found. `survey="lsst"` remains unverified (no live LSST alert has been
checked this session); "/ztf/v1/" was confirmed to resolve as a live host,
so LSST queries are requested against that same confirmed path with
survey="lsst" rather than a hypothetical, never-confirmed "/lsst/v1/" path,
per the reasoning already stated here before this session's verification.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector, to_arrays

BASE_URL = "https://api.alerce.online"
# The ALeRCE Python client selects the underlying survey via a `survey=`
# kwarg on its query methods rather than a separate URL namespace per survey,
# so both ztf and lsst queries hit this one confirmed-live path.
OBJECTS_URL = f"{BASE_URL}/ztf/v1/objects"
DETECTIONS_URL_TMPL = f"{BASE_URL}/ztf/v1/objects/{{oid}}/detections"

DEFAULT_RELEASE = "ztf"
SUPPORTED_RELEASES = ("ztf", "lsst")

# ZTF ~1" typical PSF-fit astrometry; LSST is seeing-limited and sharper.
RESOLUTION_ARCSEC = {"ztf": 1.0, "lsst": 0.2}

ZTF_FID_TO_BAND = {"1": "g", "2": "r", "3": "i"}
LSST_BANDS = {"u", "g", "r", "i", "z", "y"}


def parse_rows(payload: object, limit: int = 100) -> list[dict]:
    """ALeRCE's response shape is not live-confirmed; accept a bare list or a
    dict wrapping the rows under "items" or "results", and keep only real
    row dicts either way."""
    if isinstance(payload, dict):
        candidate = payload.get("items")
        if not isinstance(candidate, list):
            candidate = payload.get("results")
        payload = candidate
    if not isinstance(payload, list):
        return []
    return [row for row in payload[:limit] if isinstance(row, dict)]


class ALeRCEConnector(SurveyConnector):
    name = "ALeRCE"
    capabilities = ("catalogue", "light_curve")
    # Genuinely public, no-auth API — the actual point of this connector.
    credential_required = False
    # New survey launches opt-in, matching every connector added since
    # sdss.py, until the provider contract is validated against the live
    # service.
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release.lower()
        if self.release not in SUPPORTED_RELEASES:
            raise ValueError(
                f"unsupported ALeRCE survey release: {release!r}; "
                f"expected one of {SUPPORTED_RELEASES}"
            )
        self.resolution_arcsec = RESOLUTION_ARCSEC[self.release]

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        response = netclient.get(
            OBJECTS_URL,
            {"survey": self.release, "ra": query.ra_deg, "dec": query.dec_deg,
             "radius": query.radius_arcsec, "page": 1, "page_size": top,
             "count": "false"},
            timeout=60, provider="alerce",
        )
        try:
            rows = parse_rows(response.json(), top)
        except ValueError:
            rows = []
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row["oid"])
                ra_deg = float(row.get("meanra") if row.get("meanra") is not None else row["ra"])
                dec_deg = float(row.get("meandec") if row.get("meandec") is not None else row["dec"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"classifier": row.get("classifier") or row.get("classifier_name"),
                       "class_name": row.get("class_name") or row.get("class"),
                       "probability": row.get("probability"),
                       "ndet": row.get("ndet"),
                       "firstmjd": row.get("firstmjd"),
                       "lastmjd": row.get("lastmjd")},
            ))
        return sources

    def query_classified_objects(self, class_name: str, classifier: str = "lc_classifier",
                                 min_probability: float = 0.5, limit: int = 100
                                 ) -> list[SourceRef]:
        """Real ALeRCE-broker-classified objects, filtered by class -- not a
        spatial query (unlike `cone_search`).

        Live-verified this session: `classifier=lc_classifier&class=SNIa
        &order_by=probability&order_mode=DESC` against the real, confirmed-
        live `/ztf/v1/objects` endpoint returns real classified ZTF
        supernova candidates with populated `oid`/`class`/`classifier`/
        `probability` fields (a plain positional cone search on this same
        endpoint returns these fields as `null` for most objects -- the
        classifier filter is what actually selects classified rows).

        This is the real-transient source `open_world_eval.py`'s held-out
        set is built from: a classifier-assigned label plus a probability,
        both genuinely reported by the broker, not invented here.
        """
        top = max(1, min(int(limit), 200))
        response = netclient.get(
            OBJECTS_URL,
            {"survey": self.release, "classifier": classifier, "class": class_name,
             "page": 1, "page_size": top, "count": "false",
             "order_by": "probability", "order_mode": "DESC"},
            timeout=60, provider="alerce",
        )
        try:
            rows = parse_rows(response.json(), top)
        except ValueError:
            rows = []

        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row["oid"])
                ra_deg = float(row.get("meanra") if row.get("meanra") is not None else row["ra"])
                dec_deg = float(row.get("meandec") if row.get("meandec") is not None else row["dec"])
                probability = float(row.get("probability"))
            except (KeyError, TypeError, ValueError):
                continue
            if probability < min_probability:
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"classifier": row.get("classifier") or row.get("classifier_name") or classifier,
                       "class_name": row.get("class_name") or row.get("class") or class_name,
                       "probability": probability,
                       "ndet": row.get("ndet"),
                       "firstmjd": row.get("firstmjd"),
                       "lastmjd": row.get("lastmjd")},
            ))
        return sources

    def _band_for_row(self, row: dict) -> str:
        if self.release == "ztf":
            return ZTF_FID_TO_BAND.get(str(row.get("fid")), "unknown")
        band = str(row.get("band") or row.get("fid") or "").strip().lower()
        return band if band in LSST_BANDS else "unknown"

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        response = netclient.get(
            DETECTIONS_URL_TMPL.format(oid=source.object_id),
            {"survey": self.release}, timeout=60, provider="alerce",
        )
        try:
            rows = parse_rows(response.json(), limit=100_000)
        except ValueError:
            rows = []

        by_band: dict[str, list[tuple[float, float, float]]] = {}
        for row in rows:
            try:
                point = (float(row["mjd"]), float(row["magpsf"]), float(row["sigmapsf"]))
            except (KeyError, TypeError, ValueError):
                continue
            by_band.setdefault(self._band_for_row(row), []).append(point)

        curves: list[LightCurve] = []
        for band, points in sorted(by_band.items()):
            time, value, value_err = to_arrays(points)
            curves.append(LightCurve(
                source=source, release=self.release, band=band, value_kind="mag",
                time=time, value=value, value_err=value_err, time_system="MJD_UTC",
            ))
        return curves


def photometric_residual(curves: Iterable[LightCurve],
                         reference_mags: dict[str, float]) -> dict[str, dict]:
    """Per-band offset between ALeRCE-reported magnitudes and a reference.

    `reference_mags` (e.g. `{"g": 17.42, "r": 17.05}`) is supplied by the
    caller -- typically built from `crossmatch.match_catalogs()` against
    stored Gaia/PS1 photometry for the same object -- so this function is a
    pure per-band comparison, not a crossmatch. A large median residual or
    scatter flags a calibration problem worth a researcher's attention; it
    is diagnostic evidence, never a correction applied back to the stored
    curve, and it is not wired into scoring.WEIGHTS.
    """
    result: dict[str, dict] = {}
    for curve in curves:
        reference = reference_mags.get(curve.band)
        if reference is None or curve.value_kind != "mag":
            continue
        finite = curve.dropna()
        if len(finite) == 0:
            continue
        values = finite.value.astype(float)
        median_value = float(np.median(values))
        # Median absolute deviation, scaled to a sigma-equivalent scatter.
        scatter = float(np.median(np.abs(values - median_value))) * 1.4826
        result[curve.band] = {
            "residual_mag": median_value - float(reference),
            "scatter_mag": scatter,
            "n_points": int(len(values)),
        }
    return result
