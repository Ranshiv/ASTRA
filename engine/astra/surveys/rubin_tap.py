"""Rubin/LSST direct TAP connector — dormant, credential-gated.

`data.lsst.cloud/api/tap` (the Rubin Science Platform's own TAP service)
requires a real data-rights account token; ASTRA does not have one, and
`docs/LIMITATIONS.md` explicitly documents the resulting decision made while
planning this module: building a direct-TAP connector before a real token
exists would be speculative effort with no way to validate it against the
live service. `surveys/alerce.py` already delivers real, credential-free
LSST alerts/photometry through the community broker route and remains the
recommended path for LSST data today.

This connector exists anyway, DELIBERATELY DORMANT, so the moment a real
Rubin data-rights token is available the only remaining step is calling
`credentials.save_credentials("rubin", {...})` — no further code changes.
Everything here is written and tested only against mocked TAP responses; the
ADQL table/column names below (`dp02_dc2_catalogs.Object`-style, matching
Rubin's published Data Preview 0.2 schema documentation) are DOCUMENTED, NOT
LIVE-CONFIRMED, carrying the same "validate against the real service before
trusting a negative result" caveat every other metadata-only connector in
this package already carries (chandra.py/swift.py/xmm.py/des.py/hubble.py/
jwst.py) — doubly so here, since there is no way to even attempt a live
check without a token.

`credential_required = True` follows the same DPAPI-backed credential
pattern the TNS integration already established
(`credentials.save_credentials`/`load_credentials`, keyed by provider name
`"rubin"`), generalized in that module specifically so this connector would
not need a bespoke third wrapper.
"""

from __future__ import annotations

from .. import credentials, tap
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

TAP_SERVICE = "https://data.lsst.cloud/api/tap"
DEFAULT_RELEASE = "dp02"
# Rubin's published Data Preview 0.2 schema (DOCUMENTED, not live-confirmed --
# see module docstring). objectId/coord_ra/coord_dec/mag_g are the DP0.2
# `Object` catalog's own documented column names.
DEFAULT_TABLE = "dp02_dc2_catalogs.Object"
RESOLUTION_ARCSEC = 0.2  # seeing-limited LSST, matching alerce.py's LSST entry


class RubinTAPError(RuntimeError):
    """The Rubin TAP connector could not produce a usable result."""


def _auth_header(credential: dict) -> dict[str, str]:
    token = credential.get("token") or credential.get("access_token")
    if not token:
        raise RubinTAPError("stored Rubin credentials do not contain a token")
    return {"Authorization": f"Bearer {token}"}


def build_cone_adql(ra_deg: float, dec_deg: float, radius_arcsec: float,
                    *, table: str = DEFAULT_TABLE, limit: int = 100) -> str:
    """A read-only ADQL cone-search query against the DP0.2 Object catalog.

    `CIRCLE`/`CONTAINS`/`POINT` is the standard IVOA ADQL cone-search idiom
    already used implicitly by `astroquery.gaia.Gaia.cone_search` elsewhere
    in this package; `tap.bound_adql` still injects and enforces the row cap
    independently, so `limit` here is advisory, not the sole bound.
    """
    radius_deg = float(radius_arcsec) / 3600.0
    return (
        f"SELECT TOP {int(limit)} objectId, coord_ra, coord_dec, mag_g, mag_r, mag_i "
        f"FROM {table} WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
        f"CIRCLE('ICRS', {float(ra_deg)}, {float(dec_deg)}, {radius_deg})) = 1"
    )


class RubinTAPConnector(SurveyConnector):
    name = "Rubin"
    capabilities = ("catalogue",)
    credential_required = True
    enabled_by_default = False
    resolution_arcsec = RESOLUTION_ARCSEC

    def __init__(self, release: str = DEFAULT_RELEASE, table: str = DEFAULT_TABLE) -> None:
        self.release = release
        self.table = table

    def _credential_auth_header(self) -> dict[str, str]:
        credential = credentials.load_credentials("rubin")
        if credential is None:
            raise RubinTAPError(
                "Rubin/LSST direct TAP access requires a stored data-rights "
                "token; call credentials.save_credentials('rubin', {'token': ...}) "
                "first, or use surveys/alerce.py for credential-free LSST access")
        return _auth_header(credential)

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        auth_header = self._credential_auth_header()
        adql = build_cone_adql(query.ra_deg, query.dec_deg, query.radius_arcsec,
                               table=self.table, limit=limit)
        result = tap.query(TAP_SERVICE, adql, release=self.release, provider="rubin",
                           auth_header=auth_header, max_rows=limit)
        sources: list[SourceRef] = []
        for row in result.get("rows", []):
            try:
                object_id = str(row["objectid"] if "objectid" in row else row["objectId"])
                ra_deg = float(row.get("coord_ra"))
                dec_deg = float(row.get("coord_dec"))
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"mag_g": row.get("mag_g"), "mag_r": row.get("mag_r"),
                       "mag_i": row.get("mag_i")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # DP0.2's Object catalog is a co-added catalogue, not a per-visit
        # forced-photometry table; per-object time series through this direct
        # TAP path is future work once a real schema/credential are in hand.
        # surveys/alerce.py already delivers real LSST detections today.
        return []


__all__ = ["RubinTAPConnector", "RubinTAPError", "build_cone_adql",
          "TAP_SERVICE", "DEFAULT_TABLE"]
