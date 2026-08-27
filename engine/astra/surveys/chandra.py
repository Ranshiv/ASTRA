"""Chandra Source Catalog (CSC) X-ray metadata connector.

Metadata-only, matching the SDSS/Pan-STARRS shape: this connector discovers
X-ray counterparts and their basic properties, and deliberately returns no
light curves. CSC per-source time-series extraction (from event files) is a
separate, bounded job, not something this adapter builds implicitly.

`query_band_fluxes`, added for roadmap item 23, is a new additive function
-- `cone_search` above is unchanged. It targets VizieR (CDS) rather than
this module's own `CONE_URL`: that endpoint (`cda.cfa.harvard.edu/csccli/
browse`) was found live this session to require an undocumented
`packageset` parameter this codebase does not know a valid value for
(confirmed: omitting it gives `400 Missing packageset`, and several
plausible guessed values all gave `404 No file found matching these
parameters`) -- a genuine, pre-existing gap in this connector's live
contract, not something this session's additive function should route
around by guessing further. CSC 2.1 is ALSO real and live on VizieR as
`IX/70` (Evans et al. 2024), confirmed this session with a real cone
search returning real per-band fluxes (`Fluxb`/`Fluxh`/`Fluxm`/`Fluxs`/
`Fluxu`/`Fluxw` -- broad/hard/medium/soft/ultrasoft/wide) AND CSC's own
pre-computed hardness ratios (`HRhm`/`HRhs`/`HRms`), which is what
`xray_hardness_eval.py`'s flux/hardness calibration check compares against.

Also confirmed live this session: Chandra is a POINTED observatory, unlike
the genuinely all-sky VLASS/eROSITA surveys added alongside this one --
`query_band_fluxes` at an arbitrary position (RA=180, Dec=0, no known
Chandra pointing) returns zero rows, real patchy-coverage behaviour, not a
bug. M87 (RA=187.7059, Dec=12.3911), one of the most Chandra-observed
targets in the sky, returns 99 real rows -- the live test below queries
that position for exactly this reason.
"""

from __future__ import annotations

from .. import netclient
from ..tap import parse_votable
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

VIZIER_SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
VIZIER_CATALOG = "IX/70"

DEFAULT_RELEASE = "csc2.1"
CONE_URL = "https://cda.cfa.harvard.edu/csccli/browse"


def parse_rows(payload: object, limit: int = 100) -> list[dict]:
    """CSC's browse endpoint returns a list of row dicts; keep only real rows."""
    if not isinstance(payload, list):
        return []
    return [row for row in payload[:limit] if isinstance(row, dict)]


class ChandraConnector(SurveyConnector):
    name = "Chandra"
    capabilities = ("catalogue", "xray_metadata")
    resolution_arcsec = 0.5
    # New X-ray surveys launch opt-in, matching SDSS/Pan-STARRS precedent,
    # until the provider contract (endpoint, auth, rate limits) is validated.
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        response = netclient.get(
            CONE_URL,
            {"ra": query.ra_deg, "dec": query.dec_deg,
             "radius": query.radius_arcsec / 60.0, "catalog": self.release,
             "limit": top, "format": "json"},
            timeout=60, provider="chandra",
        )
        try:
            rows = parse_rows(response.json(), top)
        except ValueError:
            rows = []
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row.get("name") or row["src_id"])
                ra_deg = float(row.get("ra") if row.get("ra") is not None else row["ra_deg"])
                dec_deg = float(row.get("dec") if row.get("dec") is not None else row["dec_deg"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id,
                ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"obsid": row.get("obsid"),
                       "instrument": row.get("instrument"),
                       "flux_aper_b": row.get("flux_aper_b")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # CSC event-file/lightcurve extraction is a separate bounded job, same
        # rationale as sdss.py / panstarrs.py: this connector is discovery and
        # catalogue metadata only.
        return []


def query_band_fluxes(ra_deg: float, dec_deg: float, radius_arcsec: float, limit: int = 100
                      ) -> list[dict]:
    """Real per-band CSC 2.1 fluxes and pre-computed hardness ratios near a
    position, via VizieR `IX/70` -- see this module's docstring for why
    VizieR, not this connector's own `CONE_URL`.
    """
    top = max(1, min(int(limit), 200))
    response = netclient.get(
        VIZIER_SCS_URL,
        {"-source": VIZIER_CATALOG, "RA": ra_deg, "DEC": dec_deg,
         "SR": radius_arcsec / 3600.0, "-out.max": top},
        timeout=60, provider="vizier",
    )
    rows = parse_votable(response.text, top)
    results: list[dict] = []
    for row in rows:
        try:
            object_id = str(row["2CXO"])
            ra_val, dec_val = float(row["RAICRS"]), float(row["DEICRS"])
        except (KeyError, TypeError, ValueError):
            continue
        results.append({
            "object_id": object_id, "ra_deg": ra_val, "dec_deg": dec_val,
            "flux_broad": row.get("Fluxb"), "flux_soft": row.get("Fluxs"),
            "flux_medium": row.get("Fluxm"), "flux_hard": row.get("Fluxh"),
            "hr_hard_medium": row.get("HRhm"), "hr_hard_soft": row.get("HRhs"),
            "hr_medium_soft": row.get("HRms"),
        })
    return results
