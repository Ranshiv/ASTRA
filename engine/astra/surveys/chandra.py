"""Chandra Source Catalog (CSC) X-ray metadata connector.

Metadata-only, matching the SDSS/Pan-STARRS shape: this connector discovers
X-ray counterparts and their basic properties, and deliberately returns no
light curves. CSC per-source time-series extraction (from event files) is a
separate, bounded job, not something this adapter builds implicitly.

`cone_search` now targets VizieR (CDS) rather than this module's previous
`CONE_URL` (`cda.cfa.harvard.edu/csccli/browse`): that endpoint was found
live to require an undocumented `packageset` parameter this codebase does
not know a valid value for (confirmed: omitting it gives `400 Missing
packageset`, and several plausible guessed values all gave `404 No file
found matching these parameters`) -- a genuine, unfixed gap in that
endpoint's contract, not something worth routing around by guessing
further. CSC 2.1 is real and live on VizieR as `IX/70` (Evans et al.
2024), confirmed with a real cone search returning real per-band fluxes
(`Fluxb`/`Fluxh`/`Fluxm`/`Fluxs`/`Fluxu`/`Fluxw` -- broad/hard/medium/soft/
ultrasoft/wide) AND CSC's own pre-computed hardness ratios (`HRhm`/`HRhs`/
`HRms`), which is what `xray_hardness_eval.py`'s flux/hardness calibration
check compares against -- `query_band_fluxes` below and `cone_search` now
share the same `_vizier_rows` fetch rather than duplicating it.

Also confirmed live: Chandra is a POINTED observatory, unlike the
genuinely all-sky VLASS/eROSITA surveys added alongside this one -- a
query at an arbitrary position (RA=180, Dec=0, no known Chandra pointing)
returns zero rows, real patchy-coverage behaviour, not a bug. M87
(RA=187.7059, Dec=12.3911), one of the most Chandra-observed targets in
the sky, returns 99 real rows -- the live test below queries that position
for exactly this reason.
"""

from __future__ import annotations

from ._vizier import VizierConeConnector, vizier_cone_rows

VIZIER_CATALOG = "IX/70"

DEFAULT_RELEASE = "csc2.1"


class ChandraConnector(VizierConeConnector):
    name = "Chandra"
    capabilities = ("catalogue", "xray_metadata")
    resolution_arcsec = 0.5
    # New X-ray surveys launch opt-in, matching SDSS/Pan-STARRS precedent,
    # until the provider contract (endpoint, auth, rate limits) is validated.
    enabled_by_default = False
    id_column = "2CXO"
    ra_column = "RAICRS"
    dec_column = "DEICRS"

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        # `release` here is a human-readable label ("csc2.1"), not the VizieR
        # catalogue id -- see `_catalog` below and the module docstring.
        super().__init__(release)

    def _catalog(self) -> str:
        return VIZIER_CATALOG

    def extra_fields(self, row: dict) -> dict:
        return {"flux_broad": row.get("Fluxb"), "flux_soft": row.get("Fluxs"),
                "flux_medium": row.get("Fluxm"), "flux_hard": row.get("Fluxh"),
                "hr_hard_medium": row.get("HRhm"), "hr_hard_soft": row.get("HRhs"),
                "hr_medium_soft": row.get("HRms")}


def query_band_fluxes(ra_deg: float, dec_deg: float, radius_arcsec: float, limit: int = 100
                      ) -> list[dict]:
    """Real per-band CSC 2.1 fluxes and pre-computed hardness ratios near a
    position, via the same VizieR `IX/70` fetch `cone_search` uses.
    """
    top = max(1, min(int(limit), 200))
    rows = vizier_cone_rows(VIZIER_CATALOG, ra_deg, dec_deg, radius_arcsec, top)
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
