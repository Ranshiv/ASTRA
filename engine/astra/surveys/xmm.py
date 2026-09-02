"""XMM-Newton Serendipitous Source Catalog (4XMM) metadata connector.

Metadata-only, matching the Chandra/Swift/SDSS/Pan-STARRS shape: this
connector discovers X-ray counterparts from the 4XMM catalogue and their
basic properties, and deliberately returns no light curves. XMM's own
per-source time-series products exist but extracting and normalising them
is a separate, bounded job -- not something this adapter builds implicitly.

`cone_search` now targets VizieR (CDS) rather than this module's previous
`CONE_URL` (`xmmssc-www.star.le.ac.uk/newpages/xsa_cone`): that host was
found to not resolve at all (`ConnectionError`) -- a dead endpoint, not
something worth guessing a replacement path for. 4XMM-DR13 is real and
live on VizieR as `IX/69` (Webb et al. 2023), confirmed with a real cone
search returning `Flux8` (0.2-12 keV total flux) and 4XMM's own
pre-computed `HR1`-`HR4` hardness ratios (adjacent-band ratios across the
catalogue's five standard energy bands) -- what `xray_hardness_eval.py`'s
flux/hardness calibration check compares against -- `query_hardness_
ratios` below and `cone_search` now share the same `_vizier_rows` fetch
rather than duplicating it. Individual per-band fluxes were NOT found in
this VizieR mirror's default column set (only the pre-computed HRs and
total flux); querying them, if needed, would require either XMM's own SSC
TAP service or an explicit VizieR column request, neither attempted here --
a real, stated [GAP].

Also confirmed live: XMM, like Chandra, is a POINTED observatory with
patchy sky coverage (unlike VLASS/eROSITA); and VizieR's `4XMM` column
holds only the coordinate-designation SUFFIX (e.g. "J123049.2+122330"),
not a "4XMM "-prefixed full name -- a real naming assumption an early
version of this session's own test got wrong, caught by the live check
and corrected there rather than left unnoticed.
"""

from __future__ import annotations

from ._vizier import VizierConeConnector, vizier_cone_rows

VIZIER_CATALOG = "IX/69"

DEFAULT_RELEASE = "4xmm-dr13"


class XMMConnector(VizierConeConnector):
    name = "XMM-Newton"
    capabilities = ("catalogue", "xray_metadata")
    resolution_arcsec = 6.0
    # New X-ray surveys launch opt-in, matching Chandra/Swift/SDSS/Pan-STARRS
    # precedent, until the provider contract (endpoint, auth, rate limits)
    # is validated against the real service.
    enabled_by_default = False
    id_column = "4XMM"
    ra_column = "RA_ICRS"
    dec_column = "DE_ICRS"

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        # `release` here is a human-readable label ("4xmm-dr13"), not the
        # VizieR catalogue id -- see `_catalog` below and the module docstring.
        super().__init__(release)

    def _catalog(self) -> str:
        return VIZIER_CATALOG

    def extra_fields(self, row: dict) -> dict:
        return {"flux_total": row.get("Flux8"), "flux_total_err": row.get("e_Flux8"),
                "hr1": row.get("HR1"), "hr2": row.get("HR2"),
                "hr3": row.get("HR3"), "hr4": row.get("HR4")}


def query_hardness_ratios(ra_deg: float, dec_deg: float, radius_arcsec: float, limit: int = 100
                          ) -> list[dict]:
    """Real 4XMM-DR13 total flux and pre-computed hardness ratios near a
    position, via the same VizieR `IX/69` fetch `cone_search` uses.
    """
    top = max(1, min(int(limit), 200))
    rows = vizier_cone_rows(VIZIER_CATALOG, ra_deg, dec_deg, radius_arcsec, top)
    results: list[dict] = []
    for row in rows:
        try:
            object_id = str(row["4XMM"])
            ra_val, dec_val = float(row["RA_ICRS"]), float(row["DE_ICRS"])
        except (KeyError, TypeError, ValueError):
            continue
        results.append({
            "object_id": object_id, "ra_deg": ra_val, "dec_deg": dec_val,
            "flux_total": row.get("Flux8"), "flux_total_err": row.get("e_Flux8"),
            "hr1": row.get("HR1"), "hr2": row.get("HR2"),
            "hr3": row.get("HR3"), "hr4": row.get("HR4"),
        })
    return results
