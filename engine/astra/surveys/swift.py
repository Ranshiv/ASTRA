"""Swift-XRT point-source catalog (SXPS) metadata connector.

Metadata-only, matching the Chandra/SDSS/Pan-STARRS shape: this connector
discovers X-ray counterparts detected by Swift's XRT instrument and their
basic properties, and deliberately returns no light curves. Swift XRT does
publish per-source time series (it is a genuinely time-domain X-ray mission),
but extracting and normalising them is a separate, bounded job -- not
something this adapter builds implicitly.

`cone_search` now targets VizieR (CDS) rather than this module's previous
`CONE_URL` (`swift.ac.uk/xrt_products/sxpsCone`): that endpoint was found to
return `404 Not Found` for a real query -- a dead or moved endpoint, not
something worth guessing a replacement path for. 2SXPS is real and live on
VizieR as `IX/58` (Evans et al. 2020), confirmed with a real cone search
returning `CR0` (total 0.3-10 keV count rate) and 2SXPS's own pre-computed
`HR1`/`HR2` -- per that paper's own stated definitions,
`HR1 = (M-S)/(M+S)` and `HR2 = (H-M)/(H+M)` for Soft=0.3-1, Medium=1-2,
Hard=2-10 keV count rates, the same `(x-y)/(x+y)` convention
`xray_hardness.hardness_ratio` implements generically -- `query_hardness_
ratios` below and `cone_search` now share the same `_vizier_rows` fetch
rather than duplicating it. 2SXPS additionally publishes real per-source,
multi-epoch light curves in four energy bands plus these two hardness
ratios (per that paper's own abstract) -- a real, better-than-single-epoch
data source for `xray_hardness_eval.py`'s state-transition-detection
validation, though extracting that per-epoch time series (as opposed to
this catalogue-level summary row) remains a stated [GAP].
"""

from __future__ import annotations

from ._vizier import VizierConeConnector, vizier_cone_rows

VIZIER_CATALOG = "IX/58"

DEFAULT_RELEASE = "sxps2"


class SwiftConnector(VizierConeConnector):
    name = "Swift"
    capabilities = ("catalogue", "xray_metadata")
    resolution_arcsec = 2.5
    # New X-ray surveys launch opt-in, matching Chandra/SDSS/Pan-STARRS
    # precedent, until the provider contract (endpoint, auth, rate limits)
    # is validated against the real service.
    enabled_by_default = False
    id_column = "IAUName"

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        # `release` here is a human-readable label ("sxps2"), not the VizieR
        # catalogue id -- see `_catalog` below and the module docstring.
        super().__init__(release)

    def _catalog(self) -> str:
        return VIZIER_CATALOG

    def extra_fields(self, row: dict) -> dict:
        return {"count_rate": row.get("CR0"),
                "hr1": row.get("HR1"), "hr2": row.get("HR2")}


def query_hardness_ratios(ra_deg: float, dec_deg: float, radius_arcsec: float, limit: int = 100
                          ) -> list[dict]:
    """Real 2SXPS total count rate and pre-computed hardness ratios near a
    position, via the same VizieR `IX/58` fetch `cone_search` uses.
    """
    top = max(1, min(int(limit), 200))
    rows = vizier_cone_rows(VIZIER_CATALOG, ra_deg, dec_deg, radius_arcsec, top)
    results: list[dict] = []
    for row in rows:
        try:
            object_id = str(row["IAUName"])
            ra_val, dec_val = float(row["RAJ2000"]), float(row["DEJ2000"])
        except (KeyError, TypeError, ValueError):
            continue
        results.append({
            "object_id": object_id, "ra_deg": ra_val, "dec_deg": dec_val,
            "count_rate": row.get("CR0"), "hr1": row.get("HR1"), "hr2": row.get("HR2"),
        })
    return results
