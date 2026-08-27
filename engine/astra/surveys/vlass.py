"""VLASS radio-component catalog connector via VizieR's Simple Cone Search
(roadmap item 22).

The plan for this module assumed a CADC/CIRADA-hosted TAP endpoint,
mirroring `frb.py`'s CADC precedent. That was checked live this session and
did not pan out cleanly: CADC's own documented TAP endpoint
(`https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync`, confirmed live
and reachable) has no `vlass`-named table in its `TAP_SCHEMA`, and CIRADA's
own catalog-documentation PDF (fetched live, 11.6 MB, real) could not be
parsed with any text-extraction tool available in this environment. What
DID work, live-verified this session: VizieR (CDS) hosts the real "VLASS
QL Ep.1 Catalog, CIRADA version" (Gordon et al. 2021, ApJS 255, 30) under
its own catalogue identifier `J/ApJS/255/30/comp`, reachable through the
standard, public, credential-free IVOA Simple Cone Search (SCS) protocol --
a real cone search around RA=180.0, Dec=0.0 returned real
`CompName`/`RAJ2000`/`DEJ2000`/`Ftot`/`Fpeak`/`DupFlag`/`QualFlag` rows.
This is a more standard, stable access path than chasing CADC/CIRADA's own
infrastructure further, so this connector targets VizieR instead of CADC.

Response parsing reuses `tap.parse_votable` UNCHANGED (SCS returns the same
VOTable shape TAP's own `votable` format does) rather than writing a
second XML-table parser in this codebase.

This is VLASS EPOCH 1 (Quick Look) only. A VLASS Epoch 2 catalogue exists
("CIRADA releases VLASS Epoch2 Catalogues", per NRAO's own science-site
news) but its VizieR catalogue identifier was not found this session --
a real, stated `[GAP]`, not guessed at. Multi-epoch VLASS variability
(comparing a source's Epoch 1 vs. Epoch 2/3 flux) is therefore not
reachable through this connector yet.
"""

from __future__ import annotations

from .. import netclient
from ..tap import parse_votable
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
# VizieR's own catalogue identifier for "VLASS QL Ep.1 Catalog, CIRADA
# version" -- confirmed live this session via the SCS response's own
# <DESCRIPTION> field.
DEFAULT_CATALOG = "J/ApJS/255/30/comp"


class VLASSConnector(SurveyConnector):
    name = "VLASS"
    capabilities = ("catalogue", "radio_flux")
    resolution_arcsec = 2.5  # VLASS Quick Look synthesized beam, ~2.5"
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_CATALOG) -> None:
        # Named `release` (not `catalog`) to match every other connector's
        # `SurveyConnector.release` contract -- `describe()`'s base
        # implementation reads `self.release` unconditionally, a real bug
        # found via the full test suite (not caught by this module's own
        # tests, which never call `describe()`): a `self.catalog`-only
        # attribute raised `AttributeError` the moment this connector was
        # registered and `describe_all()` iterated over it.
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        response = netclient.get(
            SCS_URL,
            {"-source": self.release, "RA": query.ra_deg, "DEC": query.dec_deg,
             "SR": query.radius_arcsec / 3600.0, "-out.max": top},
            timeout=60, provider="vizier",
        )
        rows = parse_votable(response.text, top)
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row["CompName"])
                ra_deg = float(row["RAJ2000"])
                dec_deg = float(row["DEJ2000"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"flux_total_mjy": row.get("Ftot"), "flux_total_err_mjy": row.get("e_Ftot"),
                       "flux_peak_mjy": row.get("Fpeak"), "flux_peak_err_mjy": row.get("e_Fpeak"),
                       "dup_flag": row.get("DupFlag"), "qual_flag": row.get("QualFlag"),
                       "nvss_dist_arcsec": row.get("NVSSdist"),
                       "first_dist_arcsec": row.get("FIRSTdist"), "epoch": "1"},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # A single-epoch (QL Epoch 1) component list -- multi-epoch VLASS
        # flux extraction is a separate, not-yet-built job (see module
        # docstring's stated Epoch 2 gap), same rationale as
        # chandra.py/swift.py's metadata-only shape.
        return []


NVSS_CATALOG = "VIII/65/nvss"


def query_nvss_flux_1_4ghz(ra_deg: float, dec_deg: float, radius_arcsec: float = 15.0
                           ) -> dict | None:
    """The nearest real NVSS (1.4 GHz) source within `radius_arcsec`, for a
    genuine two-frequency spectral-index measurement against a VLASS
    (~3 GHz) detection -- confirmed live this session: `VIII/65/nvss` is a
    real VizieR catalogue with populated `S1.4`/`e_S1.4` columns (mJy).

    Returns `None` (not a fabricated zero flux) when no NVSS source falls
    within the search radius -- the common case for a source too faint for
    NVSS's shallower survey depth, not necessarily "not measured".
    """
    response = netclient.get(
        SCS_URL,
        {"-source": NVSS_CATALOG, "RA": ra_deg, "DEC": dec_deg,
         "SR": radius_arcsec / 3600.0, "-out.max": 1, "-out.orderby": "_r"},
        timeout=60, provider="vizier",
    )
    rows = parse_votable(response.text, 1)
    if not rows:
        return None
    row = rows[0]
    try:
        flux_mjy = float(row["S1.4"])
        flux_err_mjy = float(row["e_S1.4"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"frequency_ghz": 1.4, "flux_mjy": flux_mjy, "flux_err_mjy": flux_err_mjy,
           "nvss_name": row.get("NVSS")}
