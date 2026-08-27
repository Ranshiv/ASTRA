"""eROSITA eRASS1 X-ray source catalog connector (roadmap item 23).

Metadata-only, matching the Chandra/Swift/XMM shape. Reached via VizieR
(CDS), the same pivot `vlass.py`/`chandra.query_band_fluxes` already made
this session away from chasing a survey's own bespoke infrastructure:
`J/A+A/682/A34` is the real "SRG/eROSITA all-sky survey catalogs (eRASS1)"
(Merloni et al. 2024), confirmed live this session with a real cone search
returning real `IAUName`/`RA_ICRS`/`DE_ICRS`/`MLcts1`/`MLFlux1` rows (per-
band maximum-likelihood counts/flux; eRASS1's own band numbering was not
further decoded this session beyond confirming band 1 exists and is
real -- a stated [GAP], not guessed at).
"""

from __future__ import annotations

from .. import netclient
from ..tap import parse_votable
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

VIZIER_SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
DEFAULT_CATALOG = "J/A+A/682/A34"


class EROSITAConnector(SurveyConnector):
    name = "eROSITA"
    capabilities = ("catalogue", "xray_metadata")
    resolution_arcsec = 10.0  # eRASS1 typical positional accuracy scale
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_CATALOG) -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        response = netclient.get(
            VIZIER_SCS_URL,
            {"-source": self.release, "RA": query.ra_deg, "DEC": query.dec_deg,
             "SR": query.radius_arcsec / 3600.0, "-out.max": top},
            timeout=60, provider="vizier",
        )
        rows = parse_votable(response.text, top)
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row["IAUName"])
                ra_deg = float(row["RA_ICRS"])
                dec_deg = float(row["DE_ICRS"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"counts_band1": row.get("MLcts1"), "flux_band1": row.get("MLFlux1")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # eRASS1 is a single-epoch (first 6-month all-sky pass) catalogue
        # at this writing; per-epoch/per-scan extraction is a separate,
        # not-yet-built job, same rationale as chandra.py/swift.py's
        # metadata-only shape.
        return []
