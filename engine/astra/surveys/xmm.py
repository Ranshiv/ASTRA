"""XMM-Newton Serendipitous Source Catalog (4XMM) metadata connector.

Metadata-only, matching the Chandra/Swift/SDSS/Pan-STARRS shape: this
connector discovers X-ray counterparts from the 4XMM catalogue and their
basic properties, and deliberately returns no light curves. XMM's own
per-source time-series products exist but extracting and normalising them
is a separate, bounded job -- not something this adapter builds implicitly.

`query_hardness_ratios`, added for roadmap item 23, is a new additive
function -- `cone_search` above is unchanged. It targets VizieR (CDS)
rather than this module's own `CONE_URL`, mirroring `chandra.py`'s
`query_band_fluxes`: 4XMM-DR13 is real and live on VizieR as `IX/69`
(Webb et al. 2023), confirmed this session with a real cone search
returning `Flux8` (0.2-12 keV total flux) and 4XMM's own pre-computed
`HR1`-`HR4` hardness ratios (adjacent-band ratios across the catalogue's
five standard energy bands) -- what `xray_hardness_eval.py`'s flux/
hardness calibration check compares against. Individual per-band fluxes
were NOT found in this VizieR mirror's default column set (only the
pre-computed HRs and total flux); querying them, if needed, would require
either XMM's own SSC TAP service or an explicit VizieR column request,
neither attempted this session -- a real, stated [GAP].

Also confirmed live this session: XMM, like Chandra, is a POINTED
observatory with patchy sky coverage (unlike VLASS/eROSITA); and VizieR's
`4XMM` column holds only the coordinate-designation SUFFIX (e.g.
"J123049.2+122330"), not a "4XMM "-prefixed full name -- a real naming
assumption an early version of this session's own test got wrong, caught
by the live check and corrected there rather than left unnoticed.
"""

from __future__ import annotations

from .. import netclient
from ..tap import parse_votable
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

VIZIER_SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
VIZIER_CATALOG = "IX/69"

DEFAULT_RELEASE = "4xmm-dr13"
CONE_URL = "https://xmmssc-www.star.le.ac.uk/newpages/xsa_cone"


def parse_rows(payload: object, limit: int = 100) -> list[dict]:
    """The cone endpoint returns a list of row dicts; keep only real rows."""
    if not isinstance(payload, list):
        return []
    return [row for row in payload[:limit] if isinstance(row, dict)]


class XMMConnector(SurveyConnector):
    name = "XMM-Newton"
    capabilities = ("catalogue", "xray_metadata")
    resolution_arcsec = 6.0
    # New X-ray surveys launch opt-in, matching Chandra/Swift/SDSS/Pan-STARRS
    # precedent, until the provider contract (endpoint, auth, rate limits)
    # is validated against the real service.
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
            timeout=60, provider="xmm",
        )
        try:
            rows = parse_rows(response.json(), top)
        except ValueError:
            rows = []
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row.get("SRCID") or row["iauname"])
                ra_deg = float(row.get("SC_RA") if row.get("SC_RA") is not None else row["ra_deg"])
                dec_deg = float(row.get("SC_DEC") if row.get("SC_DEC") is not None else row["dec_deg"])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id,
                ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"obs_id": row.get("OBS_ID"),
                       "ep_flux": row.get("SC_EP_8_FLUX"),
                       "n_detections": row.get("N_DETECTIONS")},
            ))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # 4XMM per-source time-series extraction is a separate bounded job,
        # same rationale as chandra.py / swift.py: this connector is
        # discovery and catalogue metadata only.
        return []


def query_hardness_ratios(ra_deg: float, dec_deg: float, radius_arcsec: float, limit: int = 100
                          ) -> list[dict]:
    """Real 4XMM-DR13 total flux and pre-computed hardness ratios near a
    position, via VizieR `IX/69` -- see this module's docstring.
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
