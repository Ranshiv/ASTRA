"""AllWISE mid-IR catalogue connector via VizieR's Simple Cone Search
(roadmap item 26, panchromatic energy-balance SED).

VizieR catalogue `II/328/allwise` ("AllWISE Data Release", Cutri+ 2013)
confirmed live this session, same SCS pattern as `galex.py`/`twomass.py`:
a real cone search around RA=180.0, Dec=0.0 returned real
`AllWISE`/`RAJ2000`/`DEJ2000`/`W1mag`/`e_W1mag`/.../`W4mag`/`e_W4mag`
rows. Static point-source catalogue, metadata-only `fetch_light_curves`.
"""

from __future__ import annotations

from ._vizier import VizierConeConnector

DEFAULT_CATALOG = "II/328/allwise"


class WISEConnector(VizierConeConnector):
    name = "WISE"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 6.0
    enabled_by_default = False
    id_column = "AllWISE"

    def __init__(self, release: str = DEFAULT_CATALOG) -> None:
        super().__init__(release)

    def extra_fields(self, row: dict) -> dict:
        return {"w1_mag": row.get("W1mag"), "w1_mag_error": row.get("e_W1mag"),
                "w2_mag": row.get("W2mag"), "w2_mag_error": row.get("e_W2mag"),
                "w3_mag": row.get("W3mag"), "w3_mag_error": row.get("e_W3mag"),
                "w4_mag": row.get("W4mag"), "w4_mag_error": row.get("e_W4mag")}
