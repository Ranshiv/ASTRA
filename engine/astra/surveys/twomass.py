"""2MASS near-IR catalogue connector via VizieR's Simple Cone Search
(roadmap item 26, panchromatic energy-balance SED).

VizieR catalogue `II/246/out` ("2MASS All-Sky Catalog of Point Sources",
Cutri+ 2003) confirmed live this session, same `vizier.cds.unistra.fr` SCS
endpoint `vlass.py`/`galex.py` already use: a real cone search around
RA=180.0, Dec=0.0 returned real `2MASS`/`RAJ2000`/`DEJ2000`/`Jmag`/
`e_Jmag` rows (and the analogous H/K columns). Static point-source
catalogue, so `fetch_light_curves` is the same metadata-only shape as
`panstarrs.py`.
"""

from __future__ import annotations

from ._vizier import VizierConeConnector

DEFAULT_CATALOG = "II/246/out"


class TwoMASSConnector(VizierConeConnector):
    name = "2MASS"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 2.0
    enabled_by_default = False
    id_column = "2MASS"

    def __init__(self, release: str = DEFAULT_CATALOG) -> None:
        super().__init__(release)

    def extra_fields(self, row: dict) -> dict:
        return {"j_mag": row.get("Jmag"), "j_mag_error": row.get("e_Jmag"),
                "h_mag": row.get("Hmag"), "h_mag_error": row.get("e_Hmag"),
                "k_mag": row.get("Kmag"), "k_mag_error": row.get("e_Kmag")}
