"""GALEX UV catalogue connector via VizieR's Simple Cone Search (roadmap
item 26, panchromatic energy-balance SED).

`sed.py` already covers Gaia/ZTF/TESS optical bands; the panchromatic SED
work needs UV and IR anchors it does not have. VizieR catalogue
`II/335/galex_ais` ("Revised catalog of GALEX UV sources (GUVcat_AIS
GR6+7)", Bianchi+ 2017) was confirmed live this session via the same
`vizier.cds.unistra.fr` Simple Cone Search endpoint `vlass.py` already
uses: a real cone search around RA=180.0, Dec=0.0 returned real
`Name`/`RAJ2000`/`DEJ2000`/`FUV`/`e_FUV`/`NUV`/`e_NUV` rows. This is a
static-catalogue survey, not a time-domain one, so `fetch_light_curves`
follows the same metadata-only shape as `panstarrs.py`/`des.py`.
"""

from __future__ import annotations

from ._vizier import VizierConeConnector

DEFAULT_CATALOG = "II/335/galex_ais"


class GALEXConnector(VizierConeConnector):
    name = "GALEX"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 5.0  # GALEX PSF FWHM is a few arcsec; conservative
    enabled_by_default = False
    id_column = "Name"

    def __init__(self, release: str = DEFAULT_CATALOG) -> None:
        super().__init__(release)

    def extra_fields(self, row: dict) -> dict:
        return {"fuv_mag": row.get("FUV"), "fuv_mag_error": row.get("e_FUV"),
                "nuv_mag": row.get("NUV"), "nuv_mag_error": row.get("e_NUV")}
