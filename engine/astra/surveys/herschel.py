"""Herschel/PACS far-IR catalogue connector via VizieR's Simple Cone Search
(roadmap item 26, panchromatic energy-balance SED).

VizieR catalogue `VIII/106` ("Herschel/PACS Point Source Catalogs",
Herschel team 2017) confirmed live this session, same SCS pattern as
`galex.py`/`twomass.py`/`wise.py`: a real cone search around RA=180.0,
Dec=0.0 returned real `Name`/`Band`/`RAJ2000`/`DEJ2000`/`Flux`/`snr` rows
(flux in mJy, `snr` the signal-to-noise ratio -- there is no separate
tabulated error column, so `flux_error_mjy` is derived as `Flux / snr`
when `snr` is positive). PACS observed at 70/100/160um in different
pointed programmes rather than one uniform all-sky pass, so a given cone
may return zero, one, or several band rows for the same physical source;
`extra["band"]` carries which wavelength each row is. Pointed/targeted
coverage (not all-sky), metadata-only `fetch_light_curves`.
"""

from __future__ import annotations

from ._vizier import VizierConeConnector

DEFAULT_CATALOG = "VIII/106"


def _flux_error_mjy(flux: object, snr: object) -> float | None:
    try:
        flux_value, snr_value = float(flux), float(snr)
    except (TypeError, ValueError):
        return None
    if snr_value <= 0:
        return None
    return abs(flux_value) / snr_value


class HerschelConnector(VizierConeConnector):
    name = "Herschel"
    capabilities = ("catalogue", "mean_photometry")
    resolution_arcsec = 6.0  # PACS 70/100um beam; coarser than WISE, finer than SPIRE
    enabled_by_default = False
    id_column = "Name"

    def __init__(self, release: str = DEFAULT_CATALOG) -> None:
        super().__init__(release)

    def extra_fields(self, row: dict) -> dict:
        return {"band": row.get("Band"), "flux_mjy": row.get("Flux"),
                "flux_error_mjy": _flux_error_mjy(row.get("Flux"), row.get("snr")),
                "snr": row.get("snr")}
