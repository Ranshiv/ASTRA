"""Live, on-demand contract checks for the Gaia DR3 EB/astrophysical-parameters
queries (`surveys/gaia.py`) and the VizieR EB mass/radius catalog
(`eclipsing_binary_dimensions.py`).

Not run by pytest's default collection (see the `live` marker registered in
`tests/conftest.py`). Run explicitly with:

    pytest tests/test_eclipsing_binary_dimensions_live.py -m live

The real column names these functions assume (`frequency`,
`derived_primary_ecl_depth`, `radius_gspphot`, VizieR `J/ApJ/709/535`'s
`Name`/`Mass`/`Rad`/`Teff`) were confirmed live once while building this
module (2026-08-24), via `tap_schema.columns` and real queries against a
known Gaia eclipsing binary and the real published position of V760 Sco.
This file turns that one-off confirmation into a runnable check.
"""

from __future__ import annotations

import pytest

from astra import eclipsing_binary_dimensions as ebd
from astra.surveys import gaia

pytestmark = pytest.mark.live

# A real Gaia DR3 source with a vari_eclipsing_binary solution, confirmed
# live while building this module.
_KNOWN_EB_SOURCE_ID = 5277210824652602624
# V760 Sco's real, published position (VizieR J/ApJ/709/535).
_V760_SCO_RA_DEG, _V760_SCO_DEC_DEG = 246.18216, -34.89375


def test_query_eclipsing_binary_returns_real_gaia_solution():
    result = gaia.query_eclipsing_binary(_KNOWN_EB_SOURCE_ID)

    assert result is not None, (
        "query_eclipsing_binary returned None for a source known (this session) to "
        "have a vari_eclipsing_binary row -- either the service is down or the "
        "documented column names have drifted; check both before assuming this is "
        "a flaky network failure")
    assert result["period_days"] > 0
    assert 0.0 <= result["primary_eclipse_depth"] <= 1.0


def test_query_astrophysical_parameters_returns_real_gaia_estimate():
    result = gaia.query_astrophysical_parameters(_KNOWN_EB_SOURCE_ID)

    assert result is not None
    assert result["radius_gspphot"] is None or result["radius_gspphot"] > 0


def test_query_component_catalog_returns_real_v760_sco_row():
    rows = ebd.query_component_catalog(_V760_SCO_RA_DEG, _V760_SCO_DEC_DEG, radius_arcsec=10.0)

    assert len(rows) > 0, (
        "query_component_catalog returned zero rows for V760 Sco's real published "
        "position against VizieR J/ApJ/709/535 -- either the service is down, or "
        "the catalog ID/column names have drifted; check both before assuming "
        "this is a flaky network failure")
    assert any("V760 Sco" in row["name"] for row in rows)
    matched = next(row for row in rows if "V760 Sco" in row["name"])
    assert matched["mass_solar"] > 0
    assert matched["radius_solar"] > 0
