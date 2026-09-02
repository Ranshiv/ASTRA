"""Live, on-demand contract check for the NASA Exoplanet Archive TAP service.

Not run by pytest's default collection (see the `live` marker registered in
`tests/conftest.py`). Run explicitly with:

    pytest tests/test_exoplanet_archive_live.py -m live

`exoplanet_archive.py`'s `_PS_COLUMNS` (`pl_name`, `hostname`, `pl_orbper`,
`pl_trandur`, `pl_trandep`, `pl_rade`, `pl_tranmid`, ...) were written against
the Exoplanet Archive's published `ps` table documentation, not a live
authenticated fetch -- this file turns that documentation-only assumption
into a runnable check against a well-known confirmed planet (Kepler-10 b)
instead of a TODO comment.
"""

from __future__ import annotations

import pytest

from astra import exoplanet_archive as ea

pytestmark = pytest.mark.live


def test_query_confirmed_planets_returns_real_kepler10b_parameters(isolated_root):
    records = ea.query_confirmed_planets(host_name="Kepler-10", refresh=True)

    assert len(records) > 0, (
        "query_confirmed_planets returned zero rows for Kepler-10 against the "
        "live Exoplanet Archive TAP service -- either the service is down, or "
        "the documented ps-table column names in exoplanet_archive._PS_COLUMNS "
        "have drifted from the real schema; check both before assuming this "
        "is a flaky network failure")
    kepler10b = next((r for r in records if r.name.strip() == "Kepler-10 b"), records[0])
    assert kepler10b.period_days is not None and 0.5 < kepler10b.period_days < 1.5
    assert kepler10b.host_name == "Kepler-10"


def test_query_confirmed_planets_returns_habitability_columns(isolated_root):
    """`_PS_COLUMNS`'s ten stellar/insolation/mass columns added for
    `habitability.py` (roadmap: astrophysics & extraterrestrial-study
    feature pass) were written against the Exoplanet Archive's published
    `ps`-table column documentation, not a live fetch -- this is that
    documentation-only assumption turned into a runnable check, the same
    discipline the Kepler-10 b test above already applies to the original
    columns. Kepler-10 is a well-characterised solar-type host, so its
    stellar parameters should all be populated on the live service.
    """
    records = ea.query_confirmed_planets(host_name="Kepler-10", refresh=True)
    assert len(records) > 0
    kepler10b = next((r for r in records if r.name.strip() == "Kepler-10 b"), records[0])

    assert kepler10b.st_teff_k is not None and 5000.0 < kepler10b.st_teff_k < 6500.0, (
        "st_teff column missing or out of range against the live service -- "
        "check _PS_COLUMNS's 'st_teff' name against the real ps-table schema")
    assert kepler10b.st_radius_rsun is not None and kepler10b.st_radius_rsun > 0
    assert kepler10b.st_mass_msun is not None and kepler10b.st_mass_msun > 0
    assert kepler10b.distance_pc is not None and kepler10b.distance_pc > 0
    # pl_insol/pl_eqt/pl_orbsmax are frequently null for individual planets
    # even on well-characterised systems (they are derived quantities with
    # their own data-availability requirements) -- checked for presence in
    # the parsed record's schema, not asserted non-null.
    assert hasattr(kepler10b, "insolation_earth")
    assert hasattr(kepler10b, "eq_temp_k")
    assert hasattr(kepler10b, "semimajor_au")


def test_query_planets_bounded_returns_rows_within_teff_range(isolated_root):
    """`query_planets_bounded` (added alongside the habitability columns)
    builds a numeric-only WHERE clause -- this confirms it actually
    executes against the live service and returns rows whose `st_teff`
    truly falls inside the requested bound, not just that the request
    doesn't error."""
    records = ea.query_planets_bounded(teff_min=5700.0, teff_max=5900.0, max_rows=20, refresh=True)
    assert len(records) > 0, (
        "query_planets_bounded returned zero rows for a Sun-like Teff band "
        "against the live service -- either the service is down or the "
        "generated ADQL WHERE clause is malformed")
    for record in records:
        if record.st_teff_k is not None:
            assert 5700.0 <= record.st_teff_k <= 5900.0
