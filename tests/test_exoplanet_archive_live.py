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
