"""NEOWISE connector contract: multi-epoch grouping by allwise_cntr,
capabilities, no-op fetch."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from astropy.table import Table, MaskedColumn

from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.neowise import NEOWISEConnector


def _table(rows: list[dict], masked_cntr_indices: set[int] = frozenset()) -> Table:
    columns = {key: [row.get(key) for row in rows] for key in
              ("ra", "dec", "mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro")}
    table = Table(columns)
    cntr_values = [rows[i].get("allwise_cntr", "0") for i in range(len(rows))]
    mask = [i in masked_cntr_indices for i in range(len(rows))]
    table["allwise_cntr"] = MaskedColumn(cntr_values, mask=mask)
    return table


TWO_EPOCHS_ONE_SOURCE = _table([
    {"ra": 280.0005, "dec": -20.0, "mjd": 56748.9, "w1mpro": 13.9, "w1sigmpro": 0.06,
     "w2mpro": 14.1, "w2sigmpro": 0.21, "allwise_cntr": "2800019701351019431"},
    {"ra": 280.0004, "dec": -19.9999, "mjd": 60398.3, "w1mpro": 13.95, "w1sigmpro": 0.05,
     "w2mpro": 14.05, "w2sigmpro": 0.2, "allwise_cntr": "2800019701351019431"},
])


class TestNEOWISEConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = NEOWISEConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_groups_epochs_into_one_source(self, monkeypatch, cone: ConeQuery):
        from astroquery.ipac.irsa import Irsa
        monkeypatch.setattr(Irsa, "query_region", lambda **kw: TWO_EPOCHS_ONE_SOURCE)

        sources = NEOWISEConnector().cone_search(cone, limit=10)

        assert len(sources) == 1
        source = sources[0]
        assert source.survey == "NEOWISE"
        assert source.object_id == "2800019701351019431"
        assert source.extra["epoch_count"] == 2
        assert source.extra["mjd_first"] == pytest.approx(56748.9)
        assert source.extra["mjd_last"] == pytest.approx(60398.3)
        # The representative row is the latest epoch.
        assert source.ra_deg == pytest.approx(280.0004)
        assert source.extra["w1_mag"] == pytest.approx(13.95)

    def test_a_masked_allwise_cntr_is_excluded_not_invented_as_a_source(
            self, monkeypatch, cone: ConeQuery):
        from astroquery.ipac.irsa import Irsa
        table = _table([
            {"ra": 280.0, "dec": -20.0, "mjd": 1.0, "w1mpro": 14.0, "w1sigmpro": 0.1,
             "w2mpro": 14.2, "w2sigmpro": 0.1, "allwise_cntr": "--"},
        ], masked_cntr_indices={0})
        monkeypatch.setattr(Irsa, "query_region", lambda **kw: table)

        sources = NEOWISEConnector().cone_search(cone)
        assert sources == []

    def test_cone_search_handles_no_table(self, monkeypatch, cone: ConeQuery):
        from astroquery.ipac.irsa import Irsa
        monkeypatch.setattr(Irsa, "query_region", lambda **kw: None)
        assert NEOWISEConnector().cone_search(cone) == []

    def test_cone_search_handles_empty_table(self, monkeypatch, cone: ConeQuery):
        from astroquery.ipac.irsa import Irsa
        monkeypatch.setattr(Irsa, "query_region", lambda **kw: _table([]))
        assert NEOWISEConnector().cone_search(cone) == []

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="NEOWISE", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert NEOWISEConnector().fetch_light_curves(source) == []

    def test_describe_does_not_touch_the_network(self):
        description = NEOWISEConnector().describe()
        assert description["name"] == "NEOWISE"
        assert description["release"] == "neowiser_p1bs_psd"


@pytest.mark.live
class TestNEOWISELive:
    """Confirmed live this session (2026-09-01): IRSA's `neowiser_p1bs_psd`
    single-exposure source table answers a real cone search with real
    per-epoch ra/dec/mjd/w1mpro/w2mpro rows -- a 5 arcsec cone at RA=280.0,
    Dec=-20.0 returned 240 real detection rows collapsing to one real
    physical source (`allwise_cntr`) spanning 239 epochs from MJD 56748.9 to
    60398.3, roughly a 10-year real multi-epoch baseline. See neowise.py's
    module docstring for the full finding."""

    def test_cone_search_returns_a_real_multi_epoch_source(self):
        sources = NEOWISEConnector().cone_search(
            ConeQuery(ra_deg=280.0, dec_deg=-20.0, radius_arcsec=5.0), limit=10)
        assert len(sources) > 0
        assert all(source.survey == "NEOWISE" for source in sources)
        assert any(source.extra["epoch_count"] > 1 for source in sources)
