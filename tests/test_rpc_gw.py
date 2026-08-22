"""gw.events / gw.enrich: RPC surface for GW coincidence evidence.

Monkeypatches `gw`'s own module-level functions rather than the network
client underneath them -- the network-facing behaviour (event catalog
fetch, skymap construction from posterior samples) is exercised directly in
tests/test_gw.py against real, synthetic, and (once, during planning) live
data; these tests only need to confirm the RPC layer wires parameters
through correctly.
"""

from __future__ import annotations

from astra import candidates as candidates_mod
from astra import gw, rpc


class TestGwEvents:
    def test_lists_events_from_the_configured_catalog(self, monkeypatch):
        monkeypatch.setattr(
            gw, "fetch_event_catalog",
            lambda catalog, refresh=False, offline=False, root=None: [
                gw.GwEvent(name="GW1", catalog=catalog, gps_time=1000.0),
            ],
        )

        response = rpc.dispatch({"id": 1, "method": "gw.events",
                                 "params": {"catalog": "GWTC-2-confident"}})

        assert response["ok"] is True
        assert response["result"]["catalog"] == "GWTC-2-confident"
        assert response["result"]["events"] == [
            {"name": "GW1", "catalog": "GWTC-2-confident", "gps_time": 1000.0},
        ]

    def test_defaults_to_the_module_default_catalog(self, monkeypatch):
        captured = {}

        def fake(catalog, refresh=False, offline=False, root=None):
            captured["catalog"] = catalog
            return []

        monkeypatch.setattr(gw, "fetch_event_catalog", fake)
        rpc.dispatch({"id": 1, "method": "gw.events", "params": {}})

        assert captured["catalog"] == gw.DEFAULT_CATALOG


class TestGwEnrich:
    def test_forwards_parameters_to_enrich_candidates_gw(self, monkeypatch):
        captured = {}

        def fake(name, *, catalog, window_days, refresh, offline, root):
            captured.update(name=name, catalog=catalog, window_days=window_days,
                            refresh=refresh, offline=offline)
            return {"catalog": catalog, "events_checked": 0, "candidates": 0,
                   "counts": {"match": 0, "no_match": 0, "unavailable": 0}}

        monkeypatch.setattr(gw, "enrich_candidates_gw", fake)

        response = rpc.dispatch({"id": 1, "method": "gw.enrich", "params": {
            "name": "run1", "catalog": "GWTC-3-confident", "window_days": 7.0,
            "refresh": True, "offline": False,
        }})

        assert response["ok"] is True
        assert captured == {"name": "run1", "catalog": "GWTC-3-confident",
                            "window_days": 7.0, "refresh": True, "offline": False}

    def test_never_reports_a_score_delta(self, monkeypatch, isolated_root):
        """The RPC layer must not expose any score-changing behaviour either."""
        monkeypatch.setattr(candidates_mod, "load", lambda name, root=None: [
            candidates_mod.Candidate(
                candidate_id="c1", object_id="o1", survey="ZTF", band="g",
                ra_deg=180.0, dec_deg=10.0, path="does/not/matter.parquet",
                score={"total": 0.5}),
        ])
        monkeypatch.setattr(candidates_mod, "save", lambda built, name, root=None: None)
        monkeypatch.setattr(gw, "fetch_event_catalog",
                            lambda catalog, refresh=False, offline=False, root=None: [])

        response = rpc.dispatch({"id": 1, "method": "gw.enrich", "params": {}})

        assert response["ok"] is True
        assert response["result"]["candidates"] == 1
        assert "score" not in response["result"]
