"""frb.events / frb.enrich: RPC surface for FRB coincidence evidence.

Monkeypatches `frb`'s own module-level functions rather than the network
client underneath -- the CADC-facing behaviour is exercised directly (and
against synthetic data) in tests/test_frb.py; these tests only need to
confirm the RPC layer wires parameters through correctly.
"""

from __future__ import annotations

from astra import candidates as candidates_mod
from astra import frb, rpc


class TestFrbEvents:
    def test_lists_bursts_from_the_catalog(self, monkeypatch):
        monkeypatch.setattr(
            frb, "fetch_burst_catalog",
            lambda refresh=False, offline=False, root=None: [
                frb.FrbBurst(tns_name="FRB1", repeater_name="", ra_deg=180.0,
                            ra_err_deg=0.05, dec_deg=10.0, dec_err_deg=0.03,
                            mjd_400=58800.0),
            ],
        )

        response = rpc.dispatch({"id": 1, "method": "frb.events", "params": {}})

        assert response["ok"] is True
        assert response["result"]["bursts"][0]["tns_name"] == "FRB1"

    def test_forwards_refresh_and_offline(self, monkeypatch):
        captured = {}

        def fake(refresh=False, offline=False, root=None):
            captured.update(refresh=refresh, offline=offline)
            return []

        monkeypatch.setattr(frb, "fetch_burst_catalog", fake)
        rpc.dispatch({"id": 1, "method": "frb.events",
                     "params": {"refresh": True, "offline": True}})

        assert captured == {"refresh": True, "offline": True}


class TestFrbEnrich:
    def test_forwards_parameters_to_enrich_candidates_frb(self, monkeypatch):
        captured = {}

        def fake(name, *, window_days, sigma_threshold, refresh, offline, root):
            captured.update(name=name, window_days=window_days,
                            sigma_threshold=sigma_threshold, refresh=refresh,
                            offline=offline)
            return {"bursts_checked": 0, "candidates": 0,
                   "counts": {"match": 0, "no_match": 0, "unavailable": 0}}

        monkeypatch.setattr(frb, "enrich_candidates_frb", fake)

        response = rpc.dispatch({"id": 1, "method": "frb.enrich", "params": {
            "name": "run1", "window_days": 2.0, "sigma_threshold": 5.0,
            "refresh": True, "offline": False,
        }})

        assert response["ok"] is True
        assert captured == {"name": "run1", "window_days": 2.0,
                            "sigma_threshold": 5.0, "refresh": True, "offline": False}

    def test_never_reports_a_score_delta(self, monkeypatch, isolated_root):
        """The RPC layer must not expose any score-changing behaviour either."""
        monkeypatch.setattr(candidates_mod, "load", lambda name, root=None: [
            candidates_mod.Candidate(
                candidate_id="c1", object_id="o1", survey="ZTF", band="g",
                ra_deg=180.0, dec_deg=10.0, path="does/not/matter.parquet",
                score={"total": 0.5}),
        ])
        monkeypatch.setattr(candidates_mod, "save", lambda built, name, root=None: None)
        monkeypatch.setattr(frb, "fetch_burst_catalog",
                            lambda refresh=False, offline=False, root=None: [])

        response = rpc.dispatch({"id": 1, "method": "frb.enrich", "params": {}})

        assert response["ok"] is True
        assert response["result"]["candidates"] == 1
        assert "score" not in response["result"]
