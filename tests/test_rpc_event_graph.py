"""events.graph.correlate / events.graph.calibrate: RPC surface for the
cross-messenger event-to-event graph (association.py).

Monkeypatches association's own module-level functions -- the statistical
model itself is exercised directly in tests/test_association.py; these tests
only confirm the RPC layer wires parameters through correctly and reads from
the same deduplicated event view associate_candidates already uses.
"""

from __future__ import annotations

from astra import association, rpc


class TestEventGraphCorrelate:
    def test_forwards_parameters_and_wraps_pairs(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(association, "fetch_latest_events",
                            lambda **kwargs: [{"event_id": "a"}, {"event_id": "b"}])

        def fake_correlate(events_list, *, window_days, background_window_days):
            captured.update(events=events_list, window_days=window_days,
                            background_window_days=background_window_days)
            return [{"event_a": "a", "event_b": "b", "log_bayes_factor": 3.0}]

        monkeypatch.setattr(association, "event_to_event_correlation", fake_correlate)

        response = rpc.dispatch({"id": 1, "method": "events.graph.correlate",
                                 "params": {"window_days": 5.0, "background_window_days": 100.0}})

        assert response["ok"] is True
        assert response["result"]["events_checked"] == 2
        assert response["result"]["pairs"][0]["log_bayes_factor"] == 3.0
        assert captured["window_days"] == 5.0
        assert captured["background_window_days"] == 100.0


class TestEventGraphCalibrate:
    def test_forwards_parameters_to_calibrate_event_graph(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(association, "fetch_latest_events", lambda **kwargs: [])

        def fake_calibrate(events_list, *, window_days, background_window_days, n_trials, seed):
            captured.update(window_days=window_days, background_window_days=background_window_days,
                            n_trials=n_trials, seed=seed)
            return {"observed_pairs": 0, "calibration": {"estimated_fdr": None}}

        monkeypatch.setattr(association, "calibrate_event_graph", fake_calibrate)

        response = rpc.dispatch({"id": 1, "method": "events.graph.calibrate",
                                 "params": {"n_trials": 50, "seed": 7}})

        assert response["ok"] is True
        assert captured["n_trials"] == 50
        assert captured["seed"] == 7
