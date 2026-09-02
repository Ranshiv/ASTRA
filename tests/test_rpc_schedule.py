"""schedule.build_night / schedule.replan RPC handlers (Direction 1,
"closed-loop decision-theoretic scheduling")."""

from __future__ import annotations

from astra import rpc

START = "2026-12-01T00:00:00Z"
VISIBLE_A = {"candidate_id": "a", "ra_deg": 180.0, "dec_deg": 80.0, "tail_probability": 0.5}
VISIBLE_B = {"candidate_id": "b", "ra_deg": 0.0, "dec_deg": 80.0, "tail_probability": 0.5}


def test_build_night_schedules_a_visible_candidate():
    response = rpc.dispatch({"id": 1, "method": "schedule.build_night", "params": {
        "candidates": [VISIBLE_A], "start_utc": START, "duration_hours": 12.0,
        "exposure_hours": 0.5,
    }})
    assert response["ok"] is True
    assert len(response["result"]["observations"]) == 1
    assert response["result"]["observations"][0]["candidate_id"] == "a"


def test_build_night_reports_unscheduled_candidates():
    hidden = {"candidate_id": "hidden", "ra_deg": 180.0, "dec_deg": -85.0,
             "tail_probability": 0.5}
    response = rpc.dispatch({"id": 2, "method": "schedule.build_night", "params": {
        "candidates": [hidden], "start_utc": START, "duration_hours": 12.0,
    }})
    assert response["ok"] is True
    assert response["result"]["unscheduled_candidate_ids"] == ["hidden"]


def test_replan_preserves_executed_observations_and_schedules_new_ones():
    built = rpc.dispatch({"id": 3, "method": "schedule.build_night", "params": {
        "candidates": [VISIBLE_A], "start_utc": START, "duration_hours": 12.0,
        "exposure_hours": 0.5,
    }})
    assert built["ok"] is True

    replanned = rpc.dispatch({"id": 4, "method": "schedule.replan", "params": {
        "schedule": built["result"], "executed_candidate_ids": ["a"],
        "remaining_candidates": [VISIBLE_B], "from_utc": "2026-12-01T01:00:00Z",
    }})
    assert replanned["ok"] is True
    ids = {o["candidate_id"] for o in replanned["result"]["observations"]}
    assert "a" in ids
    assert "b" in ids


def test_build_night_rejects_missing_candidates_param():
    response = rpc.dispatch({"id": 5, "method": "schedule.build_night", "params": {
        "start_utc": START,
    }})
    assert response["ok"] is False
