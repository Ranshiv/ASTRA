"""Deterministic active-review and follow-up planning helpers."""

from __future__ import annotations

from astra import followup, review


def test_review_queue_explains_uncertainty_and_is_deterministic():
    rows = [
        {"candidate_id": "a", "score": {"model_agreement": 1},
         "artifact": {"likelihood": 0.5}, "significance": {"tail_probability": 0.5},
         "features": {"amplitude": 0.1}},
        {"candidate_id": "b", "score": {"model_agreement": 4},
         "artifact": {"likelihood": 0.1}, "significance": {"tail_probability": 0.01},
         "features": {"amplitude": 0.9}},
    ]
    first = review.select_next(rows, limit=2)
    second = review.select_next(rows, limit=2)
    assert first == second
    assert first[0]["candidate_id"] == "a"
    assert first[0]["reasons"]


def test_followup_plan_is_draft_only_and_has_windows():
    result = followup.plan(ra_deg=180.0, dec_deg=22.0,
                           start_utc="2026-08-20T00:00:00Z",
                           duration_hours=24, cadence_minutes=30)
    assert result["schema_version"] == followup.SCHEMA_VERSION
    assert result["mode"] == "draft_only"
    assert "No observation request was submitted." in result["caveats"]
    assert isinstance(result["windows"], list)


def test_followup_plan_rejects_invalid_coordinates():
    try:
        followup.plan(ra_deg=400, dec_deg=0)
    except ValueError as exc:
        assert "coordinates" in str(exc)
    else:
        raise AssertionError("invalid coordinates were accepted")


def test_followup_constraints_report_rejected_slots_and_moon_fields():
    result = followup.plan(
        ra_deg=180.0, dec_deg=22.0,
        start_utc="2026-08-20T00:00:00Z", duration_hours=4, cadence_minutes=30,
        min_altitude_deg=10, twilight_sun_altitude_deg=-6,
        min_moon_separation_deg=20, max_moon_illumination=0.95,
        max_airmass=3.0,
        facility_name="test-observatory",
        facility_constraints={"max_airmass": 2.5},
    )
    assert result["schema_version"] == 2
    assert result["constraints"]["facility"]["name"] == "test-observatory"
    assert result["constraints"]["max_airmass"] == 2.5
    assert "twilight" in result["rejected_slots"]
    permissive = followup.plan(
        ra_deg=180.0, dec_deg=22.0,
        start_utc="2026-08-20T00:00:00Z", duration_hours=4, cadence_minutes=30,
        min_altitude_deg=0, twilight_sun_altitude_deg=10,
    )
    assert permissive["samples"]
    assert "moon_separation_deg" in permissive["samples"][0]
