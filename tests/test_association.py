"""Conservative event/candidate association tests."""

from astra import association, events


def test_fetch_latest_events_collapses_to_the_newest_revision_per_event(tmp_path):
    # received_utc set explicitly, distinct to the second: `_latest_packets`
    # compares this field as a plain string, and two ingests inside the
    # same wall-clock second (the harness's own `_now()` resolution) would
    # otherwise tie and make "which revision wins" depend on dict insertion
    # order rather than the intended recency comparison.
    events.ingest("generic", {"event_id": "E1", "event_time": "2026-01-01T00:00:00Z",
                              "localization": {"ra": 10.0, "dec": 5.0}},
                 root=tmp_path, packet_id="p1", packet_version="1",
                 received_utc="2026-01-01T00:00:00+00:00")
    events.ingest("generic", {"event_id": "E1", "event_time": "2026-01-01T01:00:00Z",
                              "localization": {"ra": 10.5, "dec": 5.0}},
                 root=tmp_path, packet_id="p1", packet_version="2",
                 received_utc="2026-01-01T00:00:05+00:00")
    events.ingest("generic", {"event_id": "E2", "event_time": "2026-01-02T00:00:00Z",
                              "localization": {"ra": 20.0, "dec": -5.0}},
                 root=tmp_path, packet_id="p2", packet_version="1",
                 received_utc="2026-01-01T00:00:00+00:00")

    latest = association.fetch_latest_events(root=tmp_path)

    by_id = {row["event_id"]: row for row in latest}
    assert set(by_id) == {"E1", "E2"}
    assert by_id["E1"]["localization"]["ra_deg"] == 10.5  # the newer revision won


def _candidate():
    return {
        "candidate_id": "c1", "ra_deg": 359.999, "dec_deg": 10.0,
        "features": {
            "time_start": "2026-08-20T00:00:00Z",
            "time_end": "2026-08-20T12:00:00Z",
        },
    }


def test_point_event_association_handles_ra_wrap_and_time_overlap():
    event = {
        "event_id": "e1", "provider": "generic",
        "event_time": "2026-08-20T06:00:00Z",
        "localization": {"ra_deg": 0.001, "dec_deg": 10.0, "error_radius_arcsec": 10},
    }
    result = association.associate_one(_candidate(), event, radius_arcsec=5, window_days=0)
    assert result["matched"]
    assert result["spatial"]["state"] == "point"
    assert result["temporal"]["matched"]


def test_unlocalized_or_temporally_unknown_event_is_not_a_match_by_default():
    event = {"event_id": "e2", "event_time": None, "localization": {}}
    result = association.associate_one(_candidate(), event)
    assert not result["matched"]
    assert result["spatial"]["state"] == "unlocalized"
    assert result["temporal"]["state"] == "event_time_unknown"
