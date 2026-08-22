"""Explicit bounded alert polling into the event inbox."""

from astra import alerts, events, netclient


def test_poll_ingests_packets_and_persists_cursor(tmp_path):
    result = alerts.poll(
        "gcn", root=tmp_path,
        payload={"next_cursor": "c2", "alerts": [
            {"event_id": "e1", "packet_id": "p1", "event_time": "2026-08-21T00:00:00Z",
             "localization": {"ra_deg": 10, "dec_deg": 20}},
        ]},
    )
    assert result["state"] == "ok"
    assert result["ingested"] == 1
    assert result["cursor"] == "c2"
    assert events.list_events(root=tmp_path)[0]["event_id"] == "e1"
    assert alerts.status(tmp_path)["cursors"][0]["cursor"] == "c2"


def test_poll_keeps_malformed_packets_isolated(tmp_path):
    result = alerts.poll("fink", root=tmp_path, payload={"items": [None, {"event_id": "e2"}]})
    assert result["state"] == "partial"
    assert result["ingested"] == 1
    assert len(result["errors"]) == 1


def test_offline_poll_does_not_contact_provider(tmp_path):
    result = alerts.poll("alerce", root=tmp_path, offline=True)
    assert result["state"] == "offline"
    assert result["packets"] == []


def test_missing_packet_ids_are_stable_when_feed_order_changes(tmp_path):
    first = alerts.poll("gcn", root=tmp_path, payload={"items": [
        {"event_id": "a", "event_time": "2026-08-21T00:00:00Z"},
        {"event_id": "b", "event_time": "2026-08-21T00:01:00Z"},
    ]})
    second = alerts.poll("gcn", root=tmp_path, payload={"items": [
        {"event_id": "b", "event_time": "2026-08-21T00:01:00Z"},
        {"event_id": "a", "event_time": "2026-08-21T00:00:00Z"},
    ]})

    first_ids = {packet["event_id"]: packet["packet_id"] for packet in first["packets"]}
    second_ids = {packet["event_id"]: packet["packet_id"] for packet in second["packets"]}
    assert first_ids == second_ids
    assert len(events.list_events(root=tmp_path, packets=True)) == 2
    assert second["new_packets"] == 0
    assert alerts.status(tmp_path)["cursors"][0]["packet_count"] == 2


class _FakeResponse:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return self._payload


def test_params_override_is_forwarded_to_the_request(tmp_path, monkeypatch):
    captured = {}

    def fake_get(url, params, timeout, provider):
        captured["params"] = params
        return _FakeResponse({"alerts": []})

    monkeypatch.setattr(netclient, "get", fake_get)

    alerts.poll("alerce", root=tmp_path, params={"survey": "lsst"})

    assert captured["params"]["survey"] == "lsst"
    assert captured["params"]["limit"] == 100


def test_no_params_override_leaves_the_request_unchanged(tmp_path, monkeypatch):
    captured = {}

    def fake_get(url, params, timeout, provider):
        captured["params"] = params
        return _FakeResponse({"alerts": []})

    monkeypatch.setattr(netclient, "get", fake_get)

    alerts.poll("alerce", root=tmp_path)

    assert "survey" not in captured["params"]


def test_latency_summary_reports_median_from_event_time_to_ingestion(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "_now", lambda: "2026-08-21T00:05:00+00:00")

    result = alerts.poll("gcn", root=tmp_path, payload={"alerts": [
        {"event_id": "e1", "packet_id": "p1", "event_time": "2026-08-21T00:00:00Z"},
        {"event_id": "e2", "packet_id": "p2", "event_time": "2026-08-21T00:04:00Z"},
    ]})

    assert result["latency_summary"] is not None
    assert result["latency_summary"]["n"] == 2
    # 300s and 60s deltas -> median 180s.
    assert result["latency_summary"]["median"] == 180.0


def test_latency_summary_is_none_when_no_packet_has_a_usable_event_time(tmp_path):
    result = alerts.poll("gcn", root=tmp_path, payload={"alerts": [
        {"event_id": "e1", "packet_id": "p1"},
    ]})

    assert result["latency_summary"] is None


def test_duplicate_rate_is_zero_on_a_first_poll_and_one_on_a_repeat(tmp_path):
    payload = {"alerts": [{"event_id": "e1", "packet_id": "p1",
                           "event_time": "2026-08-21T00:00:00Z"}]}

    first = alerts.poll("gcn", root=tmp_path, payload=payload)
    second = alerts.poll("gcn", root=tmp_path, payload=payload)

    assert first["duplicate_rate"] == 0.0
    assert second["duplicate_rate"] == 1.0


def test_duplicate_rate_is_none_when_nothing_was_ingested(tmp_path):
    result = alerts.poll("gcn", root=tmp_path, payload={"alerts": []})
    assert result["ingested"] == 0
    assert result["duplicate_rate"] is None
