"""Event-native packet normalization, persistence, and replay tests."""

from __future__ import annotations

import hashlib
import json

from astra import events


def test_ingest_indexes_packet_and_cluster(isolated_root):
    payload = {
        "event_id": "S2026abc",
        "packet_id": "notice-1",
        "event_time": "2026-08-20T12:00:00Z",
        "localization": {"ra_deg": 180.1, "dec_deg": 22.4,
                          "error_radius_arcsec": 4.0},
        "classifications": [{"label": "SN", "probability": 0.72}],
    }
    packet = events.ingest("gcn", payload, root=isolated_root.root)

    assert packet["event_id"] == "S2026abc"
    assert packet["localization"]["type"] == "point"
    assert hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() == packet["raw_sha256"]
    assert events.list_events(root=isolated_root.root)[0]["packet_count"] == 1
    loaded = events.get_packet(packet["packet_key"], root=isolated_root.root,
                               include_raw=True)
    assert json.loads(loaded["raw"])["event_id"] == "S2026abc"


def test_reingest_same_packet_is_idempotent(isolated_root):
    payload = {"event_id": "evt", "packet_id": "p", "ra": 1, "dec": 2}
    first = events.ingest("generic", payload, root=isolated_root.root)
    second = events.ingest("generic", payload, root=isolated_root.root)

    assert first["packet_key"] == second["packet_key"]
    assert len(events.list_events(root=isolated_root.root)) == 1
    assert events.list_events(root=isolated_root.root, packets=True)[0]["packet_id"] == "p"


def test_packet_revision_replaces_index_without_losing_event_identity(isolated_root):
    first = events.ingest("gcn", {"event_id": "evt", "packet_id": "p",
                                   "packet_version": "1", "classification": "initial"},
                          root=isolated_root.root, packet_version="1")
    second = events.ingest("gcn", {"event_id": "evt", "packet_id": "p",
                                    "packet_version": "2", "classification": "revised"},
                           root=isolated_root.root, packet_version="2")
    assert first["packet_key"] != second["packet_key"]
    assert events.list_events(root=isolated_root.root)[0]["packet_count"] == 2


def test_voevent_xml_has_stable_minimum_fields(isolated_root):
    xml = """<VOEvent ivorn='ivo://example/test#1'>
      <Who><Date>2026-08-20T12:00:00Z</Date></Who>
      <WhereWhen><ObservationLocation><AstroCoords>
        <Position2D><Value2><C1>12.5</C1><C2>-4.25</C2></Value2></Position2D>
      </AstroCoords></ObservationLocation></WhereWhen>
    </VOEvent>"""
    packet = events.ingest("voevent", xml, root=isolated_root.root)
    assert packet["event_id"] == "ivo://example/test#1"
    assert packet["event_time"] == "2026-08-20T12:00:00Z"
    assert packet["localization"]["type"] == "point"


def test_oversized_raw_packet_is_rejected(isolated_root):
    try:
        events.ingest("generic", "x" * (events.MAX_RAW_BYTES + 1),
                      root=isolated_root.root)
    except ValueError as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError("oversized event packet was accepted")
