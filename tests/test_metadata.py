"""The SQLite metadata index: sources/labels/jobs/caches/events/alerts.

No test file existed for this module despite it being the persistence layer
behind nearly every other RPC handler; this covers each table's CRUD surface
and the invariants their own docstrings call out (idempotent upsert,
append-only votes/requests, resumable fetch cursor, TTL cache access-time
refresh).
"""

from __future__ import annotations

import json

import pytest

from astra import metadata


def _source(key="ztf:dr24:obj1", survey="ZTF", object_id="obj1", **extra) -> dict:
    return {"source_key": key, "survey": survey, "release": "dr24",
            "object_id": object_id, "ra_deg": 180.1, "dec_deg": 22.4,
            "extra": {"band": "g", **extra}}


class TestConnectAndSchema:
    def test_creates_the_database_file(self, tmp_path):
        metadata.connect(tmp_path).close()
        assert metadata.database_path(tmp_path).exists()

    def test_records_the_current_schema_version(self, tmp_path):
        with metadata.connect(tmp_path) as db:
            row = db.execute(
                "SELECT version FROM schema_migrations WHERE version=?",
                (metadata.SCHEMA_VERSION,)).fetchone()
        assert row is not None

    def test_reconnecting_does_not_lose_added_columns(self, tmp_path):
        """A DB created by an older schema (missing a later-added column)
        must be brought up to the current shape idempotently, per
        `_add_missing_columns`'s own docstring."""
        metadata.connect(tmp_path).close()
        # Second connect() must not raise even though the columns already exist.
        with metadata.connect(tmp_path) as db:
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
        assert "idempotency_key" in columns
        assert "cancel_requested" in columns


class TestSources:
    def test_upsert_then_list_round_trips(self, tmp_path):
        assert metadata.upsert_sources(tmp_path, [_source()]) == 1
        rows = metadata.list_sources(tmp_path)
        assert len(rows) == 1
        assert rows[0]["source_key"] == "ztf:dr24:obj1"
        assert rows[0]["extra"] == {"band": "g"}

    def test_upsert_with_no_sources_is_a_no_op(self, tmp_path):
        assert metadata.upsert_sources(tmp_path, []) == 0
        assert metadata.list_sources(tmp_path) == []

    def test_upsert_is_idempotent_on_source_key(self, tmp_path):
        metadata.upsert_sources(tmp_path, [_source()])
        metadata.upsert_sources(tmp_path, [_source(ra_deg=181.0)])
        rows = metadata.list_sources(tmp_path)
        assert len(rows) == 1

    def test_pending_sources_excludes_a_done_source(self, tmp_path):
        metadata.upsert_sources(tmp_path, [_source()])
        metadata.mark_source_fetched(tmp_path, "ztf:dr24:obj1", metadata.FETCH_DONE)
        assert metadata.pending_sources(tmp_path) == []

    def test_pending_sources_includes_a_never_attempted_source(self, tmp_path):
        metadata.upsert_sources(tmp_path, [_source()])
        pending = metadata.pending_sources(tmp_path)
        assert len(pending) == 1
        assert pending[0]["fetch_status"] is None

    def test_a_failed_source_is_retried_until_max_attempts(self, tmp_path):
        metadata.upsert_sources(tmp_path, [_source()])
        for _ in range(metadata.MAX_FETCH_ATTEMPTS):
            metadata.mark_source_fetched(tmp_path, "ztf:dr24:obj1",
                                         metadata.FETCH_FAILED, error="boom")
            pending = metadata.pending_sources(tmp_path)
        # After MAX_FETCH_ATTEMPTS failures, the source is no longer pending.
        assert pending == []

    def test_pending_sources_filters_by_survey(self, tmp_path):
        metadata.upsert_sources(tmp_path, [
            _source(key="ztf:1", survey="ZTF"),
            _source(key="gaia:1", survey="Gaia", object_id="g1"),
        ])
        assert len(metadata.pending_sources(tmp_path, survey="Gaia")) == 1

    def test_acquisition_progress_counts_by_state(self, tmp_path):
        metadata.upsert_sources(tmp_path, [
            _source(key="a", object_id="a"), _source(key="b", object_id="b"),
        ])
        metadata.mark_source_fetched(tmp_path, "a", metadata.FETCH_DONE)
        metadata.mark_source_fetched(tmp_path, "b", metadata.FETCH_FAILED, error="x")
        progress = metadata.acquisition_progress(tmp_path)
        assert progress["total"] == 2
        assert progress["done"] == 1
        assert progress["failed"] == 1
        assert progress["complete_fraction"] == pytest.approx(0.5)
        assert progress["recent_failures"][0]["object_id"] == "b"


class TestLabels:
    def test_put_then_read_back(self, tmp_path):
        metadata.put_label(tmp_path, "cand-1", "interesting", "looks real")
        rows = metadata.labels(tmp_path)
        assert rows["cand-1"]["label"] == "interesting"
        assert rows["cand-1"]["note"] == "looks real"

    def test_put_overwrites_a_prior_label_for_the_same_candidate(self, tmp_path):
        metadata.put_label(tmp_path, "cand-1", "interesting", "")
        metadata.put_label(tmp_path, "cand-1", "artifact", "changed my mind")
        rows = metadata.labels(tmp_path)
        assert len(rows) == 1
        assert rows["cand-1"]["label"] == "artifact"

    def test_move_label_renames_the_candidate_key(self, tmp_path):
        metadata.put_label(tmp_path, "old-key", "interesting", "")
        metadata.move_label(tmp_path, "old-key", "new-key")
        rows = metadata.labels(tmp_path)
        assert "new-key" in rows
        assert "old-key" not in rows

    def test_move_label_does_not_clobber_an_existing_destination(self, tmp_path):
        metadata.put_label(tmp_path, "old-key", "interesting", "")
        metadata.put_label(tmp_path, "new-key", "artifact", "")
        metadata.move_label(tmp_path, "old-key", "new-key")
        rows = metadata.labels(tmp_path)
        # The move is a no-op (guarded by NOT EXISTS): both keys survive.
        assert rows["new-key"]["label"] == "artifact"
        assert rows["old-key"]["label"] == "interesting"


class TestFollowupRequests:
    def test_request_then_result_round_trips(self, tmp_path):
        metadata.put_followup_request(tmp_path, "req-1", "cand-1", "Palomar", "urgent")
        updated = metadata.put_followup_result(tmp_path, "req-1", "observed", "clear skies")
        assert updated["status"] == "observed"
        assert updated["result_note"] == "clear skies"

    def test_result_for_an_unknown_request_raises_key_error(self, tmp_path):
        with pytest.raises(KeyError):
            metadata.put_followup_result(tmp_path, "does-not-exist", "observed", "")

    def test_history_is_append_only_per_candidate(self, tmp_path):
        metadata.put_followup_request(tmp_path, "req-1", "cand-1", "Palomar", "")
        metadata.put_followup_request(tmp_path, "req-2", "cand-1", "Keck", "")
        history = metadata.followup_history(tmp_path, "cand-1")
        assert [entry["request_id"] for entry in history] == ["req-1", "req-2"]


class TestLabelVotes:
    def test_put_then_read_back_for_one_candidate(self, tmp_path):
        metadata.put_label_vote(tmp_path, "vote-1", "cand-1", "alice", "interesting", "")
        votes = metadata.label_votes(tmp_path, "cand-1")
        assert len(votes) == 1
        assert votes[0]["reviewer_id"] == "alice"
        # Experimental-condition fields default to None for an ordinary vote.
        assert votes[0]["arm"] is None

    def test_multiple_reviewers_each_add_a_row_rather_than_overwrite(self, tmp_path):
        metadata.put_label_vote(tmp_path, "vote-1", "cand-1", "alice", "interesting", "")
        metadata.put_label_vote(tmp_path, "vote-2", "cand-1", "bob", "artifact", "")
        assert len(metadata.label_votes(tmp_path, "cand-1")) == 2

    def test_all_label_votes_spans_every_candidate(self, tmp_path):
        metadata.put_label_vote(tmp_path, "vote-1", "cand-1", "alice", "interesting", "")
        metadata.put_label_vote(tmp_path, "vote-2", "cand-2", "alice", "artifact", "")
        assert len(metadata.all_label_votes(tmp_path)) == 2

    def test_experimental_fields_round_trip(self, tmp_path):
        metadata.put_label_vote(
            tmp_path, "vote-1", "cand-1", "alice", "interesting", "",
            arm="active_learning", displayed_score=0.87, decision_latency_ms=2500,
            self_reported_confidence=0.6, presentation_index=3)
        vote = metadata.label_votes(tmp_path, "cand-1")[0]
        assert vote["arm"] == "active_learning"
        assert vote["displayed_score"] == pytest.approx(0.87)
        assert vote["decision_latency_ms"] == 2500
        assert vote["presentation_index"] == 3


class TestJobs:
    def test_put_then_get_round_trips(self, tmp_path):
        metadata.put_job(tmp_path, "job-1", "acquire.cone", "running",
                         params={"ra_deg": 180.0}, project_id="proj-1")
        job = metadata.get_job(tmp_path, "job-1")
        assert job["status"] == "running"
        assert job["params"] == {"ra_deg": 180.0}
        assert job["project_id"] == "proj-1"
        assert job["cancel_requested"] is False

    def test_get_missing_job_returns_none(self, tmp_path):
        assert metadata.get_job(tmp_path, "does-not-exist") is None

    def test_put_again_updates_the_existing_row_not_a_duplicate(self, tmp_path):
        metadata.put_job(tmp_path, "job-1", "acquire.cone", "running")
        metadata.put_job(tmp_path, "job-1", "acquire.cone", "done", result={"n": 5})
        job = metadata.get_job(tmp_path, "job-1")
        assert job["status"] == "done"
        assert job["result"] == {"n": 5}
        assert len(metadata.list_jobs(tmp_path)) == 1

    def test_updating_without_params_keeps_the_prior_params(self, tmp_path):
        metadata.put_job(tmp_path, "job-1", "acquire.cone", "running", params={"ra_deg": 1.0})
        metadata.put_job(tmp_path, "job-1", "acquire.cone", "done")
        assert metadata.get_job(tmp_path, "job-1")["params"] == {"ra_deg": 1.0}

    def test_list_jobs_filters_by_status(self, tmp_path):
        metadata.put_job(tmp_path, "job-1", "m", "running")
        metadata.put_job(tmp_path, "job-2", "m", "done")
        running = metadata.list_jobs(tmp_path, statuses=("running",))
        assert [j["job_id"] for j in running] == ["job-1"]

    def test_find_job_by_idempotency_key_returns_the_latest(self, tmp_path):
        metadata.put_job(tmp_path, "job-1", "m", "done", idempotency_key="key-1")
        found = metadata.find_job_by_idempotency(tmp_path, "key-1")
        assert found["job_id"] == "job-1"

    def test_find_job_by_unknown_idempotency_key_returns_none(self, tmp_path):
        assert metadata.find_job_by_idempotency(tmp_path, "nope") is None

    def test_request_and_clear_cancel_round_trip(self, tmp_path):
        metadata.put_job(tmp_path, "job-1", "m", "running")
        metadata.request_job_cancel(tmp_path, "job-1")
        assert metadata.get_job(tmp_path, "job-1")["cancel_requested"] is True
        metadata.clear_job_cancel(tmp_path, "job-1")
        assert metadata.get_job(tmp_path, "job-1")["cancel_requested"] is False


class TestCatalogCache:
    def _put(self, tmp_path, **overrides):
        kwargs = dict(cache_key="k1", provider="simbad", release="SIMBAD-current",
                      query_hash="h1", query={"ra_deg": 1.0}, object_id="obj-1",
                      ra_deg=1.0, dec_deg=2.0, radius_arcsec=2.0, status="match",
                      response={"matches": []}, error=None,
                      fetched_utc="2026-01-01T00:00:00", expires_utc="2026-02-01T00:00:00")
        kwargs.update(overrides)
        metadata.put_catalog_cache(tmp_path, **kwargs)

    def test_put_then_get_round_trips(self, tmp_path):
        self._put(tmp_path)
        entry = metadata.get_catalog_cache(tmp_path, "k1")
        assert entry["provider"] == "simbad"
        assert entry["response"] == {"matches": []}

    def test_get_missing_key_returns_none(self, tmp_path):
        assert metadata.get_catalog_cache(tmp_path, "does-not-exist") is None

    def test_get_refreshes_accessed_utc(self, tmp_path):
        self._put(tmp_path)
        first = metadata.get_catalog_cache(tmp_path, "k1")["accessed_utc"]
        self._put(tmp_path, fetched_utc="2026-01-02T00:00:00")  # re-store, different fetch time
        second = metadata.get_catalog_cache(tmp_path, "k1")
        assert second["fetched_utc"] == "2026-01-02T00:00:00"

    def test_put_upserts_on_cache_key(self, tmp_path):
        self._put(tmp_path, status="match")
        self._put(tmp_path, status="no_match", response=None)
        entry = metadata.get_catalog_cache(tmp_path, "k1")
        assert entry["status"] == "no_match"
        assert entry["response"] is None

    def test_summary_groups_by_provider_and_status(self, tmp_path):
        self._put(tmp_path, cache_key="k1", provider="simbad", status="match")
        self._put(tmp_path, cache_key="k2", provider="simbad", status="match")
        self._put(tmp_path, cache_key="k3", provider="vsx", status="no_match")
        summary = metadata.catalog_cache_summary(tmp_path)
        assert summary["total"] == 3
        assert {(e["provider"], e["status"], e["count"]) for e in summary["entries"]} == {
            ("simbad", "match", 2), ("vsx", "no_match", 1),
        }


class TestLiteratureCache:
    def test_put_then_get_round_trips(self, tmp_path):
        metadata.put_literature_cache(
            tmp_path, cache_key="k1", provider="ads", release="ADS-current",
            query_hash="h1", query={"terms": ["nova"]}, status="match",
            response={"records": []}, error=None,
            fetched_utc="2026-01-01T00:00:00", expires_utc="2026-02-01T00:00:00")
        entry = metadata.get_literature_cache(tmp_path, "k1")
        assert entry["provider"] == "ads"
        assert entry["response"] == {"records": []}

    def test_get_missing_key_returns_none(self, tmp_path):
        assert metadata.get_literature_cache(tmp_path, "nope") is None

    def test_summary_totals_across_providers(self, tmp_path):
        for i in range(2):
            metadata.put_literature_cache(
                tmp_path, cache_key=f"k{i}", provider="ads", release="r",
                query_hash=f"h{i}", query={}, status="match", response=None,
                error=None, fetched_utc="2026-01-01T00:00:00",
                expires_utc="2026-02-01T00:00:00")
        assert metadata.literature_cache_summary(tmp_path)["total"] == 2


class TestTapCache:
    def test_put_then_get_round_trips(self, tmp_path):
        metadata.put_tap_cache(
            tmp_path, cache_key="k1", service="exoplanetarchive", release="r1",
            query_hash="h1", query={"q": "select"}, status="match",
            response={"rows": []}, error=None,
            fetched_utc="2026-01-01T00:00:00", expires_utc="2026-02-01T00:00:00")
        entry = metadata.get_tap_cache(tmp_path, "k1")
        assert entry["service"] == "exoplanetarchive"

    def test_summary_totals_across_services(self, tmp_path):
        metadata.put_tap_cache(
            tmp_path, cache_key="k1", service="s1", release="r", query_hash="h1",
            query={}, status="match", response=None, error=None,
            fetched_utc="2026-01-01T00:00:00", expires_utc="2026-02-01T00:00:00")
        assert metadata.tap_cache_summary(tmp_path)["total"] == 1


class TestAlertCursors:
    def test_put_then_get_round_trips(self, tmp_path):
        metadata.put_alert_cursor(tmp_path, "alerce", cursor="c-123",
                                  packet_count=5, last_poll_utc="2026-01-01T00:00:00")
        entry = metadata.get_alert_cursor(tmp_path, "alerce")
        assert entry["cursor"] == "c-123"
        assert entry["packet_count"] == 5

    def test_get_missing_provider_returns_none(self, tmp_path):
        assert metadata.get_alert_cursor(tmp_path, "nope") is None

    def test_put_upserts_on_provider(self, tmp_path):
        metadata.put_alert_cursor(tmp_path, "alerce", cursor="c-1", packet_count=1,
                                  last_poll_utc=None)
        metadata.put_alert_cursor(tmp_path, "alerce", cursor="c-2", packet_count=2,
                                  last_poll_utc=None)
        assert metadata.get_alert_cursor(tmp_path, "alerce")["cursor"] == "c-2"

    def test_summary_lists_every_provider(self, tmp_path):
        metadata.put_alert_cursor(tmp_path, "alerce", cursor=None, packet_count=0,
                                  last_poll_utc=None)
        metadata.put_alert_cursor(tmp_path, "ztf", cursor=None, packet_count=0,
                                  last_poll_utc=None)
        providers = {row["provider"] for row in metadata.alert_cursor_summary(tmp_path)}
        assert providers == {"alerce", "ztf"}


def _packet(packet_key="p1", event_id="S123456", packet_id="p1", **overrides) -> dict:
    payload = {"packet_key": packet_key, "event_id": event_id, "packet_id": packet_id,
              "provider": "gwosc", "release": "O4", "packet_version": "1",
              "event_time": "2026-01-01T00:00:00", "received_utc": "2026-01-01T00:01:00",
              "localization": {"ra_deg": 180.0}, "classifications": ["BBH"],
              "related_ids": [], "raw_sha256": "abc123", "raw_path": "/tmp/p1.json",
              "status": "received", "project_id": None}
    payload.update(overrides)
    return payload


class TestEventPacketsAndClusters:
    def test_put_then_get_round_trips(self, tmp_path):
        metadata.put_event_packet(tmp_path, _packet())
        entry = metadata.get_event_packet(tmp_path, "p1")
        assert entry["event_id"] == "S123456"
        assert entry["localization"] == {"ra_deg": 180.0}

    def test_get_missing_packet_returns_none(self, tmp_path):
        assert metadata.get_event_packet(tmp_path, "nope") is None

    def test_a_second_packet_for_the_same_event_extends_the_cluster(self, tmp_path):
        metadata.put_event_packet(tmp_path, _packet(packet_key="p1", packet_id="p1"))
        metadata.put_event_packet(tmp_path, _packet(
            packet_key="p2", packet_id="p2", received_utc="2026-01-01T00:05:00"))
        clusters = metadata.list_event_clusters(tmp_path)
        assert len(clusters) == 1
        assert clusters[0]["packet_count"] == 2
        assert clusters[0]["last_seen_utc"] == "2026-01-01T00:05:00"

    def test_the_same_packet_replayed_does_not_duplicate_the_cluster_count(self, tmp_path):
        packet = _packet()
        metadata.put_event_packet(tmp_path, packet)
        metadata.put_event_packet(tmp_path, packet)
        clusters = metadata.list_event_clusters(tmp_path)
        assert clusters[0]["packet_count"] == 1

    def test_list_event_packets_filters_by_provider(self, tmp_path):
        metadata.put_event_packet(tmp_path, _packet(packet_key="p1", provider="gwosc"))
        metadata.put_event_packet(tmp_path, _packet(
            packet_key="p2", packet_id="p2", event_id="E2", provider="chime"))
        gwosc_only = metadata.list_event_packets(tmp_path, provider="gwosc")
        assert [p["packet_key"] for p in gwosc_only] == ["p1"]

    def test_list_event_clusters_filters_by_provider(self, tmp_path):
        metadata.put_event_packet(tmp_path, _packet(packet_key="p1", provider="gwosc"))
        metadata.put_event_packet(tmp_path, _packet(
            packet_key="p2", packet_id="p2", event_id="E2", provider="chime"))
        assert len(metadata.list_event_clusters(tmp_path, provider="chime")) == 1
