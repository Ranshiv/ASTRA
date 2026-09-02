"""Local candidate feed file (broadcast.py) -- not a network push, see the
module docstring for why."""

from __future__ import annotations

import json

from astra import broadcast, candidates


def _candidate(candidate_id: str, score_total: float, artifact_likelihood: float = 0.1):
    return candidates.Candidate(
        candidate_id=candidate_id, object_id=candidate_id, survey="ZTF", band="g",
        ra_deg=10.0, dec_deg=20.0, score={"total": score_total},
        artifact={"likelihood": artifact_likelihood},
    )


class TestGenerateFeed:
    def test_only_candidates_at_or_above_threshold_are_included(self, tmp_path, monkeypatch):
        rows = [_candidate("a", 0.9), _candidate("b", 0.4), _candidate("c", 0.5)]
        monkeypatch.setattr(candidates, "load", lambda name, root: rows)

        result = broadcast.generate_feed("default", threshold=0.5, root=tmp_path)

        payload = json.loads((tmp_path / "reports" / "default_feed.json").read_text())
        ids = {row["candidate_id"] for row in payload["candidates"]}
        assert ids == {"a", "c"}
        assert result["count"] == 2
        assert result["threshold"] == 0.5

    def test_envelope_schema_fields_are_present(self, tmp_path, monkeypatch):
        rows = [_candidate("a", 0.9)]
        monkeypatch.setattr(candidates, "load", lambda name, root: rows)

        broadcast.generate_feed("default", root=tmp_path)

        payload = json.loads((tmp_path / "reports" / "default_feed.json").read_text())
        assert payload["schema_version"] == broadcast.SCHEMA_VERSION
        assert "generated_utc" in payload
        assert payload["threshold"] == broadcast.DEFAULT_THRESHOLD
        entry = payload["candidates"][0]
        assert set(entry) == {"candidate_id", "survey", "object_id", "band", "ra_deg",
                              "dec_deg", "score_total", "artifact_likelihood", "published_utc"}

    def test_second_call_overwrites_the_same_stable_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(candidates, "load", lambda name, root: [_candidate("a", 0.9)])
        first = broadcast.generate_feed("default", root=tmp_path)

        monkeypatch.setattr(candidates, "load", lambda name, root: [_candidate("b", 0.9)])
        second = broadcast.generate_feed("default", root=tmp_path)

        assert first["path"] == second["path"]
        payload = json.loads((tmp_path / "reports" / "default_feed.json").read_text())
        assert {row["candidate_id"] for row in payload["candidates"]} == {"b"}

    def test_no_qualifying_candidates_produces_a_valid_empty_feed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(candidates, "load", lambda name, root: [_candidate("a", 0.1)])

        result = broadcast.generate_feed("default", threshold=0.5, root=tmp_path)

        assert result["count"] == 0
        payload = json.loads((tmp_path / "reports" / "default_feed.json").read_text())
        assert payload["candidates"] == []
