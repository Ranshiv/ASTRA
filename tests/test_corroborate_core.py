"""corroborate/core.py: domain-agnostic association and agreement scoring
(Direction 3, "corroboration as a general multi-instrument anomaly
library")."""

from __future__ import annotations

import math

from astra.corroborate import core


def _record(instrument, identifier, position):
    return core.InstrumentRecord(instrument=instrument, identifier=identifier, position=position)


def _distance_1d(a, b):
    return abs(a[0] - b[0])


class TestMatchRecords:
    def test_matches_the_nearest_counterpart_within_radius(self):
        records = [_record("A", "a1", (0.0,))]
        counterparts = [_record("B", "b1", (0.5,)), _record("B", "b2", (5.0,))]

        matches = core.match_records(records, counterparts, _distance_1d, radius=1.0)

        assert len(matches) == 1
        assert matches[0].counterpart.identifier == "b1"
        assert matches[0].distance == 0.5
        assert matches[0].competitors == 0

    def test_no_match_beyond_radius(self):
        records = [_record("A", "a1", (0.0,))]
        counterparts = [_record("B", "b1", (5.0,))]
        assert core.match_records(records, counterparts, _distance_1d, radius=1.0) == []

    def test_counts_competitors_within_radius(self):
        records = [_record("A", "a1", (0.0,))]
        counterparts = [_record("B", "b1", (0.1,)), _record("B", "b2", (0.2,)),
                        _record("B", "b3", (0.9,))]
        matches = core.match_records(records, counterparts, _distance_1d, radius=1.0)
        assert matches[0].competitors == 2

    def test_empty_inputs_yield_no_matches(self):
        assert core.match_records([], [], _distance_1d, radius=1.0) == []


class TestGroupRecords:
    def test_groups_across_instruments_within_radius(self):
        by_instrument = {
            "A": [_record("A", "a1", (0.0,))],
            "B": [_record("B", "b1", (0.3,))],
        }
        groups = core.group_records(by_instrument, _distance_1d, radius=1.0)
        assert len(groups) == 1
        assert groups[0].instruments == ["A", "B"]

    def test_default_anchor_is_the_largest_catalogue(self):
        by_instrument = {
            "A": [_record("A", f"a{i}", (float(i),)) for i in range(5)],
            "B": [_record("B", "b1", (0.0,))],
        }
        groups = core.group_records(by_instrument, _distance_1d, radius=0.5)
        assert len(groups) == 5  # anchored on A's 5 records

    def test_explicit_anchor_is_honoured(self):
        by_instrument = {
            "A": [_record("A", f"a{i}", (float(i),)) for i in range(5)],
            "B": [_record("B", "b1", (0.0,))],
        }
        groups = core.group_records(by_instrument, _distance_1d, radius=0.5, anchor="B")
        assert len(groups) == 1

    def test_unknown_explicit_anchor_raises(self):
        by_instrument = {"A": [_record("A", "a1", (0.0,))]}
        try:
            core.group_records(by_instrument, _distance_1d, radius=1.0, anchor="Z")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "Z" in str(exc)

    def test_solo_instrument_still_yields_a_group(self):
        by_instrument = {"A": [_record("A", "a1", (0.0,))]}
        groups = core.group_records(by_instrument, _distance_1d, radius=1.0)
        assert len(groups) == 1
        assert groups[0].instruments == ["A"]

    def test_ambiguous_flag_set_when_multiple_counterparts_compete(self):
        by_instrument = {
            "A": [_record("A", "a1", (0.0,))],
            "B": [_record("B", "b1", (0.1,)), _record("B", "b2", (0.2,))],
        }
        groups = core.group_records(by_instrument, _distance_1d, radius=1.0, anchor="A")
        assert "B" in groups[0].ambiguous

    def test_a_counterpart_claimed_by_two_groups_is_blended(self):
        by_instrument = {
            "A": [_record("A", "a1", (0.0,)), _record("A", "a2", (0.05,))],
            "B": [_record("B", "b1", (0.02,))],  # equidistant-ish, matches both anchors
        }
        groups = core.group_records(by_instrument, _distance_1d, radius=1.0)
        assert all("B" in group.blended for group in groups
                   if "B" in group.members)

    def test_empty_input_yields_no_groups(self):
        assert core.group_records({}, _distance_1d, radius=1.0) == []


class TestCombineComponents:
    def test_weighted_mean_over_usable_components(self):
        weights = {"a": 1.0, "b": 1.0}
        result = core.combine_components({"a": 1.0, "b": 0.0}, weights)
        assert result.total == 0.5
        assert result.weight_used == 2.0

    def test_none_and_nan_components_are_excluded(self):
        weights = {"a": 1.0, "b": 1.0}
        result = core.combine_components({"a": 1.0, "b": None}, weights)
        assert result.total == 1.0
        assert result.weight_used == 1.0

        nan_result = core.combine_components({"a": 1.0, "b": float("nan")}, weights)
        assert nan_result.total == 1.0
        assert nan_result.weight_used == 1.0

    def test_components_outside_weights_are_ignored(self):
        result = core.combine_components({"a": 1.0, "unknown": 5.0}, {"a": 1.0})
        assert result.total == 1.0

    def test_no_usable_components_yields_zero_total(self):
        result = core.combine_components({"a": None}, {"a": 1.0})
        assert result.total == 0.0
        assert result.weight_used == 0.0

    def test_to_dict_rounds_and_reports_reasons(self):
        result = core.combine_components({"a": 1.0 / 3}, {"a": 1.0}, reasons=["because"])
        payload = result.to_dict()
        assert payload["reasons"] == ["because"]
        assert isinstance(payload["total"], float)
