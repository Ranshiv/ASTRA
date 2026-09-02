"""schedule.py: greedy-insertion sequencing, slew-reducing local search,
and mid-night replanning (Direction 1, "closed-loop decision-theoretic
scheduling")."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from astra import schedule as sch

START = "2026-12-01T00:00:00Z"  # a long winter night at the default site -- plenty of window

# Circumpolar at the default site (latitude 43.65 N) -- always above 30 deg
# altitude regardless of the hour, the same "always visible" convention
# test_information_gain.py already uses (dec=80.0).
VISIBLE_A = {"candidate_id": "a", "ra_deg": 180.0, "dec_deg": 80.0, "tail_probability": 0.5}
VISIBLE_B = {"candidate_id": "b", "ra_deg": 0.0, "dec_deg": 80.0, "tail_probability": 0.5}
VISIBLE_C = {"candidate_id": "c", "ra_deg": 90.0, "dec_deg": 80.0, "tail_probability": 0.5}
NEVER_VISIBLE = {"candidate_id": "hidden", "ra_deg": 180.0, "dec_deg": -85.0,
                 "tail_probability": 0.5}


class TestWithinAnyWindow:
    def test_start_and_end_both_inside_one_window_is_true(self):
        base = datetime(2026, 12, 1, tzinfo=timezone.utc)
        windows = [(base, base + timedelta(hours=2))]
        assert sch._within_any_window(base + timedelta(minutes=10),
                                      base + timedelta(minutes=40), windows) is True

    def test_end_outside_every_window_is_false(self):
        base = datetime(2026, 12, 1, tzinfo=timezone.utc)
        windows = [(base, base + timedelta(hours=1))]
        assert sch._within_any_window(base + timedelta(minutes=50),
                                      base + timedelta(hours=2), windows) is False

    def test_no_windows_is_false(self):
        base = datetime(2026, 12, 1, tzinfo=timezone.utc)
        assert sch._within_any_window(base, base + timedelta(hours=1), []) is False


class TestLocalSearchRespectsWindows:
    def _timeline(self) -> list:
        base = datetime(2026, 12, 1, tzinfo=timezone.utc)
        # a(170) is far from both b(0) and c(175); swapping a and b (slots
        # 0/1) brings a next to c (5 deg apart) instead of b, reducing
        # total slew from |170-0|+|0-175|=345 to |0-170|+|170-175|=175.
        obs_a = sch.ScheduledObservation(
            candidate_id="a", ra_deg=170.0, dec_deg=80.0,
            start_utc=base.isoformat(), end_utc=(base + timedelta(hours=1)).isoformat(),
            exposure_hours=1.0, entropy_bits=1.0, slew_deg_from_previous=None)
        obs_b = sch.ScheduledObservation(
            candidate_id="b", ra_deg=0.0, dec_deg=80.0,
            start_utc=(base + timedelta(hours=1)).isoformat(),
            end_utc=(base + timedelta(hours=2)).isoformat(),
            exposure_hours=1.0, entropy_bits=1.0, slew_deg_from_previous=None)
        obs_c = sch.ScheduledObservation(
            candidate_id="c", ra_deg=175.0, dec_deg=80.0,
            start_utc=(base + timedelta(hours=2)).isoformat(),
            end_utc=(base + timedelta(hours=3)).isoformat(),
            exposure_hours=1.0, entropy_bits=1.0, slew_deg_from_previous=None)
        return [obs_a, obs_b, obs_c], base

    def test_without_windows_arg_swap_behavior_is_unchanged(self):
        timeline, _ = self._timeline()
        before_ids = [o.candidate_id for o in timeline]
        violations = sch._local_search_reduce_slew(timeline, passes=3)
        assert violations == 0
        assert [o.candidate_id for o in timeline] != before_ids  # a swap happened

    def test_swap_rejected_when_it_would_violate_a_true_window(self):
        timeline, base = self._timeline()
        before_ids = [o.candidate_id for o in timeline]
        # Candidate "a" is only ever visible in slot a's own (original)
        # time -- the beneficial swap would move it into slot b's time,
        # which must be rejected.
        windows_by_id = {
            "a": [(base, base + timedelta(hours=1))],
            "b": [(base, base + timedelta(hours=3))],
            "c": [(base, base + timedelta(hours=3))],
        }
        violations = sch._local_search_reduce_slew(
            timeline, passes=3, windows_by_candidate_id=windows_by_id)
        assert violations > 0
        # "a" must stay in its own slot -- b and c may still legitimately
        # swap between themselves, since both have wide-enough windows.
        assert timeline[0].candidate_id == before_ids[0] == "a"

    def test_swap_accepted_when_all_windows_are_wide_enough(self):
        timeline, base = self._timeline()
        windows_by_id = {
            "a": [(base, base + timedelta(hours=3))],
            "b": [(base, base + timedelta(hours=3))],
            "c": [(base, base + timedelta(hours=3))],
        }
        violations = sch._local_search_reduce_slew(
            timeline, passes=3, windows_by_candidate_id=windows_by_id)
        assert violations == 0


class TestBuildNightSchedule:
    def test_schedules_a_visible_candidate(self):
        result = sch.build_night_schedule([VISIBLE_A], start_utc=START, duration_hours=12.0,
                                          exposure_hours=0.5)
        assert len(result.observations) == 1
        assert result.observations[0].candidate_id == "a"
        assert result.unscheduled_candidate_ids == []

    def test_never_visible_candidate_is_reported_unscheduled(self):
        result = sch.build_night_schedule([NEVER_VISIBLE], start_utc=START, duration_hours=12.0)
        assert result.observations == []
        assert result.unscheduled_candidate_ids == ["hidden"]

    def test_schedules_multiple_non_overlapping_candidates(self):
        result = sch.build_night_schedule(
            [VISIBLE_A, VISIBLE_B, VISIBLE_C], start_utc=START, duration_hours=12.0,
            exposure_hours=0.5)
        scheduled_ids = {o.candidate_id for o in result.observations}
        assert scheduled_ids == {"a", "b", "c"}

    def test_no_two_observations_overlap(self):
        result = sch.build_night_schedule(
            [VISIBLE_A, VISIBLE_B, VISIBLE_C], start_utc=START, duration_hours=6.0,
            exposure_hours=0.5)
        ordered = sorted(result.observations, key=lambda o: o.start_utc)
        for previous, current in zip(ordered, ordered[1:]):
            assert previous.end_utc <= current.start_utc

    def test_higher_priority_candidate_is_not_left_unscheduled_when_time_is_tight(self):
        # Only enough room for one exposure; the far-more-confident (lower
        # entropy, lower priority) candidate should yield to the ambiguous
        # (higher-entropy, higher-priority) one when both compete for the
        # same slot.
        tight_a = {**VISIBLE_A, "tail_probability": 0.5}   # max entropy -> highest priority
        tight_b = {**VISIBLE_B, "tail_probability": 0.02}  # low entropy -> lower priority
        # duration_hours slightly over exposure_hours: followup.plan's
        # sampled windows end at the last SAMPLE inside the window, not at
        # start+duration exactly, so a window this tight only fits one
        # exposure -- exactly the contention this test wants.
        result = sch.build_night_schedule([tight_a, tight_b], start_utc=START,
                                          duration_hours=0.6, exposure_hours=0.5)
        assert len(result.observations) == 1
        assert result.observations[0].candidate_id == "a"
        assert result.unscheduled_candidate_ids == ["b"]

    def test_slew_is_recorded_between_consecutive_observations(self):
        result = sch.build_night_schedule(
            [VISIBLE_A, VISIBLE_B], start_utc=START, duration_hours=6.0, exposure_hours=0.5)
        ordered = sorted(result.observations, key=lambda o: o.start_utc)
        assert ordered[0].slew_deg_from_previous is None
        assert ordered[1].slew_deg_from_previous is not None
        assert ordered[1].slew_deg_from_previous > 0

    def test_total_exposure_hours_matches_observation_count(self):
        result = sch.build_night_schedule(
            [VISIBLE_A, VISIBLE_B, VISIBLE_C], start_utc=START, duration_hours=12.0,
            exposure_hours=0.5)
        assert result.total_exposure_hours == len(result.observations) * 0.5

    def test_empty_candidate_list_yields_an_empty_schedule(self):
        result = sch.build_night_schedule([], start_utc=START, duration_hours=12.0)
        assert result.observations == []
        assert result.unscheduled_candidate_ids == []

    def test_rejects_nonpositive_exposure_hours(self):
        try:
            sch.build_night_schedule([VISIBLE_A], start_utc=START, exposure_hours=0.0)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_to_dict_round_trips_the_key_fields(self):
        result = sch.build_night_schedule([VISIBLE_A], start_utc=START, duration_hours=12.0)
        payload = result.to_dict()
        assert payload["observations"][0]["candidate_id"] == "a"
        assert payload["total_exposure_hours"] > 0


class TestReplan:
    def test_preserves_already_executed_observations(self):
        original = sch.build_night_schedule(
            [VISIBLE_A, VISIBLE_B], start_utc=START, duration_hours=12.0, exposure_hours=0.5)
        executed_ids = [o.candidate_id for o in original.observations[:1]]
        cutoff = (datetime.fromisoformat(START.replace("Z", "+00:00"))
                 + timedelta(hours=6)).isoformat()

        replanned = sch.replan(original, executed_candidate_ids=executed_ids,
                               remaining_candidates=[VISIBLE_C], from_utc=cutoff)

        replanned_ids = {o.candidate_id for o in replanned.observations}
        for executed_id in executed_ids:
            assert executed_id in replanned_ids

    def test_new_candidates_can_be_scheduled_in_the_remaining_time(self):
        original = sch.build_night_schedule([VISIBLE_A], start_utc=START, duration_hours=12.0,
                                            exposure_hours=0.5)
        cutoff = (datetime.fromisoformat(START.replace("Z", "+00:00"))
                 + timedelta(hours=1)).isoformat()

        replanned = sch.replan(original, executed_candidate_ids=["a"],
                               remaining_candidates=[VISIBLE_B], from_utc=cutoff)

        assert "b" in {o.candidate_id for o in replanned.observations}

    def test_replanned_observations_stay_ordered_by_time(self):
        original = sch.build_night_schedule(
            [VISIBLE_A, VISIBLE_B], start_utc=START, duration_hours=12.0, exposure_hours=0.5)
        cutoff = (datetime.fromisoformat(START.replace("Z", "+00:00"))
                 + timedelta(hours=3)).isoformat()

        replanned = sch.replan(original, executed_candidate_ids=[], remaining_candidates=[VISIBLE_C],
                               from_utc=cutoff)

        starts = [o.start_utc for o in replanned.observations]
        assert starts == sorted(starts)
