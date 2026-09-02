"""gw_adapter.py: synthetic two-detector coincidence population and
grouping (Direction 3's second domain). All data here is explicitly
synthetic -- no real GWOSC/Gravity Spy ingestion, see the module docstring."""

from __future__ import annotations

from astra.corroborate import gw_adapter


class TestGenerateSyntheticDetectorPair:
    def test_produces_the_requested_counts(self):
        population = gw_adapter.generate_synthetic_detector_pair(
            n_real_events=10, n_glitches_a=20, n_glitches_b=20,
            systematics_correlation=0.0, seed=1)
        assert len(population.by_instrument["detector_a"]) == 30  # 10 real + 20 glitches
        assert len(population.by_instrument["detector_b"]) == 30

    def test_truth_labels_every_record(self):
        population = gw_adapter.generate_synthetic_detector_pair(
            n_real_events=5, n_glitches_a=5, n_glitches_b=5, seed=2)
        all_ids = [record.identifier for records in population.by_instrument.values()
                  for record in records]
        assert set(population.truth) == set(all_ids)
        assert sum(population.truth.values()) == 10  # 5 real events x 2 detectors

    def test_is_reproducible_for_a_fixed_seed(self):
        first = gw_adapter.generate_synthetic_detector_pair(seed=7)
        second = gw_adapter.generate_synthetic_detector_pair(seed=7)
        first_times = sorted(r.position for r in first.by_instrument["detector_a"])
        second_times = sorted(r.position for r in second.by_instrument["detector_a"])
        assert first_times == second_times

    def test_zero_correlation_adds_no_extra_b_glitches(self):
        population = gw_adapter.generate_synthetic_detector_pair(
            n_real_events=0, n_glitches_a=20, n_glitches_b=20,
            systematics_correlation=0.0, seed=3)
        assert len(population.by_instrument["detector_b"]) == 20

    def test_full_correlation_pairs_every_a_glitch_with_a_b_glitch(self):
        population = gw_adapter.generate_synthetic_detector_pair(
            n_real_events=0, n_glitches_a=20, n_glitches_b=20,
            systematics_correlation=1.0, seed=4)
        # 20 correlated B glitches injected (one per A glitch) + 0 remaining
        # independent B glitches (n_glitches_b - n_correlated = 0).
        assert len(population.by_instrument["detector_b"]) == 20


class TestGroupPopulation:
    def test_real_events_are_grouped_across_both_detectors(self):
        population = gw_adapter.generate_synthetic_detector_pair(
            n_real_events=20, n_glitches_a=0, n_glitches_b=0,
            real_event_jitter_seconds=0.001, seed=5)
        groups = gw_adapter.group_population(population, window_seconds=0.05)
        both_detector_groups = [g for g in groups if len(g.members) == 2]
        assert len(both_detector_groups) == 20

    def test_independent_glitches_mostly_stay_ungrouped(self):
        population = gw_adapter.generate_synthetic_detector_pair(
            n_real_events=0, n_glitches_a=200, n_glitches_b=200,
            systematics_correlation=0.0, window_seconds=0.05,
            duration_seconds=10_000.0, seed=6)
        groups = gw_adapter.group_population(population, window_seconds=0.05)
        coincident = [g for g in groups if len(g.members) == 2]
        # With a 0.05s window over a 10,000s duration and independent
        # random glitch times, accidental coincidence should be rare.
        assert len(coincident) < 5

    def test_high_correlation_produces_many_false_coincidences(self):
        population = gw_adapter.generate_synthetic_detector_pair(
            n_real_events=0, n_glitches_a=100, n_glitches_b=100,
            systematics_correlation=1.0, window_seconds=0.05, seed=6)
        groups = gw_adapter.group_population(population, window_seconds=0.05)
        coincident = [g for g in groups if len(g.members) == 2]
        assert len(coincident) > 80  # nearly every A glitch has a paired B glitch
