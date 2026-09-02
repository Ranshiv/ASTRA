"""schedule_eval.py: retrospective scheduling-policy simulator (Direction
1 evaluation harness). All data is synthetic, ground truth known by
construction -- the same discipline discard_pile_eval.py/
review_experiment_eval.py already use."""

from __future__ import annotations

from astra import schedule_eval as se


class TestSimulatePolicy:
    def test_resolves_at_least_some_candidates_over_many_nights(self):
        population, truth = se._synthetic_population(
            __import__("numpy").random.default_rng(0), n_candidates=30,
            real_fraction=0.4, field_size_deg=1.0)
        result = se.simulate_policy(population, truth, policy="information_gain",
                                    n_nights=15, seed=0)
        assert result.resolved_by_night[-1] > 0

    def test_resolved_count_is_nondecreasing_across_nights(self):
        population, truth = se._synthetic_population(
            __import__("numpy").random.default_rng(1), n_candidates=20,
            real_fraction=0.5, field_size_deg=1.0)
        result = se.simulate_policy(population, truth, policy="random", n_nights=10, seed=1)
        for previous, current in zip(result.resolved_by_night, result.resolved_by_night[1:]):
            assert current >= previous

    def test_telescope_hours_are_nondecreasing(self):
        population, truth = se._synthetic_population(
            __import__("numpy").random.default_rng(2), n_candidates=20,
            real_fraction=0.5, field_size_deg=1.0)
        result = se.simulate_policy(population, truth, policy="rank_order", n_nights=10, seed=2)
        for previous, current in zip(result.telescope_hours_by_night,
                                     result.telescope_hours_by_night[1:]):
            assert current >= previous

    def test_unknown_policy_raises(self):
        try:
            se._priority_fn("not_a_policy", __import__("numpy").random.default_rng(0))
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_is_reproducible_for_a_fixed_seed(self):
        population, truth = se._synthetic_population(
            __import__("numpy").random.default_rng(3), n_candidates=20,
            real_fraction=0.5, field_size_deg=1.0)
        first = se.simulate_policy(population, truth, policy="information_gain",
                                   n_nights=8, seed=5)
        second = se.simulate_policy(population, truth, policy="information_gain",
                                    n_nights=8, seed=5)
        assert first.resolved_by_night == second.resolved_by_night


class TestEvaluateSchedulingPolicies:
    def test_returns_every_policy(self):
        result = se.evaluate_scheduling_policies(n_candidates=40, n_nights=8, n_runs=4,
                                                 hour_budgets=(16.0, 40.0), seed=0)
        assert set(result["by_policy"]) == set(se.POLICIES)

    def test_curve_has_one_point_per_hour_budget(self):
        budgets = (16.0, 40.0, 80.0)
        result = se.evaluate_scheduling_policies(n_candidates=40, n_nights=8, n_runs=4,
                                                 hour_budgets=budgets, seed=0)
        for policy_result in result["by_policy"].values():
            assert [p["hour_budget"] for p in policy_result["curve"]] == list(budgets)

    def test_information_gain_leads_on_entropy_reduced_at_an_early_budget(self):
        # The metric information_gain is actually designed to optimise --
        # measured, not assumed: see evaluate_scheduling_policies' own
        # docstring for why resolved_candidates does NOT show the same
        # ordering at a small time budget.
        result = se.evaluate_scheduling_policies(n_candidates=60, n_nights=8, n_runs=6,
                                                 hour_budgets=(16.0,), seed=0)
        info_gain = result["by_policy"]["information_gain"]["curve"][0]
        rank_order = result["by_policy"]["rank_order"]["curve"][0]
        assert (info_gain["entropy_reduced_bits"]["mean"]
               > rank_order["entropy_reduced_bits"]["mean"])

    def test_every_checkpoint_reports_a_confidence_interval(self):
        result = se.evaluate_scheduling_policies(n_candidates=30, n_nights=6, n_runs=4,
                                                 hour_budgets=(16.0,), seed=0)
        for policy_result in result["by_policy"].values():
            point = policy_result["curve"][0]["resolved_candidates"]
            low, high = point["ci"]
            assert low <= point["mean"] <= high


class TestEvaluateRobustness:
    def test_reports_one_point_per_noise_level(self):
        result = se.evaluate_robustness(n_candidates=30, n_nights=6, n_runs=3,
                                        noise_levels=(0.0, 0.5), seed=0)
        assert len(result["points"]) == 2
        assert [p["noise_std"] for p in result["points"]] == [0.0, 0.5]

    def test_zero_noise_information_gain_has_a_positive_early_advantage(self):
        result = se.evaluate_robustness(n_candidates=50, n_nights=8, n_runs=6,
                                        noise_levels=(0.0,), seed=0)
        assert result["points"][0]["information_gain_advantage"] > 0

    def test_heavy_noise_erodes_the_advantage(self):
        result = se.evaluate_robustness(n_candidates=50, n_nights=8, n_runs=6,
                                        noise_levels=(0.0, 1.5), seed=0)
        clean, noisy = result["points"]
        assert noisy["information_gain_advantage"] < clean["information_gain_advantage"]
