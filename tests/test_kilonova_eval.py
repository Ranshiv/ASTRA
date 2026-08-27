"""kilonova_eval.py: counterpart recall at a fixed telescope budget and
distance-conditioned calibration, against synthetic sky-probability maps
and distance posteriors (no live GW download needed for these)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import kilonova as kn
from astra import kilonova_eval as evaluation


def _uniform_sky_probability(n_pixels: int = 1000) -> np.ndarray:
    probability = np.full(n_pixels, 1.0 / n_pixels)
    return probability


def _peaked_sky_probability(n_pixels: int = 1000, peak_pixels: int = 20) -> np.ndarray:
    # Most of the probability mass concentrated in a small footprint --
    # the realistic case a real GW skymap presents.
    probability = np.full(n_pixels, 1e-6)
    probability[:peak_pixels] = 1.0
    return probability / probability.sum()


def _distance_samples(mean_mpc: float = 40.0, spread_mpc: float = 8.0, n: int = 2000,
                      seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = rng.normal(mean_mpc, spread_mpc, n)
    return samples[samples > 0]


class TestFluxAtDistance:
    def test_matches_a_direct_model_evaluation(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        reference_flux = evaluation.reference_flux_at_epochs([params], [1.0], 6231.0)
        rescaled = evaluation.flux_at_distance(reference_flux, 40.0)
        direct = kn.blackbody_band_flux(
            np.array([86400.0]), params, 6231.0, distance_mpc=40.0)
        assert rescaled[0] == pytest.approx(direct[0], rel=1e-6)

    def test_obeys_inverse_square_law(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        reference_flux = evaluation.reference_flux_at_epochs([params], [1.0], 6231.0)
        near = evaluation.flux_at_distance(reference_flux, 10.0)
        far = evaluation.flux_at_distance(reference_flux, 20.0)
        assert far[0] == pytest.approx(near[0] / 4.0, rel=1e-9)


class TestCounterpartRecallAtBudget:
    def test_rejects_zero_pointings(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        with pytest.raises(evaluation.KilonovaEvalError):
            evaluation.counterpart_recall_at_budget(
                [params], sky_probability=_uniform_sky_probability(),
                distance_samples_mpc=_distance_samples(), n_pointings=0, limiting_ab_mag=22.0)

    def test_rejects_empty_distance_samples(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        with pytest.raises(evaluation.KilonovaEvalError):
            evaluation.counterpart_recall_at_budget(
                [params], sky_probability=_uniform_sky_probability(),
                distance_samples_mpc=np.array([]), n_pointings=10, limiting_ab_mag=22.0)

    def test_recall_bounded_between_zero_and_one(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        result = evaluation.counterpart_recall_at_budget(
            [params], sky_probability=_peaked_sky_probability(),
            distance_samples_mpc=_distance_samples(), n_pointings=20,
            limiting_ab_mag=22.0, n_trials=300, seed=1)
        assert 0.0 <= result["recall_at_budget"] <= 1.0
        assert 0.0 <= result["footprint_fraction"] <= 1.0

    def test_full_footprint_gives_perfect_sky_coverage(self):
        # Tiling every pixel means the true position is ALWAYS in the
        # footprint, whether or not it is bright enough to detect.
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        n_pixels = 1000
        result = evaluation.counterpart_recall_at_budget(
            [params], sky_probability=_uniform_sky_probability(n_pixels),
            distance_samples_mpc=_distance_samples(), n_pointings=n_pixels,
            limiting_ab_mag=22.0, n_trials=200, seed=2)
        assert result["footprint_fraction"] == pytest.approx(1.0)

    def test_no_signal_case_gives_zero_recall_not_a_crash(self):
        # An impossibly faint limiting magnitude: nothing is ever
        # detected, even with full sky coverage -- the explicit
        # no-signal regression case.
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        n_pixels = 200
        result = evaluation.counterpart_recall_at_budget(
            [params], sky_probability=_uniform_sky_probability(n_pixels),
            distance_samples_mpc=_distance_samples(), n_pointings=n_pixels,
            limiting_ab_mag=-5.0, n_trials=100, seed=3)
        assert result["recall_at_budget"] == 0.0

    def test_recall_improves_with_a_fainter_limiting_magnitude(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        common = dict(components=[params], sky_probability=_peaked_sky_probability(),
                     distance_samples_mpc=_distance_samples(mean_mpc=100.0, spread_mpc=5.0),
                     n_pointings=20, n_trials=400, seed=4)
        shallow = evaluation.counterpart_recall_at_budget(limiting_ab_mag=15.0, **common)
        deep = evaluation.counterpart_recall_at_budget(limiting_ab_mag=25.0, **common)
        assert deep["recall_at_budget"] >= shallow["recall_at_budget"]

    def test_more_pointings_never_decreases_footprint_fraction(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        common = dict(components=[params], sky_probability=_peaked_sky_probability(),
                     distance_samples_mpc=_distance_samples(), limiting_ab_mag=22.0,
                     n_trials=400, seed=5)
        few = evaluation.counterpart_recall_at_budget(n_pointings=5, **common)
        many = evaluation.counterpart_recall_at_budget(n_pointings=100, **common)
        assert many["footprint_fraction"] >= few["footprint_fraction"]


class TestDistanceConditionedCalibration:
    def test_rejects_empty_distance_samples(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        with pytest.raises(evaluation.KilonovaEvalError):
            evaluation.distance_conditioned_calibration(
                [params], np.array([]), epoch_days=1.0)

    def test_reports_all_named_metrics(self):
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        result = evaluation.distance_conditioned_calibration(
            [params], _distance_samples(), epoch_days=1.0, n_trials=40, n_resamples=200, seed=6)
        assert result["n_trials"] == 40
        assert "parameter_bias" in result
        assert "posterior_coverage" in result
        assert "sbc" in result

    def test_coverage_is_reasonably_well_calibrated(self):
        # Bootstrapping the SAME finite distance-sample array for both the
        # truth draw and the interval should give coverage close to
        # nominal -- this is a well-posed, internally consistent
        # calibration check, not dependent on any real GW event.
        params = kn.KilonovaParams(m_ej=0.01, v_ej=0.2, kappa=1.0)
        result = evaluation.distance_conditioned_calibration(
            [params], _distance_samples(n=5000), epoch_days=1.0,
            n_trials=150, n_resamples=500, levels=(0.68, 0.9), seed=7)
        coverage_90 = result["posterior_coverage"]["levels"]["0.9"]["apparent_mag"]["empirical"]
        assert coverage_90 is not None
        assert 0.7 <= coverage_90 <= 1.0


def test_not_referenced_by_rpc():
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "kilonova_eval" not in source
