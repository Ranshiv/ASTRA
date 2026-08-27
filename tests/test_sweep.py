"""Hyperparameter search, and the discipline that keeps it honest.

The behaviour under test is mostly refusal. A grid search always produces a
ranking; the question is whether that ranking means anything. With three
injection seeds the intervals are wide, and this project has already been
burned once by reading a single small-n run as a result (PCA 0.742 beating the
autoencoder's 0.622 on 54 sequences, which mostly measured data starvation).
So `best()` must decline to name a winner whose interval overlaps the
runner-up's.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import sweep


def _trial(name, aucs, ap=None):
    return sweep.TrialResult(
        parameters={"latent_dim": name},
        roc_auc=list(aucs),
        average_precision=list(ap if ap is not None else aucs),
    )


class TestGrid:
    def test_defaults_are_inside_the_grid(self):
        """A sweep that cannot reproduce the current settings cannot tell you
        whether changing them helped."""
        configs = sweep.grid("autoencoder")
        assert {"latent_dim": 16, "channels": (16, 32, 64),
                "learning_rate": 1e-3} in configs

    def test_capacity_is_explored_in_both_directions(self):
        latents = {config["latent_dim"] for config in sweep.grid("autoencoder")}
        assert min(latents) < 16 < max(latents)

    def test_kl_weight_only_applies_to_the_vae(self):
        assert all("kl_weight" not in c for c in sweep.grid("autoencoder"))
        assert all("kl_weight" in c for c in sweep.grid("vae"))

    def test_overrides_replace_a_dimension(self):
        configs = sweep.grid("autoencoder", {"latent_dim": (4,)})
        assert {c["latent_dim"] for c in configs} == {4}

    def test_neural_ode_gets_its_own_dimensions(self):
        """Sweeping latent_dim/channels for neural_ode would vary nothing --
        it needs ode_hidden_dim/ode_steps instead."""
        configs = sweep.grid("neural_ode")
        assert all("latent_dim" not in c and "channels" not in c for c in configs)
        assert all("ode_hidden_dim" in c and "ode_steps" in c for c in configs)

    def test_neural_ode_capacity_is_explored_in_both_directions(self):
        hidden_dims = {config["ode_hidden_dim"] for config in sweep.grid("neural_ode")}
        assert min(hidden_dims) < 32 < max(hidden_dims)


class TestTrialSummary:
    def test_interval_is_reported_across_seeds(self):
        summary = _trial(8, [0.70, 0.75, 0.80]).to_dict()["roc_auc"]
        assert summary["mean"] == pytest.approx(0.75)
        assert summary["ci95"][0] <= 0.75 <= summary["ci95"][1]

    def test_a_trial_that_never_scored_is_absent_not_zero(self):
        """A configuration that crashed has produced no evidence; recording it
        as 0.0 would rank it below a genuinely bad one."""
        trial = sweep.TrialResult(parameters={}, note="RuntimeError: boom")
        assert trial.to_dict()["roc_auc"] is None
        assert not np.isfinite(trial.mean_auc)

    def test_non_finite_scores_are_dropped_from_the_summary(self):
        summary = _trial(8, [0.7, float("nan"), 0.9]).to_dict()["roc_auc"]
        assert summary["mean"] == pytest.approx(0.8)


class TestWinnerSelection:
    def test_clear_separation_names_a_winner(self):
        result = sweep.SweepResult(kind="autoencoder", seeds=[1, 2, 3], trials=[
            _trial(8, [0.90, 0.91, 0.92]),
            _trial(16, [0.50, 0.51, 0.52]),
        ])
        best = result.best()
        assert best is not None and best.parameters["latent_dim"] == 8
        assert result.to_dict()["separated"] is True

    def test_overlapping_intervals_refuse_to_name_a_winner(self):
        """The case this project keeps hitting: a ranking that is really noise."""
        result = sweep.SweepResult(kind="autoencoder", seeds=[1, 2, 3], trials=[
            _trial(8, [0.70, 0.62, 0.78]),
            _trial(16, [0.69, 0.60, 0.76]),
        ])
        assert result.best() is None

        payload = result.to_dict()
        assert payload["best"] is None
        assert payload["separated"] is False
        assert "not evidence" in payload["note"]

    def test_ranking_is_still_reported_when_undecided(self):
        """Refusing to declare a winner is not refusing to show the numbers."""
        result = sweep.SweepResult(kind="autoencoder", seeds=[1, 2], trials=[
            _trial(8, [0.70, 0.62]), _trial(16, [0.69, 0.60]),
        ])
        trials = result.to_dict()["trials"]
        assert len(trials) == 2
        assert trials[0]["roc_auc"]["mean"] >= trials[1]["roc_auc"]["mean"]

    def test_a_single_trial_is_its_own_winner(self):
        result = sweep.SweepResult(kind="autoencoder", seeds=[1, 2],
                                   trials=[_trial(8, [0.7, 0.8])])
        assert result.best() is not None

    def test_no_scored_trials_yields_no_winner(self):
        result = sweep.SweepResult(kind="autoencoder", seeds=[1, 2], trials=[
            sweep.TrialResult(parameters={}, note="failed"),
        ])
        assert result.best() is None
        assert result.to_dict()["best"] is None


class TestRunGuards:
    def test_a_single_seed_is_refused(self):
        """One seed cannot distinguish a better configuration from a luckier
        injection draw, which is the whole reason this module exists."""
        with pytest.raises(ValueError, match="at least two seeds"):
            sweep.run(seeds=(42,))

    def test_an_empty_store_is_reported_not_crashed(self, isolated_root):
        pytest.importorskip("torch")
        result = sweep.run(survey="ztf", seeds=(1, 2))
        assert result.rows == 0
        assert "need at least 20" in result.to_dict()["note"]

    def test_neural_ode_requires_irregular_mode(self):
        """The other modes' 2-channel batches would fail deep inside the
        model with an opaque tensor-shape error; this guard is the clear
        version of that same refusal."""
        with pytest.raises(ValueError, match="requires mode='irregular'"):
            sweep.run(kind="neural_ode", seeds=(1, 2), mode="time")


class TestWithoutTorch:
    def test_missing_torch_explains_the_trade_off(self, monkeypatch):
        """A released build genuinely cannot do this; "No module named 'torch'"
        would read like a broken installation."""
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)

        with pytest.raises(RuntimeError, match="PyTorch is not available"):
            sweep.run(seeds=(1, 2))
