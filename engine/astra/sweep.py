"""Hyperparameter search for the deep models (plan section 13).

Learning rate, latent dimension, channel widths and KL weight have been fixed
defaults since Phase 5, with no tuning study behind them. This module measures
them instead of assuming them.

Three design choices are deliberate and worth stating, because each guards
against a way this kind of study usually misleads:

**Multi-seed by default.** The first injection-recovery run in this project
scored PCA reconstruction (0.742) above both deep models (0.622, 0.609) on 54
sequences. That is a real measurement and a bad conclusion: at n=54 it mostly
says the deep models were data-starved. A single seed cannot separate "this
configuration is better" from "this injection draw favoured it", so every trial
is scored across independent seeds and reported with an interval.

**Capacity is not chosen by the memory limit.** The autoencoder is 93,921
parameters and peaks at 38 MB of VRAM against roughly 2.3 GB free — about 1.5%
of the budget. The 4 GB card is not the binding constraint for these models, so
the grid explores capacity in both directions. Plan section 13's rule cuts both
ways: bigger is not automatically better either.

**Overlapping intervals are reported as overlapping.** `best()` refuses to name
a winner when the top two configurations' seed intervals overlap. Ranking by
mean alone would manufacture a decision the data does not support.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

# Explored in both directions from the current defaults, which are latent 16,
# channels (16, 32, 64), lr 1e-3, KL weight 1.0.
DEFAULT_GRID: dict[str, tuple] = {
    "latent_dim": (8, 16, 32),
    "channels": ((16, 32), (16, 32, 64), (32, 64, 128)),
    "learning_rate": (3e-4, 1e-3, 3e-3),
}

# Only meaningful for the VAE; skipped for the other kinds rather than recorded
# as a dimension that did nothing.
KL_WEIGHT_GRID: tuple[float, ...] = (0.1, 1.0, 4.0)

# The patch transformer is parameterised differently: latent_dim and channels
# do not describe it, and sweeping them would vary nothing while tripling the
# runtime. Attention is O(L^2) in memory, so the grid stays modest on purpose.
TRANSFORMER_GRID: dict[str, tuple] = {
    "transformer_dim": (32, 64, 128),
    "transformer_layers": (2, 4),
    "patch_size": (8, 16, 32),
    "learning_rate": (3e-4, 1e-3),
}

DEFAULT_SEEDS: tuple[int, ...] = (17, 29, 43)


@dataclass
class TrialResult:
    """One configuration, scored across every seed it survived."""

    parameters: dict
    roc_auc: list[float] = field(default_factory=list)
    average_precision: list[float] = field(default_factory=list)
    model_parameters: int = 0
    seconds: float = 0.0
    note: str = ""

    @property
    def scored_seeds(self) -> int:
        return len(self.roc_auc)

    def _summary(self, values: list[float]) -> dict | None:
        finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if not len(finite):
            return None
        return {
            "mean": round(float(np.mean(finite)), 4),
            "std": (round(float(np.std(finite, ddof=1)), 4)
                    if len(finite) > 1 else 0.0),
            "ci95": [round(float(np.quantile(finite, 0.025)), 4),
                     round(float(np.quantile(finite, 0.975)), 4)],
        }

    @property
    def mean_auc(self) -> float:
        summary = self._summary(self.roc_auc)
        return summary["mean"] if summary else float("nan")

    def to_dict(self) -> dict:
        return {
            "parameters": self.parameters,
            "model_parameters": self.model_parameters,
            "scored_seeds": self.scored_seeds,
            "roc_auc": self._summary(self.roc_auc),
            "average_precision": self._summary(self.average_precision),
            "seconds": round(self.seconds, 2),
            "note": self.note,
        }


@dataclass
class SweepResult:
    kind: str
    seeds: list[int] = field(default_factory=list)
    trials: list[TrialResult] = field(default_factory=list)
    rows: int = 0
    note: str = ""

    def ranked(self) -> list[TrialResult]:
        scored = [t for t in self.trials if np.isfinite(t.mean_auc)]
        return sorted(scored, key=lambda t: -t.mean_auc)

    def best(self) -> TrialResult | None:
        """The winner, or None when the seed intervals do not separate.

        Refusing to pick is the point. With three seeds the intervals are wide,
        and naming a winner whose interval overlaps the runner-up's would turn
        injection-draw noise into a recommendation.
        """
        ranked = self.ranked()
        if not ranked:
            return None
        if len(ranked) == 1:
            return ranked[0]

        first, second = ranked[0], ranked[1]
        first_summary = first.to_dict()["roc_auc"]
        second_summary = second.to_dict()["roc_auc"]
        if not (first_summary and second_summary):
            return None
        # Separated only if the leader's lower bound clears the runner-up's upper.
        if first_summary["ci95"][0] > second_summary["ci95"][1]:
            return first
        return None

    def to_dict(self) -> dict:
        winner = self.best()
        ranked = self.ranked()
        return {
            "kind": self.kind,
            "seeds": self.seeds,
            "rows": self.rows,
            "trials": [t.to_dict() for t in ranked]
                      + [t.to_dict() for t in self.trials
                         if not np.isfinite(t.mean_auc)],
            "best": winner.parameters if winner else None,
            "separated": winner is not None,
            "note": self.note or (
                "" if winner else
                "No configuration separates from the runner-up across these "
                "seeds; the ranking is not evidence of a better setting."),
        }


def grid(kind: str = "autoencoder",
         overrides: dict[str, tuple] | None = None) -> list[dict]:
    """Every configuration to try, as a list of parameter dictionaries.

    Each kind gets the dimensions that actually describe it. Sweeping
    `latent_dim` over a patch transformer would vary nothing while tripling the
    runtime, and reporting it as a searched dimension would be misleading.
    """
    base = TRANSFORMER_GRID if kind == "transformer" else DEFAULT_GRID
    space = {**base, **(overrides or {})}
    if kind == "vae" and "kl_weight" not in space:
        space["kl_weight"] = KL_WEIGHT_GRID

    names = sorted(space)
    return [dict(zip(names, values))
            for values in itertools.product(*(space[name] for name in names))]


def _score_trial(parameters: dict, kind: str, batch, seeds: tuple[int, ...],
                 fraction: float, strength: float, epochs: int) -> TrialResult:
    """Train and score one configuration on every seed."""
    from . import evaluate, tensors, train

    result = TrialResult(parameters=dict(parameters))
    started = time.time()

    for seed in seeds:
        injection = evaluate.build_injected(batch.values, batch.identities,
                                            fraction=fraction,
                                            strength=strength, seed=seed)
        train_values, val_values, _, _ = tensors.train_test_split(
            tensors.SequenceBatch(values=injection.values,
                                  identities=injection.identities,
                                  length=batch.length),
            test_fraction=0.2, seed=seed)

        defaults = train.ModelConfig()
        model_config = train.ModelConfig(
            length=batch.length,
            latent_dim=int(parameters.get("latent_dim", defaults.latent_dim)),
            channels=tuple(parameters.get("channels", defaults.channels)),
            patch_size=int(parameters.get("patch_size", defaults.patch_size)),
            transformer_dim=int(parameters.get("transformer_dim",
                                               defaults.transformer_dim)),
            transformer_layers=int(parameters.get("transformer_layers",
                                                  defaults.transformer_layers)),
        )
        config = train.TrainConfig(
            kind=kind, epochs=epochs, seed=seed,
            learning_rate=float(parameters.get("learning_rate", 1e-3)),
            kl_weight=float(parameters.get("kl_weight", 1.0)),
            model=model_config,
        )

        try:
            report = train.train(train_values, val_values, config, name="sweep")
            model, _ = train.load_model(report.checkpoint)
            scores = train.reconstruction_scores(model, injection.values)
            scored = evaluate.score_method(kind, scores, injection.labels)
        except Exception as exc:  # noqa: BLE001 - one bad config must not end the sweep
            result.note = f"{type(exc).__name__}: {exc}"
            continue

        result.model_parameters = report.parameters
        result.roc_auc.append(scored.roc_auc)
        result.average_precision.append(scored.average_precision)

    result.seconds = time.time() - started
    return result


def run(kind: str = "autoencoder",
        survey: str | None = None,
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        fraction: float = 0.1,
        strength: float = 6.0,
        epochs: int = 20,
        limit: int = 10_000,
        overrides: dict[str, tuple] | None = None,
        mode: str = "time") -> SweepResult:
    """Search the grid, scoring each configuration by injection recovery.

    `survey` should normally be set. Pooling ZTF and TESS sequences lets the
    models separate by instrument rather than by behaviour, and a sweep run on
    a pooled population would tune for that separation.
    """
    from . import evaluate, tensors  # noqa: F401 - evaluate used by _score_trial

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        from .rpc import DEEP_UNAVAILABLE

        raise RuntimeError(DEEP_UNAVAILABLE) from exc

    if len(seeds) < 2:
        raise ValueError("a sweep needs at least two seeds to report an interval")

    batch = tensors.build(survey=survey, limit=limit, mode=mode)
    if len(batch) < 20:
        return SweepResult(kind=kind, seeds=list(seeds), rows=len(batch),
                           note=f"only {len(batch)} usable sequences; "
                                f"need at least 20")

    result = SweepResult(kind=kind, seeds=list(seeds), rows=len(batch))
    for parameters in grid(kind, overrides):
        result.trials.append(_score_trial(parameters, kind, batch, tuple(seeds),
                                          fraction, strength, epochs))
    return result


def run_recorded(kind: str = "autoencoder", survey: str | None = None,
                 seeds: tuple[int, ...] = DEFAULT_SEEDS,
                 mode: str = "time", **kwargs) -> dict:
    """Run a sweep and persist it as a reproducible experiment."""
    from . import experiment

    def work() -> dict:
        return run(kind=kind, survey=survey, seeds=seeds, mode=mode,
                   **kwargs).to_dict()

    record = experiment.run(
        "hyperparameter_sweep",
        {"kind": kind, "survey": survey, "seeds": list(seeds),
         "resample_mode": mode, **kwargs},
        work, seed=seeds[0],
        notes="Grid search over deep-model capacity and optimisation, scored "
              "by injection recovery across independent seeds. A winner is "
              "reported only when its seed interval separates from the "
              "runner-up's.")
    return {"experiment_id": record.provenance.experiment_id, **record.results}
