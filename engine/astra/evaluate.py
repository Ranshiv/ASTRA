"""Quantitative comparison of anomaly methods (plan sections 13 and 20).

Plan section 13 is explicit: never assume the newest deep-learning model is
automatically best, and settle it with numbers. Section 20 asks for precision,
recall, F1 and ROC-AUC — all of which need labels that do not exist yet,
because the human-in-the-loop labelling system is Phase 7.

Injection-recovery closes that gap honestly. Known anomalies of known shape
and amplitude are injected into a fraction of real light curves, and every
method is scored on how well it recovers them. The labels are then true by
construction rather than assumed.

The obvious caveat, stated rather than buried: this measures sensitivity to
the anomaly types actually injected. A method that wins here is good at
finding flares, eclipses and step changes; it is not thereby proven good at
finding phenomena nobody thought to inject, which is the discovery case the
project ultimately cares about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ANOMALY_KINDS = ("flare", "eclipse", "step", "noise_burst")


@dataclass
class InjectionResult:
    """Sequences with injected anomalies, plus the labels that came with them."""

    values: np.ndarray          # (n, 2, length)
    labels: np.ndarray          # 1 where an anomaly was injected
    kinds: list[str]            # "" for untouched rows
    identities: list[dict]

    def __len__(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict:
        return {
            "rows": len(self),
            "injected": int(self.labels.sum()),
            "kinds": {k: self.kinds.count(k) for k in ANOMALY_KINDS},
        }


@dataclass
class MethodScore:
    """One method's performance at recovering the injected anomalies."""

    name: str
    roc_auc: float
    average_precision: float
    precision_at_k: float
    recall_at_k: float
    seconds: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "roc_auc": round(self.roc_auc, 4),
            "average_precision": round(self.average_precision, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "seconds": round(self.seconds, 2),
            "note": self.note,
        }


@dataclass
class Comparison:
    methods: list[MethodScore] = field(default_factory=list)
    injection: dict = field(default_factory=dict)

    def best(self) -> MethodScore | None:
        ranked = [m for m in self.methods if np.isfinite(m.roc_auc)]
        return max(ranked, key=lambda m: m.roc_auc) if ranked else None

    def to_dict(self) -> dict:
        winner = self.best()
        return {
            "injection": self.injection,
            "methods": sorted((m.to_dict() for m in self.methods),
                              key=lambda m: -m["roc_auc"]),
            "best_method": winner.name if winner else None,
        }


def inject(sequence: np.ndarray, kind: str,
           rng: np.random.Generator, strength: float = 6.0) -> np.ndarray:
    """Add one anomaly of a known shape to a normalised sequence.

    Amplitudes are in units of the curve's own MAD, because the sequences are
    MAD-normalised. A strength of 6 is a clear but not absurd event — large
    enough to be real, small enough that recovering it is not trivial.
    """
    out = sequence.copy()
    values, mask = out[0], out[1]
    length = len(values)

    if kind == "flare":
        # Fast rise, exponential decay — the classic stellar flare profile.
        start = rng.integers(int(length * 0.1), int(length * 0.8))
        duration = max(3, int(length * rng.uniform(0.02, 0.06)))
        decay = np.exp(-np.arange(duration) / max(duration / 3.0, 1.0))
        end = min(start + duration, length)
        values[start:end] += strength * decay[:end - start]

    elif kind == "eclipse":
        # Flat-bottomed dip, as a transiting or eclipsing companion produces.
        start = rng.integers(int(length * 0.1), int(length * 0.8))
        duration = max(4, int(length * rng.uniform(0.03, 0.08)))
        end = min(start + duration, length)
        values[start:end] -= strength * 0.7

    elif kind == "step":
        # Persistent level change: often instrumental, which is why the
        # system must be able to recognise it (plan section 4).
        start = rng.integers(int(length * 0.2), int(length * 0.8))
        values[start:] += strength * 0.5

    elif kind == "noise_burst":
        # A stretch of inflated scatter, the signature of bad epochs.
        start = rng.integers(int(length * 0.1), int(length * 0.8))
        duration = max(5, int(length * rng.uniform(0.05, 0.12)))
        end = min(start + duration, length)
        values[start:end] += rng.normal(0.0, strength * 0.5, size=end - start)

    else:
        raise ValueError(f"unknown anomaly kind: {kind!r}")

    # Injected signal only counts where the curve was actually observed.
    out[0] = values * mask
    return out


def build_injected(values: np.ndarray, identities: list[dict],
                   fraction: float = 0.1, strength: float = 6.0,
                   seed: int = 42) -> InjectionResult:
    """Inject anomalies into a random subset, leaving the rest untouched."""
    rng = np.random.default_rng(seed)
    n = len(values)

    out = values.copy()
    labels = np.zeros(n, dtype=int)
    kinds = [""] * n

    if n == 0:
        return InjectionResult(out, labels, kinds, identities)

    count = max(1, int(round(n * fraction)))
    chosen = rng.choice(n, size=min(count, n), replace=False)

    for index in chosen:
        kind = str(rng.choice(ANOMALY_KINDS))
        out[index] = inject(values[index], kind, rng, strength)
        labels[index] = 1
        kinds[index] = kind

    return InjectionResult(out, labels, kinds, identities)


def score_method(name: str, scores: np.ndarray, labels: np.ndarray,
                 seconds: float = 0.0, note: str = "") -> MethodScore:
    """Standard retrieval metrics, with precision/recall at the true count."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    positives = int(labels.sum())
    if positives == 0 or positives == len(labels) or len(labels) == 0:
        return MethodScore(name, float("nan"), float("nan"), float("nan"),
                           float("nan"), seconds, "degenerate label set")

    finite = np.isfinite(scores)
    if not finite.all():
        # A method that cannot score a row must not be silently credited.
        scores = np.where(finite, scores, float(np.nanmin(scores[finite]))
                          if finite.any() else 0.0)

    # k is set to the true number of injected anomalies, so precision@k and
    # recall@k coincide and read as "of the k it ranked highest, how many
    # were real".
    top_k = np.argsort(-scores)[:positives]
    hits = int(labels[top_k].sum())

    return MethodScore(
        name=name,
        roc_auc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
        precision_at_k=hits / positives,
        recall_at_k=hits / positives,
        seconds=seconds,
        note=note,
    )


def compare_on_sequences(injection: InjectionResult,
                         include_deep: bool = True,
                         epochs: int = 20,
                         seed: int = 42) -> Comparison:
    """Run baselines and deep models over the same injected data.

    The baselines see features derived from the sequences; the deep models see
    the sequences directly. Both are evaluated on identical labels, which is
    what makes the comparison fair.
    """
    import time

    from . import anomaly
    from .featurematrix import FeatureMatrix
    from .features import FEATURE_NAMES

    comparison = Comparison(injection=injection.to_dict())
    values, labels = injection.values, injection.labels
    if len(values) < 10:
        return comparison

    # Baselines operate on summary statistics of the sequence, which is the
    # closest fair analogue of the Phase 4 feature pipeline.
    summary = sequence_summary(values)
    matrix = FeatureMatrix(
        values=summary,
        identities=injection.identities,
        feature_names=tuple(f"seq_{i}" for i in range(summary.shape[1])),
    )

    started = time.time()
    ensemble = anomaly.detect(matrix, seed=seed)
    elapsed = time.time() - started

    if ensemble.detectors:
        usable = matrix.finite_mask()
        for name, detector in ensemble.detectors.items():
            comparison.methods.append(
                score_method(f"baseline_{name}", detector.scores,
                             labels[usable], elapsed / 4))
        comparison.methods.append(
            score_method("baseline_ensemble", ensemble.consensus,
                         labels[usable], elapsed))

    if include_deep:
        comparison.methods.extend(_deep_methods(values, labels, epochs, seed))

    return comparison


def sequence_summary(values: np.ndarray) -> np.ndarray:
    """Compact statistics of each sequence, for the baseline detectors."""
    observed = values[:, 0, :]
    mask = values[:, 1, :]

    def masked(fn, default=0.0):
        out = np.empty(len(values))
        for i in range(len(values)):
            points = observed[i][mask[i] > 0]
            out[i] = fn(points) if points.size > 2 else default
        return out

    columns = [
        masked(np.std), masked(lambda x: np.max(x) - np.min(x)),
        masked(lambda x: np.percentile(x, 95) - np.percentile(x, 5)),
        masked(lambda x: np.mean(np.abs(np.diff(x)))),
        masked(lambda x: np.max(np.abs(np.diff(x)))),
        masked(lambda x: float(np.mean(x ** 3))),
        masked(lambda x: float(np.mean(x ** 4))),
        masked(np.min), masked(np.max),
        mask.mean(axis=1),
    ]
    return np.column_stack(columns)


def _deep_methods(values: np.ndarray, labels: np.ndarray,
                  epochs: int, seed: int) -> list[MethodScore]:
    """Train each deep model on the data and score reconstruction error.

    Training is unsupervised and runs on everything, injected rows included.
    That is the realistic setting: in a real campaign nobody knows which
    objects are anomalous, and a rare contaminant should not stop the model
    learning the ordinary population.

    Which kinds run depends on `values`' own channel count: the three
    convolutional/attention models share the 2-channel (value, mask)
    contract from `tensors.py`'s "time"/"season"/"phase" modes, while
    `neural_ode` needs the 3-channel (value, mask, time-delta) contract only
    the "irregular" mode produces -- the two representations cannot be
    stacked into one array, so a caller wanting all four methods compared
    calls `compare_on_sequences` twice, once per representation, against
    labels built the same way (`build_injected`) on each.
    """
    import time

    try:
        import torch  # noqa: F401
    except ImportError:
        return [MethodScore("deep_autoencoder", float("nan"), float("nan"),
                            float("nan"), float("nan"),
                            note="PyTorch not installed")]

    from . import tensors, train

    results: list[MethodScore] = []
    train_values, val_values, _, _ = tensors.train_test_split(
        tensors.SequenceBatch(values=values,
                              identities=[{}] * len(values),
                              length=values.shape[-1]),
        test_fraction=0.2, seed=seed,
    )

    channels = values.shape[1] if values.ndim == 3 else 2
    kinds = ("neural_ode",) if channels == 3 else ("autoencoder", "vae", "transformer")
    for kind in kinds:
        started = time.time()
        try:
            cfg = train.TrainConfig(
                kind=kind, epochs=epochs, seed=seed,
                model=train.ModelConfig(length=values.shape[-1]),
            )
            report = train.train(train_values, val_values, cfg, name="eval")
            model, _ = train.load_model(report.checkpoint)
            scores = train.reconstruction_scores(model, values)
            note = f"{report.device}, {report.epochs_run} epochs, " \
                   f"batch {report.batch_size}x{report.accumulation_steps}"
            results.append(score_method(f"deep_{kind}", scores, labels,
                                        time.time() - started, note))
        except Exception as exc:  # noqa: BLE001 - report, do not abort the study
            results.append(MethodScore(f"deep_{kind}", float("nan"),
                                       float("nan"), float("nan"),
                                       float("nan"), time.time() - started,
                                       note=f"failed: {exc}"))

    return results


def compare_cadence_representations(survey: str | None = None, *,
                                    length: int = 128, limit: int = 10_000,
                                    fraction: float = 0.1, strength: float = 6.0,
                                    epochs: int = 20, seed: int = 42,
                                    root=None) -> dict:
    """The two `compare_on_sequences` calls the "gaps are masked, not
    modelled" limitation named as its own missing evidence: build the SAME
    injected-anomaly study twice on the same stored curves, once under
    `tensors.build(mode="time")` (irregular cadence resampled onto a
    uniform grid and MASKED -- the three convolutional/attention models'
    contract) and once under `mode="irregular"` (the raw time-delta
    between real observations kept as a third channel -- `neural_ode`'s
    own contract, `models.make_neural_ode`'s whole reason to exist). Both
    runs share `fraction`/`strength`/`seed`, so the same anomalies land at
    the same indices in both -- the comparison isolates the cadence
    REPRESENTATION, not a second confound from re-randomised injections.

    Reuses `tensors.build`/`build_injected`/`compare_on_sequences`
    UNCHANGED; this function only calls them twice and reports both
    results side by side. Like every other research-evidence function in
    this codebase, this is diagnostic evidence only -- it does not choose
    a cadence representation for production and is not wired into
    `scoring.WEIGHTS`/`evidence.py`.
    """
    from . import tensors

    result: dict = {"survey": survey, "length": length, "fraction": fraction,
                    "strength": strength, "seed": seed, "modes": {}}
    for mode in ("time", "irregular"):
        batch = tensors.build(survey=survey, length=length, limit=limit, root=root, mode=mode)
        if len(batch) < 10:
            result["modes"][mode] = {"ready": False, "reason": "fewer than 10 usable sequences",
                                     "rows": len(batch)}
            continue
        injection = build_injected(batch.values, batch.identities,
                                   fraction=fraction, strength=strength, seed=seed)
        comparison = compare_on_sequences(injection, include_deep=True,
                                          epochs=epochs, seed=seed)
        result["modes"][mode] = {"ready": True, **comparison.to_dict()}

    ready_modes = {mode: payload for mode, payload in result["modes"].items()
                  if payload.get("ready")}
    result["comparable"] = len(ready_modes) == 2
    if result["comparable"]:
        # neural_ode only ever runs under "irregular" (see _deep_methods),
        # so the cadence-representation-specific comparison is neural_ode's
        # own PR-AUC there against the best 2-channel model's PR-AUC under
        # "time" -- not a claim that any two individual models are
        # otherwise comparable, only that this is what each cadence
        # representation's best available detector achieves.
        time_best = max((m for m in ready_modes["time"]["methods"]
                         if np.isfinite(m["average_precision"])),
                        key=lambda m: m["average_precision"], default=None)
        irregular_neural_ode = next(
            (m for m in ready_modes["irregular"]["methods"] if m["name"] == "deep_neural_ode"),
            None)
        result["cadence_shift_summary"] = {
            "time_mode_best_method": time_best["name"] if time_best else None,
            "time_mode_best_average_precision": time_best["average_precision"] if time_best else None,
            "irregular_mode_neural_ode_average_precision": (
                irregular_neural_ode["average_precision"] if irregular_neural_ode else None),
        }
    return result
