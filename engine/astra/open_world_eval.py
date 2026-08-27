"""The metric backlog item 14 actually names: recovery on held-out REAL
transients not used to design injections.

`evaluate.build_injected()`'s labels are true by construction because the
anomaly was placed there deliberately -- that is a self-consistent measure
of recovering what was injected, never a measure of recovering something
real the detector never saw. This module builds the genuinely different
comparison: real ALeRCE-classified transient light curves (`surveys.alerce.
ALeRCEConnector.query_classified_objects`, real photometry via the same
connector's `fetch_light_curves`) held out from BOTH the diffusion
generator's training population (`open_world_injection.
extract_real_patches`'s `exclude_object_ids`) and the detector's own
training population, then asks whether a detector trained on synthetic
injections -- hand-designed (`evaluate.build_injected`) or generative
(`open_world_injection.build_injected_open_world`) -- actually flags those
real, never-seen transients as anomalous.

Negatives in the held-out set are drawn from the design population itself
(real, already-locally-stored curves), NOT individually verified as
non-transient -- an honest caveat carried on every `HeldOutSet`, the same
"state the limitation, don't gloss over it" discipline `artifact.py`'s own
rejection of the SNAD dataset already models for this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_SEEDS: tuple[int, ...] = (17, 29, 43)


@dataclass
class HeldOutSet:
    values: np.ndarray          # (n, 2, length)
    labels: np.ndarray          # 1 = real classified transient, 0 = presumed normal
    identities: list[dict]
    note: str = ""

    def __len__(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict:
        return {
            "rows": len(self),
            "positive": int(self.labels.sum()),
            "note": self.note,
        }


def assemble_held_out_set(class_names: tuple[str, ...], design_values: np.ndarray,
                          design_identities: list[dict], *, length: int = 256,
                          limit_per_class: int = 20, min_probability: float = 0.5,
                          negative_count: int | None = None, seed: int = 42,
                          connector=None) -> HeldOutSet:
    """Real ALeRCE-classified positives plus design-population negatives.

    `connector` defaults to a real `ALeRCEConnector` but accepts an
    injected fake for tests (mirroring every other connector test's
    `monkeypatch.setattr(netclient, "get", ...)` convention one layer up:
    this function's own tests inject a fake connector object directly,
    since it calls two connector methods, not `netclient.get` itself).
    """
    from . import tensors
    from .surveys.alerce import ALeRCEConnector

    connector = connector or ALeRCEConnector()
    rng = np.random.default_rng(seed)

    positive_sequences: list[np.ndarray] = []
    positive_identities: list[dict] = []
    for class_name in class_names:
        try:
            sources = connector.query_classified_objects(
                class_name, min_probability=min_probability, limit=limit_per_class)
        except Exception:  # noqa: BLE001 - one bad class must not abort assembly
            continue
        for source in sources:
            try:
                curves = connector.fetch_light_curves(source)
            except Exception:  # noqa: BLE001 - one bad object must not abort assembly
                continue
            for curve in curves:
                sequence = tensors.resample(curve, length=length)
                if sequence is None:
                    continue
                positive_sequences.append(sequence)
                positive_identities.append({
                    "object_id": source.object_id, "survey": connector.name,
                    "class_name": class_name, "band": curve.band,
                })
                break  # one representative band per real object

    n_positive = len(positive_sequences)
    negative_count = negative_count if negative_count is not None else n_positive
    n_design = len(design_values)
    negative_indices = (rng.choice(n_design, size=min(negative_count, n_design), replace=False)
                        if n_design and negative_count > 0 else np.empty(0, dtype=int))

    positive_block = (np.stack(positive_sequences).astype(np.float32) if positive_sequences
                      else np.empty((0, 2, length), dtype=np.float32))
    negative_block = design_values[negative_indices].astype(np.float32)

    values = np.concatenate([positive_block, negative_block]) if (
        len(positive_block) or len(negative_block)) else np.empty((0, 2, length), dtype=np.float32)
    labels = np.concatenate([
        np.ones(n_positive, dtype=int),
        np.zeros(len(negative_indices), dtype=int),
    ])
    identities = positive_identities + [design_identities[i] for i in negative_indices]

    return HeldOutSet(
        values=values, labels=labels, identities=identities,
        note=("negatives are real, already-locally-stored curves not individually "
             "verified as non-transient -- presumed normal by construction of the "
             "query (they were not returned by a transient classifier), not confirmed"),
    )


def held_out_recovery(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Thin wrapper over `evaluate.score_method` -- no reimplementation."""
    from . import evaluate

    return evaluate.score_method("held_out_recovery", scores, labels).to_dict()


def _summary(values: list[float]) -> dict | None:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(finite):
        return None
    return {
        "mean": round(float(np.mean(finite)), 4),
        "std": round(float(np.std(finite, ddof=1)), 4) if len(finite) > 1 else 0.0,
        "ci95": [round(float(np.quantile(finite, 0.025)), 4),
                 round(float(np.quantile(finite, 0.975)), 4)],
        "n": len(finite),
    }


def evaluate_open_world_generalization(design_values: np.ndarray, design_identities: list[dict],
                                       held_out: HeldOutSet, *, diffusion_generator,
                                       diffusion_cfg, fraction: float = 0.1,
                                       seeds: tuple[int, ...] = DEFAULT_SEEDS,
                                       epochs: int = 15, model_config=None) -> dict:
    """Does open-world (generative) injection training generalise to real
    held-out transients better than closed-world (hand-designed) injection
    training does?

    Both arms train the SAME architecture (a plain reconstruction
    autoencoder, via `train.train()`/`models.ModelConfig` -- the existing,
    unmodified deep-model training path `evaluate._deep_methods` already
    uses) the same way, differing ONLY in which injection scheme built
    their training population, then both are scored via
    `train.reconstruction_scores()` on the exact same held-out real set.
    Multi-seed mean/CI95, reported honestly -- NEVER asserts which arm
    wins, the same restraint `sweep.SweepResult.best()`/
    `stellar_manifold_eval.evaluate_manifold_contribution` already apply
    elsewhere in this codebase.
    """
    from . import evaluate, tensors
    from . import open_world_injection as owi
    from . import train as train_mod

    if len(seeds) < 2:
        raise ValueError("evaluate_open_world_generalization needs at least two seeds")
    if len(held_out) == 0 or len(np.unique(held_out.labels)) < 2:
        raise ValueError("held_out must contain both positive and negative rows")

    length = design_values.shape[-1]
    closed_scores: list[float] = []
    open_scores: list[float] = []

    for seed in seeds:
        closed_injection = evaluate.build_injected(
            design_values, design_identities, fraction=fraction, seed=seed)
        open_injection = owi.build_injected_open_world(
            design_values, design_identities, diffusion_generator, diffusion_cfg,
            fraction=fraction, seed=seed)

        for injection, bucket in ((closed_injection, closed_scores),
                                  (open_injection, open_scores)):
            train_values, val_values, _, _ = tensors.train_test_split(
                tensors.SequenceBatch(values=injection.values, identities=injection.identities,
                                      length=length),
                test_fraction=0.2, seed=seed)
            cfg = train_mod.TrainConfig(
                kind="autoencoder", epochs=epochs, seed=seed,
                model=model_config or train_mod.ModelConfig(length=length))
            try:
                report = train_mod.train(train_values, val_values, cfg, name="open_world_eval")
                model, _ = train_mod.load_model(report.checkpoint)
                scores = train_mod.reconstruction_scores(model, held_out.values)
                result = evaluate.score_method("open_world_eval", scores, held_out.labels)
                bucket.append(result.roc_auc)
            except Exception:  # noqa: BLE001 - one bad seed must not abort the study
                bucket.append(float("nan"))

    return {
        "closed_world": _summary(closed_scores),
        "open_world": _summary(open_scores),
        "held_out": held_out.to_dict(),
    }
