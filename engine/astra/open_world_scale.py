"""Resumable real-data scale runner for the open-world generalisation study
(backlog item 14, gap 2).

`open_world_eval.evaluate_open_world_generalization` was validated on small
synthetic-stand-in data. This module runs the same real mechanism against
real data at a modest but genuine scale: real ALeRCE-classified transients
for the held-out set, real locally-stored light curves as the design
population, a real diffusion generator trained on real patches.

Checkpointed per stage (held-out set assembly, generator training, the
final generalisation result), the same resumable-study discipline
`stageb.py`'s `run()` already establishes for exactly the same reason: a
real run against a live external service and real training time must
survive an interruption without redoing completed, expensive work.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCALE_SCHEMA_VERSION = 1
DEFAULT_SEEDS: tuple[int, ...] = (17, 29, 43)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_scale_study(*, class_names: tuple[str, ...] = ("SNIa",),
                    design_survey: str = "ZTF", design_length: int = 256,
                    design_limit: int = 2000, limit_per_class: int = 20,
                    negative_count: int | None = None,
                    seeds: tuple[int, ...] = DEFAULT_SEEDS,
                    fraction: float = 0.1, epochs: int = 15,
                    patch_length: int = 32, diffusion_epochs: int = 30,
                    root: Path | None = None, checkpoint: Path | None = None,
                    connector=None) -> dict:
    """Run or resume the real-data open-world generalisation study.

    `connector` defaults to a real `ALeRCEConnector` (opt-in, credential-
    free, live-verified this session); tests inject a fake one, the same
    convention `open_world_eval.assemble_held_out_set` already uses.
    """
    from . import config, evaluate, experiment, tensors
    from . import open_world_eval as owe
    from . import open_world_injection as owi
    from . import diffusion as diff
    from . import diffusion_train as diff_train

    unique_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if len(unique_seeds) < 2:
        raise ValueError("the generalisation study needs at least two independent seeds")

    workspace = root or config.PATHS.projects
    checkpoint_path = checkpoint or (workspace / "results" / "open-world" / "scale_study.json")

    configuration = {
        "class_names": list(class_names), "design_survey": design_survey,
        "design_length": design_length, "design_limit": design_limit,
        "limit_per_class": limit_per_class, "negative_count": negative_count,
        "seeds": list(unique_seeds), "fraction": fraction, "epochs": epochs,
        "patch_length": patch_length, "diffusion_epochs": diffusion_epochs,
    }

    def work() -> dict:
        state = _load_checkpoint(checkpoint_path)
        compatible = state.get("schema_version") == SCALE_SCHEMA_VERSION \
            and state.get("configuration") == configuration
        stages: dict[str, Any] = state.get("stages", {}) if compatible else {}

        design_batch = tensors.build(survey=design_survey, length=design_length,
                                     limit=design_limit)
        if len(design_batch) < 20:
            return {"ready": False,
                   "reason": f"only {len(design_batch)} usable design sequences; need at least 20",
                   "checkpoint": str(checkpoint_path)}

        # Stage 1: real held-out set.
        if "held_out" not in stages:
            held_out = owe.assemble_held_out_set(
                class_names, design_batch.values, design_batch.identities,
                length=design_length, limit_per_class=limit_per_class,
                negative_count=negative_count, connector=connector)
            if len(held_out) == 0 or len(np.unique(held_out.labels)) < 2:
                return {"ready": False,
                       "reason": "held-out set has no positives or no negatives "
                                 "(real ALeRCE query likely returned nothing usable)",
                       "checkpoint": str(checkpoint_path)}
            stages["held_out"] = {
                "values": held_out.values.tolist(), "labels": held_out.labels.tolist(),
                "identities": held_out.identities,
                "note": held_out.note, "summary": held_out.to_dict(),
            }
            _atomic_json(checkpoint_path, {"schema_version": SCALE_SCHEMA_VERSION,
                                           "configuration": configuration, "stages": stages})

        held_out_stage = stages["held_out"]
        from .open_world_eval import HeldOutSet
        held_out = HeldOutSet(
            values=np.asarray(held_out_stage["values"], dtype=np.float32),
            labels=np.asarray(held_out_stage["labels"], dtype=int),
            identities=held_out_stage["identities"],
            note=held_out_stage["note"],
        )

        # Stage 2: real diffusion generator, trained on real patches. Only
        # the NEGATIVE half of the held-out set is drawn from the design
        # population (positives come from a separate real ALeRCE fetch, so
        # they were never in local ZTF storage to begin with) -- excluding
        # those specific negative object IDs is what actually prevents the
        # generator from training on data reused for evaluation.
        if "generator_checkpoint" not in stages:
            held_out_ids = {
                identity.get("object_id") for identity, label
                in zip(held_out.identities, held_out.labels) if label == 0
            }
            patches = owi.extract_real_patches(
                survey=design_survey, patch_length=patch_length,
                sequence_length=design_length, exclude_object_ids=held_out_ids)
            if len(patches) < 20:
                return {"ready": False,
                       "reason": f"only {len(patches)} real patches extracted; need at least 20",
                       "checkpoint": str(checkpoint_path)}
            cut = max(1, int(round(len(patches) * 0.8)))
            gen_cfg = diff.DiffusionConfig(patch_length=patch_length, epochs=diffusion_epochs)
            report = diff_train.train_diffusion(
                patches[:cut], patches[cut:], gen_cfg,
                checkpoint_dir=workspace / "results" / "open-world", name="scale_generator")
            stages["generator_checkpoint"] = report.checkpoint
            stages["generator_config"] = gen_cfg.to_dict()
            _atomic_json(checkpoint_path, {"schema_version": SCALE_SCHEMA_VERSION,
                                           "configuration": configuration, "stages": stages})

        generator, _ = diff_train.load_diffusion_model(stages["generator_checkpoint"])
        gen_cfg = diff.DiffusionConfig(**{
            **stages["generator_config"],
            "channels": tuple(stages["generator_config"]["channels"]),
        })

        # Stage 3: the real two-arm generalisation comparison.
        if "result" not in stages:
            result = owe.evaluate_open_world_generalization(
                design_batch.values, design_batch.identities, held_out,
                diffusion_generator=generator, diffusion_cfg=gen_cfg,
                fraction=fraction, seeds=unique_seeds, epochs=epochs)
            stages["result"] = result
            _atomic_json(checkpoint_path, {"schema_version": SCALE_SCHEMA_VERSION,
                                           "configuration": configuration, "stages": stages})

        return {
            "ready": True,
            "checkpoint": str(checkpoint_path),
            "held_out_summary": held_out_stage["summary"],
            "generator_checkpoint": stages["generator_checkpoint"],
            "result": stages["result"],
        }

    record = experiment.run(
        "open_world_scale_study", configuration, work, seed=unique_seeds[0],
        root=workspace,
        notes="Real-data scale run of the open-world vs. closed-world injection "
              "generalisation comparison, on real ALeRCE-classified held-out "
              "transients. Never asserts a winner; reports both arms honestly.")
    return {"experiment_id": record.provenance.experiment_id, **record.results}
