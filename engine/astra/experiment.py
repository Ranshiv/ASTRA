"""Experiment records and reproducibility (plan sections 19 and 37).

Section 19 lists what every experiment must store: an identifier, dataset
version, model version, feature version, preprocessing version,
hyperparameters, hardware, execution time, results and random seeds. Section 37
adds that a researcher must be able to reopen an old experiment and reproduce
it.

The awkward part is code version. This project is not a git repository, so
there is no commit to record. Rather than store nothing, the engine's own
source is hashed: every `.py` file under `astra/` is read in sorted order and
digested. That gives a content-addressed code version which is arguably
stronger than a commit hash, because it changes if and only if the code that
ran actually changed — uncommitted edits included.

`verify` re-derives the whole provenance and reports what drifted, so
"reproducible" is a checkable claim rather than an assertion.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config, features as features_mod, hardware

EXPERIMENT_SCHEMA_VERSION = 1

# Version of the preprocessing chain.  The contract is deliberately data,
# rather than a comment that must be remembered when code changes.  Any change
# to a value here changes the schema hash and makes old results incomparable.
PREPROCESSING_VERSION = 2
PREPROCESSING_CONTRACT = {
    "resampling": {"kind": "uniform_grid", "length": 2048,
                   "gap_policy": "mask_interpolated"},
    "normalization": {"kind": "mad", "center": "median",
                      "scale_floor": "1e-8"},
    "channels": {"value": "normalized_value", "validity": "finite_observation"},
    "time": {"frame": "BJD_TDB", "encoding": "relative_days"},
}


# How each resampling mode lays a curve onto the fixed grid. The mode is part
# of the preprocessing contract, so selecting one changes the schema hash by
# construction. That matters: this file previously required PREPROCESSING_VERSION
# to be bumped by hand when resampling changed, and nothing enforced it, so a
# silent change would have made old and new results silently incomparable.
RESAMPLING_CONTRACTS = {
    "time": {"kind": "uniform_grid", "length": 2048,
             "gap_policy": "mask_interpolated"},
    "season": {"kind": "season_segmented_grid", "length": 2048,
               "gap_policy": "segment_on_observing_gap",
               "allocation": "proportional_to_observation_count"},
    "phase": {"kind": "phase_folded_grid", "length": 2048,
              "gap_policy": "fold_on_credible_period",
              "requires": "period_snr>=5"},
}


def preprocessing_schema_hash(mode: str = "time") -> str:
    payload = {
        "version": PREPROCESSING_VERSION,
        "contract": {**PREPROCESSING_CONTRACT,
                     "resampling": RESAMPLING_CONTRACTS.get(
                         mode, PREPROCESSING_CONTRACT["resampling"])},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def preprocessing_contract(mode: str = "time") -> dict:
    """Return a JSON-safe copy of the active preprocessing contract."""
    return json.loads(json.dumps({
        "version": PREPROCESSING_VERSION,
        "resample_mode": mode,
        "contract": {**PREPROCESSING_CONTRACT,
                     "resampling": RESAMPLING_CONTRACTS.get(
                         mode, PREPROCESSING_CONTRACT["resampling"])},
        "schema_hash": preprocessing_schema_hash(mode),
    }))


@dataclass
class Provenance:
    """Everything needed to know what produced a result."""

    experiment_id: str
    created_utc: str
    code_version: str
    feature_version: int
    preprocessing_version: int
    code_revision: str | None = None
    feature_schema_hash: str = ""
    preprocessing_schema_hash: str = ""
    dataset_hash: str | None = None
    dataset_id: str | None = None
    model_version: str | None = None
    seed: int = 42
    hardware: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Experiment:
    """One recorded run: what was configured, what came out, and how long."""

    provenance: Provenance
    kind: str = "generic"
    configuration: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    runtime_seconds: float = 0.0
    notes: str = ""
    schema_version: int = EXPERIMENT_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "provenance": self.provenance.to_dict(),
            "configuration": self.configuration,
            "results": self.results,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "notes": self.notes,
        }


def code_version(engine_root: Path | None = None) -> str:
    """Content hash of the engine source.

    Sorted so the digest is stable across filesystems, and text is read as
    bytes so line-ending differences do not silently change the version.
    """
    engine_root = engine_root or Path(__file__).parent
    digest = hashlib.sha256()

    for path in sorted(engine_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(engine_root).as_posix().encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            continue

    return digest.hexdigest()[:16]


def code_revision(engine_root: Path | None = None) -> str | None:
    """Return the containing Git commit when the checkout has one.

    ASTRA remains usable from source archives and packaged sidecars, so a Git
    checkout is an enhancement to provenance, not a runtime requirement.
    """
    root = (engine_root or Path(__file__).parent).resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def model_version(checkpoint_path: str | Path | None) -> str | None:
    """Content hash of a trained checkpoint, so weights are pinned too."""
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def capture_environment() -> dict:
    """Interpreter, platform and library versions."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for module in ("numpy", "scipy", "sklearn", "astropy", "astroquery",
                   "lightkurve", "pyarrow", "torch"):
        try:
            env[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - absence is recorded, not fatal
            env[module] = "absent"
    return env


def next_experiment_id(root: Path | None = None) -> str:
    """Sequential identifier, in the `#0241` style of plan section 19."""
    existing = list_experiments(root)
    numbers = []
    for entry in existing:
        raw = str(entry.get("experiment_id", ""))
        if raw.startswith("EXP-") and raw[4:].isdigit():
            numbers.append(int(raw[4:]))
    return f"EXP-{(max(numbers) + 1 if numbers else 1):04d}"


def create(kind: str, configuration: dict, seed: int = 42,
           dataset_hash: str | None = None, dataset_id: str | None = None,
           checkpoint_path: str | Path | None = None,
           root: Path | None = None) -> Experiment:
    """Open a new experiment record with provenance captured up front.

    A study that resampled in a non-default mode records `resample_mode` in its
    configuration; the preprocessing schema hash follows from it automatically,
    so `verify` reports the difference instead of two incomparable runs looking
    alike.
    """
    resample_mode = str(configuration.get("resample_mode") or "time")
    provenance = Provenance(
        experiment_id=next_experiment_id(root),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        code_version=code_version(),
        code_revision=code_revision(),
        feature_version=features_mod.FEATURE_VERSION,
        preprocessing_version=PREPROCESSING_VERSION,
        feature_schema_hash=features_mod.schema_hash(),
        preprocessing_schema_hash=preprocessing_schema_hash(resample_mode),
        dataset_hash=dataset_hash,
        dataset_id=dataset_id,
        model_version=model_version(checkpoint_path),
        seed=seed,
        hardware=hardware.select_device().to_dict(),
        environment=capture_environment(),
    )
    return Experiment(provenance=provenance, kind=kind,
                      configuration=configuration)


def run(kind: str, configuration: dict, work, seed: int = 42,
        dataset_hash: str | None = None, notes: str = "",
        root: Path | None = None) -> Experiment:
    """Run `work()`, timing it and recording the result as an experiment.

    A failing run is still recorded. An experiment that crashed is a result —
    losing it would mean repeating the same mistake.
    """
    record = create(kind, configuration, seed=seed,
                    dataset_hash=dataset_hash, root=root)
    started = time.time()
    try:
        record.results = work() or {}
    except Exception as exc:  # noqa: BLE001 - recorded rather than swallowed
        record.results = {"error": str(exc), "failed": True}
        record.notes = notes
        record.runtime_seconds = time.time() - started
        save(record, root)
        raise
    record.runtime_seconds = time.time() - started
    record.notes = notes
    save(record, root)
    return record


def experiment_path(experiment_id: str, root: Path | None = None) -> Path:
    root = root or config.PATHS.projects
    return root / "experiments" / f"{experiment_id}.json"


def save(record: Experiment, root: Path | None = None) -> Path:
    path = experiment_path(record.provenance.experiment_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def load(experiment_id: str, root: Path | None = None) -> Experiment:
    payload = json.loads(
        experiment_path(experiment_id, root).read_text(encoding="utf-8"))
    return Experiment(
        provenance=Provenance(**payload["provenance"]),
        kind=payload.get("kind", "generic"),
        configuration=payload.get("configuration", {}),
        results=payload.get("results", {}),
        runtime_seconds=payload.get("runtime_seconds", 0.0),
        notes=payload.get("notes", ""),
        schema_version=payload.get("schema_version", 1),
    )


def list_experiments(root: Path | None = None) -> list[dict]:
    root = root or config.PATHS.projects
    directory = root / "experiments"
    if not directory.exists():
        return []

    listing = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        provenance = payload.get("provenance", {})
        listing.append({
            "experiment_id": provenance.get("experiment_id"),
            "kind": payload.get("kind"),
            "created_utc": provenance.get("created_utc"),
            "code_version": provenance.get("code_version"),
            "runtime_seconds": payload.get("runtime_seconds"),
            "failed": bool(payload.get("results", {}).get("failed")),
        })
    return listing


def verify(experiment_id: str, root: Path | None = None) -> dict:
    """Report what has drifted since an experiment was recorded.

    Reproducibility is not a property of a record, it is a property of the
    record plus the current machine. This says which of the two moved.
    """
    record = load(experiment_id, root)
    stored = record.provenance
    current_env = capture_environment()

    drift: dict[str, dict] = {}

    current_code = code_version()
    if current_code != stored.code_version:
        drift["code_version"] = {"recorded": stored.code_version,
                                 "current": current_code}

    current_revision = code_revision()
    if stored.code_revision and current_revision != stored.code_revision:
        drift["code_revision"] = {"recorded": stored.code_revision,
                                   "current": current_revision}

    if stored.feature_version != features_mod.FEATURE_VERSION:
        drift["feature_version"] = {"recorded": stored.feature_version,
                                 "current": features_mod.FEATURE_VERSION}
    if stored.feature_schema_hash and stored.feature_schema_hash != features_mod.schema_hash():
        drift["feature_schema_hash"] = {"recorded": stored.feature_schema_hash,
                                        "current": features_mod.schema_hash()}

    if stored.preprocessing_version != PREPROCESSING_VERSION:
        drift["preprocessing_version"] = {
            "recorded": stored.preprocessing_version,
            "current": PREPROCESSING_VERSION}
    recorded_mode = str(record.configuration.get("resample_mode") or "time")
    if stored.preprocessing_schema_hash and \
            stored.preprocessing_schema_hash != preprocessing_schema_hash(recorded_mode):
        drift["preprocessing_schema_hash"] = {
            "recorded": stored.preprocessing_schema_hash,
            "current": preprocessing_schema_hash(recorded_mode)}

    library_drift = {
        name: {"recorded": value, "current": current_env.get(name)}
        for name, value in (stored.environment or {}).items()
        if current_env.get(name) != value
    }
    if library_drift:
        drift["environment"] = library_drift

    return {
        "experiment_id": experiment_id,
        "reproducible": not drift,
        "drift": drift,
        "seed": stored.seed,
        "note": ("Environment matches; rerunning with this seed should "
                 "reproduce the result."
                 if not drift else
                 "Environment has changed since this run; results may differ."),
    }


def compare(experiment_ids: list[str], metric: str,
            root: Path | None = None) -> dict:
    """Line up one metric across several experiments.

    Comparability is checked rather than assumed: experiments recorded under
    different feature or preprocessing versions are flagged, because a metric
    computed from different inputs is not the same metric.
    """
    rows = []
    versions: set[tuple[int, int, str, str]] = set()

    for experiment_id in experiment_ids:
        try:
            record = load(experiment_id, root)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        versions.add((record.provenance.feature_version,
                      record.provenance.preprocessing_version,
                      record.provenance.feature_schema_hash or "",
                      record.provenance.preprocessing_schema_hash or ""))
        rows.append({
            "experiment_id": experiment_id,
            "kind": record.kind,
            "value": record.results.get(metric),
            "feature_version": record.provenance.feature_version,
            "runtime_seconds": record.runtime_seconds,
        })

    scored = [r for r in rows if isinstance(r["value"], (int, float))]
    scored.sort(key=lambda r: -r["value"])

    return {
        "metric": metric,
        "rows": rows,
        "best": scored[0] if scored else None,
        "comparable": len(versions) <= 1,
        "warning": (None if len(versions) <= 1 else
                    "Experiments span different feature or preprocessing "
                    "versions; the metric is not directly comparable."),
    }
