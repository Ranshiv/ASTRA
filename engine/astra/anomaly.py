"""Baseline anomaly detection (plan sections 13 and 29, phase 4).

Four independent detectors run over the same feature matrix:

  Isolation Forest   isolates points with short random-partition paths
  Local Outlier Factor   compares local density to a point's neighbours
  One-Class SVM      learns a boundary around the bulk of the data
  PCA reconstruction  flags points a low-rank model cannot rebuild

They are kept separate rather than blended into one model because they fail
differently: Isolation Forest is weak on local anomalies, LOF is weak in
high dimensions, One-Class SVM is sensitive to its kernel width, and PCA only
sees linear structure. Plan section 16 allocates 10% of the candidate score to
"model agreement", so the ensemble records how many detectors concur rather
than hiding that behind a single number.

Nothing here is supervised. Plan section 13's warning applies: a newer method
is not automatically better, and the comparison has to be quantitative.

Consensus is an average of per-detector RANKS, not an average of the raw
normalised scores. Measured on 301 real ZTF sequences (docs/LIMITATIONS.md,
Phase 8): an equal-weight mean of normalised scores scored WORSE (0.824) than
Isolation Forest alone (0.840), because One-Class SVM's weaker, noisier score
(0.646) dragged the mean down.
`_normalise` already min-max rescales every detector to [0, 1], so the
problem is not literal scale -- it is DISTRIBUTION SHAPE. A detector whose
raw scores plateau or skew (One-Class SVM's own kernel-saturation note below
describes exactly this: many points far from the training mass can share
close to the same decision-function value) still gets stretched across the
full [0, 1] range by min-max normalisation, which can hand disproportionately
high normalised scores to a lot of ordinary rows sitting on that plateau, not
just the genuine outliers. Averaging that skewed distribution in with the
other detectors' scores lets the skew leak into the consensus. Rank
aggregation is invariant to each detector's score distribution by
construction -- only the relative ORDER within a detector matters, so a
detector cannot distort the consensus merely by having an unusual shape,
however it happens to be normalised.
This is not a claim that rank-averaging beats a fixed, performance-weighted
average in general -- it is the safer default because it does not bake in
weights measured on one dataset with one injection scheme, which is exactly
what plan section 16 warns against treating as settled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .featurematrix import FeatureMatrix

# Expected share of anomalies. Rare astronomical events are rare; a high
# contamination value would flag ordinary stars and drown the interesting ones.
DEFAULT_CONTAMINATION = 0.05

DETECTOR_NAMES = ("isolation_forest", "lof", "one_class_svm", "pca_reconstruction")


def calibrate_scores(scores: np.ndarray, *, reference: np.ndarray | None = None,
                     method: str = "empirical_cdf") -> np.ndarray:
    """Map batch-relative scores to an empirical probability scale.

    Min-max scores answer “how high in this batch?” but are not comparable
    across runs.  The empirical CDF is monotone, robust to detector plateaus,
    and makes ``0.99`` mean the value lies in the most extreme one percent of
    the supplied reference population.  A separate ``reference`` population
    can be used for deployment; absent one, the current batch is used and the
    caller should report that calibration is batch-relative.
    """
    if method != "empirical_cdf":
        raise ValueError("unknown calibration method; expected empirical_cdf")
    values = np.asarray(scores, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    baseline = np.asarray(reference if reference is not None else values[finite], dtype=float)
    baseline = np.sort(baseline[np.isfinite(baseline)])
    if baseline.size == 0:
        return result
    # side="right" gives tied values the upper endpoint of their probability
    # mass, which is conservative for anomaly review.
    ranks = np.searchsorted(baseline, values[finite], side="right")
    result[finite] = ranks / baseline.size
    return np.clip(result, 0.0, 1.0)


def calibration_report(scores: np.ndarray, reference: np.ndarray | None = None) -> dict:
    """Summarise calibration provenance without hiding the reference choice."""
    values = np.asarray(scores, dtype=float)
    finite = values[np.isfinite(values)]
    baseline = np.asarray(reference, dtype=float) if reference is not None else finite
    baseline = baseline[np.isfinite(baseline)]
    return {
        "method": "empirical_cdf",
        "reference_rows": int(len(baseline)),
        "scored_rows": int(len(finite)),
        "reference_external": reference is not None,
        "minimum": float(np.min(baseline)) if len(baseline) else None,
        "maximum": float(np.max(baseline)) if len(baseline) else None,
    }


@dataclass
class DetectorScores:
    """Per-detector scores, normalised so 1.0 is most anomalous."""

    name: str
    scores: np.ndarray
    flagged: np.ndarray  # boolean, detector's own outlier decision

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "flagged": int(np.count_nonzero(self.flagged)),
            "score_mean": float(np.mean(self.scores)) if self.scores.size else 0.0,
            "score_max": float(np.max(self.scores)) if self.scores.size else 0.0,
        }


@dataclass
class EnsembleResult:
    """Combined output over the usable rows of a feature matrix."""

    identities: list[dict]
    detectors: dict[str, DetectorScores] = field(default_factory=dict)
    consensus: np.ndarray = field(default_factory=lambda: np.empty(0))
    agreement: np.ndarray = field(default_factory=lambda: np.empty(0))
    contamination: float = DEFAULT_CONTAMINATION
    skipped_rows: int = 0

    def ranked(self, top: int = 50, reference: np.ndarray | None = None) -> list[dict]:
        """Highest-scoring candidates first, with the evidence attached.

        ``reference`` is an optional external population (e.g. a persisted,
        cross-run calibration reference) to calibrate ``consensus_calibrated``
        against; absent one, calibration falls back to this batch, matching
        ``calibrate_scores``'s own default.
        """
        if self.consensus.size == 0:
            return []

        calibrated = calibrate_scores(self.consensus, reference=reference)
        order = np.argsort(-self.consensus)[:top]
        results = []
        for rank, index in enumerate(order, start=1):
            entry = {
                "rank": rank,
                "consensus_score": float(self.consensus[index]),
                "consensus_calibrated": float(calibrated[index]),
                "model_agreement": int(self.agreement[index]),
                **self.identities[index],
            }
            for name, detector in self.detectors.items():
                entry[f"score_{name}"] = float(detector.scores[index])
            results.append(entry)
        return results

    def to_dict(self, reference: np.ndarray | None = None) -> dict:
        return {
            "rows_scored": len(self.identities),
            "rows_skipped": self.skipped_rows,
            "contamination": self.contamination,
            "detectors": [d.to_dict() for d in self.detectors.values()],
            "calibration": calibration_report(self.consensus, reference=reference),
        }


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Map raw detector output onto 0..1, where 1 is most anomalous."""
    if scores.size == 0:
        return scores
    low, high = float(np.min(scores)), float(np.max(scores))
    if high <= low:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def _rank_consensus(stacked: np.ndarray) -> np.ndarray:
    """Average per-detector RANKS rather than the raw normalised scores.

    `stacked` is (n_detectors, n_rows), each row already 0..1 from
    `_normalise`. That min-max rescaling already equalises scale, so
    averaging the raw values in still lets one detector's score
    DISTRIBUTION (a plateau or skew, not a scale difference) leak into the
    consensus -- see the module docstring for the measured effect. Ranking
    each detector's scores first removes the dependence on distribution
    shape entirely: a detector whose values plateau or skew still produces a
    full, evenly-spread rank from 1 to n, so it can no longer hand a
    disproportionate share of high scores to ordinary rows just because its
    own scores happened to cluster there.

    `scipy.stats.rankdata(method="average")` rather than a plain double
    argsort: detector scores can tie exactly (LOF and PCA reconstruction
    error both do on small or degenerate batches), and a plain argsort-based
    rank would assign those ties arbitrary distinct positions instead of the
    shared average rank.
    """
    from scipy.stats import rankdata

    n_rows = stacked.shape[1]
    if n_rows == 0:
        return np.empty(0)

    ranked = np.vstack([rankdata(row, method="average") / n_rows
                        for row in stacked])
    return np.mean(ranked, axis=0)


def prepare(matrix: FeatureMatrix) -> tuple[np.ndarray, list[dict], int]:
    """Select usable rows and standardise the features.

    Detectors are distance-based, and the raw features span wildly different
    scales — `n_points` is in the hundreds while `stetson_k` is around one —
    so without standardisation the largest-magnitude column would dominate
    every distance computation.
    """
    mask = matrix.finite_mask()
    usable = matrix.values[mask]
    identities = [identity for identity, keep in zip(matrix.identities, mask) if keep]
    skipped = int(np.count_nonzero(~mask))

    if usable.shape[0] == 0:
        return usable, identities, skipped

    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(usable), identities, skipped


def run_isolation_forest(x: np.ndarray, contamination: float,
                         seed: int = 42) -> DetectorScores:
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(contamination=contamination, random_state=seed,
                            n_estimators=200)
    flagged = model.fit_predict(x) == -1
    # score_samples is higher for inliers, so it is negated before normalising.
    return DetectorScores("isolation_forest",
                          _normalise(-model.score_samples(x)), flagged)


def run_lof(x: np.ndarray, contamination: float) -> DetectorScores:
    from sklearn.neighbors import LocalOutlierFactor

    neighbours = max(2, min(20, x.shape[0] - 1))
    model = LocalOutlierFactor(n_neighbors=neighbours, contamination=contamination)
    flagged = model.fit_predict(x) == -1
    return DetectorScores("lof",
                          _normalise(-model.negative_outlier_factor_), flagged)


def median_heuristic_gamma(x: np.ndarray, sample_cap: int = 500,
                           seed: int = 42) -> float:
    """Kernel width from the median pairwise distance.

    sklearn's `gamma="scale"` is tuned for drawing a classification boundary,
    and on standardised high-dimensional data it produces a kernel narrow
    enough to saturate: every point far from the training mass returns a
    kernel value of zero against all support vectors, so the decision function
    is *constant* out there. The detector can then still flag an outlier but
    cannot rank one against another, which is exactly what a candidate
    ranking needs.

    Setting gamma to 1 / (2 * median_distance^2) keeps the kernel responsive
    at the distances the data actually spans. Distances are computed on a
    subsample because the pairwise matrix is quadratic.
    """
    rows = x.shape[0]
    if rows < 2:
        return 1.0 / max(x.shape[1], 1)

    if rows > sample_cap:
        rng = np.random.default_rng(seed)
        sample = x[rng.choice(rows, size=sample_cap, replace=False)]
    else:
        sample = x

    differences = sample[:, None, :] - sample[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=-1))
    upper = distances[np.triu_indices_from(distances, k=1)]

    median_distance = float(np.median(upper)) if upper.size else 0.0
    if median_distance <= 0:
        return 1.0 / max(x.shape[1], 1)
    return 1.0 / (2.0 * median_distance ** 2)


def run_one_class_svm(x: np.ndarray, contamination: float) -> DetectorScores:
    from sklearn.svm import OneClassSVM

    model = OneClassSVM(nu=contamination, kernel="rbf",
                        gamma=median_heuristic_gamma(x))
    flagged = model.fit_predict(x) == -1
    return DetectorScores("one_class_svm",
                          _normalise(-model.decision_function(x)), flagged)


def run_pca_reconstruction(x: np.ndarray, contamination: float,
                           variance_kept: float = 0.95) -> DetectorScores:
    """Flag rows a low-rank model cannot rebuild.

    Objects that behave like the bulk population are reconstructed well by
    the leading components; a large residual means the object does not lie on
    the manifold the population occupies.
    """
    from sklearn.decomposition import PCA

    components = max(1, min(x.shape[1], x.shape[0] - 1))
    model = PCA(n_components=components)
    reconstructed = model.inverse_transform(model.fit_transform(x))
    error = np.sum((x - reconstructed) ** 2, axis=1)

    # Rank truncated to the components explaining `variance_kept`, so the
    # residual is meaningful rather than numerically zero.
    cumulative = np.cumsum(model.explained_variance_ratio_)
    keep = int(np.searchsorted(cumulative, variance_kept) + 1)
    keep = max(1, min(keep, components))

    truncated = PCA(n_components=keep)
    reconstructed = truncated.inverse_transform(truncated.fit_transform(x))
    error = np.sum((x - reconstructed) ** 2, axis=1)

    threshold = float(np.quantile(error, 1.0 - contamination))
    return DetectorScores("pca_reconstruction", _normalise(error),
                          error >= threshold)


def detect(matrix: FeatureMatrix,
           contamination: float = DEFAULT_CONTAMINATION,
           seed: int = 42) -> EnsembleResult:
    """Run every detector and combine them into a consensus ranking."""
    x, identities, skipped = prepare(matrix)

    result = EnsembleResult(identities=identities, contamination=contamination,
                            skipped_rows=skipped)
    # Every detector needs enough neighbours to say anything meaningful.
    if x.shape[0] < 10:
        return result

    runners = {
        "isolation_forest": lambda: run_isolation_forest(x, contamination, seed),
        "lof": lambda: run_lof(x, contamination),
        "one_class_svm": lambda: run_one_class_svm(x, contamination),
        "pca_reconstruction": lambda: run_pca_reconstruction(x, contamination),
    }

    for name, runner in runners.items():
        try:
            result.detectors[name] = runner()
        except Exception:  # noqa: BLE001 - one detector failing must not lose the rest
            continue

    if not result.detectors:
        return result

    stacked = np.vstack([d.scores for d in result.detectors.values()])
    result.consensus = _rank_consensus(stacked)
    result.agreement = np.sum(
        np.vstack([d.flagged for d in result.detectors.values()]), axis=0
    ).astype(int)
    return result


def save_ranking(result: EnsembleResult, name: str, top: int = 200,
                 root: Path | None = None) -> Path:
    """Write the ranked candidates as JSON for the interface and for review."""
    root = root or config.PATHS.projects
    path = root / "candidates" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"summary": result.to_dict(), "candidates": result.ranked(top)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


DEFAULT_CALIBRATION_CAP = 20_000


def _calibration_reference_path(name: str, root: Path | None) -> Path:
    root = root or config.PATHS.projects
    return root / "calibration" / f"{name}.json"


def load_calibration_reference(name: str, root: Path | None = None) -> np.ndarray:
    """Load the persisted, cross-run consensus-score reference for ``name``.

    Absent a prior run (or a corrupt/missing file), returns an empty array so
    callers naturally fall back to batch-relative calibration via
    ``calibrate_scores``'s own default.
    """
    path = _calibration_reference_path(name, root)
    if not path.exists():
        return np.empty(0)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = np.asarray(payload.get("scores", []), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        return np.empty(0)
    return scores[np.isfinite(scores)]


def update_calibration_reference(name: str, root: Path | None, scores: np.ndarray,
                                 cap: int = DEFAULT_CALIBRATION_CAP) -> Path:
    """Fold this run's consensus scores into the persisted reference for ``name``.

    Bounded by ``cap`` via FIFO eviction (oldest scores drop first), so the
    reference stays a recent, representative sample instead of growing
    unbounded or going stale by never evicting.
    """
    path = _calibration_reference_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_calibration_reference(name, root)
    new_scores = np.asarray(scores, dtype=float)
    new_scores = new_scores[np.isfinite(new_scores)]
    combined = np.concatenate([existing, new_scores])
    if combined.size > cap:
        combined = combined[-cap:]

    path.write_text(json.dumps({"scores": combined.tolist()}), encoding="utf-8")
    return path
