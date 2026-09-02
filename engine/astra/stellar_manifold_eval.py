"""Evaluation harness for `stellar_manifold.py` (backlog item 12's metric:
"improvement in anomaly precision at fixed recall").

Two things this codebase did not already have, confirmed while planning:

1. Injecting a TRUE-BY-CONSTRUCTION anomaly directly into engineered feature
   columns. `evaluate.build_injected()` only ever injects into light-curve
   SEQUENCES and recomputes features afterward -- there was no precedent for
   perturbing a feature column directly. `inject_cmd_outliers` does exactly
   that, in the same spirit: a known, deliberate CMD-position offset with a
   true label, not an assumption about what "anomalous" means.
2. "Precision at a FIXED RECALL level." This codebase already has precision
   at a fixed top-k COUNT (`evaluate.score_method`) and precision at a fixed
   0.5 THRESHOLD (`ranker.metrics`) -- neither is the PR-curve-threshold-sweep
   convention this backlog item asks for. `precision_at_recall` is new.

`evaluate_manifold_contribution` reuses the real, unmodified `anomaly.detect()`
ensemble -- this is not a new detector, only new evidence and a new way of
scoring it. It NEVER asserts which arm wins: it reports the "with" and
"without" arms' precision at each requested recall level, multi-seed with a
mean/CI95 summary, the same restraint `sweep.SweepResult.best()` already
applies elsewhere in this codebase.

Both arms are built from the exact SAME row population (fully Gaia- and
manifold-matched, fully finite) so a reported difference cannot be an
artefact of the two arms silently comparing different objects -- the same
"different populations" trap `featurematrix.join_gaia_columns`'s own
docstring already warns about (citing the ztf_gaia entry in
docs/LIMITATIONS.md).
"""

from __future__ import annotations

import numpy as np

from .research import stats as research_stats

from .featurematrix import FeatureMatrix, STELLAR_MANIFOLD_COLUMNS

DEFAULT_SEEDS: tuple[int, ...] = (17, 29, 43)
_MANIFOLD_DERIVED_COLUMNS = (
    "manifold_residual_mag", "manifold_arc_length", "manifold_teff_k",
)


def precision_at_recall(labels, scores, target_recall: float = 0.5) -> float | None:
    """Best precision among operating points achieving >= target_recall.

    None (never 0.0) when that recall level is unreachable -- e.g. a
    degenerate label set with no positives at all, mirroring
    `evaluate.score_method`'s own "degenerate label set" handling.
    """
    from sklearn.metrics import precision_recall_curve

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels) or len(labels) == 0:
        return None

    finite = np.isfinite(scores)
    if not finite.all():
        scores = np.where(finite, scores,
                          float(np.nanmin(scores[finite])) if finite.any() else 0.0)

    precision, recall, _ = precision_recall_curve(labels, scores)
    achievable = recall >= target_recall
    if not achievable.any():
        return None
    return float(np.max(precision[achievable]))


def inject_cmd_outliers(matrix: FeatureMatrix, fraction: float = 0.1,
                        offset_mag: float = 3.0,
                        seed: int = 42) -> tuple[FeatureMatrix, np.ndarray]:
    """Shift a fraction of rows' Gaia absolute magnitude by a known offset.

    Simulates an over-/under-luminous CMD outlier (a blend, a bad distance,
    a genuinely unusual object). `manifold_residual_mag`/`arc_length`/`teff_k`
    are RECOMPUTED for injected rows from the perturbed position -- leaving
    them stale would make any measured improvement an artefact of the join
    simply not reflecting the injected anomaly, the same "features are
    recomputed after injection" discipline `ablation.py` already documents
    for sequence-space injection.
    """
    from . import stellar_manifold

    required = ("gaia_bp_rp", "gaia_abs_g_mag", *_MANIFOLD_DERIVED_COLUMNS)
    missing = [name for name in required if name not in matrix.feature_names]
    if missing:
        raise ValueError(
            "inject_cmd_outliers requires a Gaia- and manifold-joined "
            f"matrix; missing columns: {missing}"
        )

    rng = np.random.default_rng(seed)
    n = len(matrix)
    values = matrix.values.copy()
    labels = np.zeros(n, dtype=int)
    if n == 0:
        return FeatureMatrix(values=values, identities=matrix.identities,
                             feature_names=matrix.feature_names,
                             feature_version=matrix.feature_version), labels

    count = max(1, int(round(n * fraction)))
    chosen = rng.choice(n, size=min(count, n), replace=False)

    bp_rp_col = matrix.feature_names.index("gaia_bp_rp")
    abs_g_col = matrix.feature_names.index("gaia_abs_g_mag")
    residual_col = matrix.feature_names.index("manifold_residual_mag")
    arc_col = matrix.feature_names.index("manifold_arc_length")
    teff_col = matrix.feature_names.index("manifold_teff_k")

    for index in chosen:
        direction = float(rng.choice([-1.0, 1.0]))
        values[index, abs_g_col] += direction * offset_mag

        # Extinction lives in the identity dict (see featurematrix.
        # GAIA_EXTINCTION_IDENTITY_KEYS), not a value column.
        identity = matrix.identities[index]
        a_g = identity.get("gaia_a_g")
        ebpminrp = identity.get("gaia_ebpminrp")
        result = stellar_manifold.isochrone_residual(
            values[index, bp_rp_col], values[index, abs_g_col], a_g, ebpminrp)
        values[index, residual_col] = result["residual_mag"]
        values[index, arc_col] = result["arc_length_fraction"]
        values[index, teff_col] = result["teff_k"]
        labels[index] = 1

    injected = FeatureMatrix(values=values, identities=matrix.identities,
                             feature_names=matrix.feature_names,
                             feature_version=matrix.feature_version)
    return injected, labels


def _summary(values: list[float | None]) -> dict | None:
    """Delegates to `research.stats.summary` -- see that module's docstring
    for why this shape (mean/std/ci95 over repeated seeds, not object-group
    bootstrap) is the right one here. Filters `None` first: unlike the
    other eval modules in this family, this one's callers can supply
    unresolved (missing-Gaia-column) rows as `None` rather than NaN, which
    `research.stats.summary` does not itself understand. Was this module's
    own local reimplementation; migrated per docs/LIMITATIONS.md's tracked
    debt."""
    return research_stats.summary([v for v in values if v is not None])


def evaluate_manifold_contribution(
        matrix_with_gaia: FeatureMatrix,
        fractions: tuple[float, ...] = (0.1,),
        target_recalls: tuple[float, ...] = (0.5, 0.8),
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        offset_mag: float = 3.0) -> dict:
    """Does the isochrone-residual feature improve precision at fixed recall?

    `matrix_with_gaia` must already carry `featurematrix.GAIA_JOIN_COLUMNS`
    (i.e. `join_gaia_columns` has run). This function joins the manifold
    columns itself, then restricts BOTH arms to the same fully-matched,
    fully-finite row population before injecting and scoring, so a reported
    difference is attributable to the feature, not to the two arms silently
    scoring different objects.

    Raises rather than silently degrading when fewer than 20 rows survive
    that restriction, or when `anomaly.detect()` has to drop a row it was
    promised was already finite -- both are "the input is not what this
    function requires," not a recoverable edge case worth guessing through.
    """
    from . import anomaly
    from .featurematrix import join_stellar_manifold_columns

    if len(seeds) < 2:
        raise ValueError("evaluate_manifold_contribution needs at least two seeds")

    joined, _ = join_stellar_manifold_columns(matrix_with_gaia)
    gaia_matched_col = joined.feature_names.index("gaia_matched")
    manifold_matched_col = joined.feature_names.index("manifold_matched")

    keep = (joined.finite_mask()
           & (joined.values[:, gaia_matched_col] == 1.0)
           & (joined.values[:, manifold_matched_col] == 1.0))
    if int(keep.sum()) < 20:
        raise ValueError(
            f"only {int(keep.sum())} fully Gaia- and manifold-matched, "
            "finite rows; need at least 20 to evaluate"
        )

    restricted = joined.subset(list(np.where(keep)[0]))
    without_names = tuple(name for name in restricted.feature_names
                          if name not in STELLAR_MANIFOLD_COLUMNS)

    results: dict[float, dict] = {}
    for fraction in fractions:
        per_recall: dict[float, dict[str, list[float | None]]] = {
            target: {"with_manifold": [], "without_manifold": []}
            for target in target_recalls
        }

        for seed in seeds:
            injected_with, labels = inject_cmd_outliers(
                restricted, fraction=fraction, offset_mag=offset_mag, seed=seed)
            injected_without = injected_with.subset(
                list(range(len(injected_with))), feature_names=without_names)

            ensemble_with = anomaly.detect(injected_with, seed=seed)
            ensemble_without = anomaly.detect(injected_without, seed=seed)
            if ensemble_with.skipped_rows or ensemble_without.skipped_rows:
                raise ValueError(
                    "anomaly.detect() dropped a row from an input this "
                    "function already restricted to be fully finite"
                )

            for target in target_recalls:
                per_recall[target]["with_manifold"].append(
                    precision_at_recall(labels, ensemble_with.consensus, target))
                per_recall[target]["without_manifold"].append(
                    precision_at_recall(labels, ensemble_without.consensus, target))

        results[fraction] = {
            str(target): {arm: _summary(values) for arm, values in arms.items()}
            for target, arms in per_recall.items()
        }

    return results
