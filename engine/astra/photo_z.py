"""Empirical (training-set) photometric redshift from multi-band photometry.

Shaped like the rest of this module family: opt-in, cites its method,
states its scope limits explicitly, never wired into `evidence.WEIGHTS`/
`scoring.combine()`/`rpc.py`.

Photo-z estimation has two standard families in the literature: SED
template fitting (needs a real galaxy/QSO spectral template library this
codebase does not have -- explicitly out of scope) and empirical/
training-set regression (needs a real sample with known spectroscopic
redshifts to calibrate against -- what this module does). `fit_photo_z_knn`
is k-nearest-neighbour regression in photometry space (e.g. Csabai et al.
2003), via `sklearn.neighbors.KNeighborsRegressor` (`scikit-learn` already
a core dependency, used unchanged) -- a real, bounded, standard technique,
not a novel method.

`build_calibration_sample` gets its real spectroscopic-redshift training
labels from `surveys/sdss.py`'s new `query_spectroscopic_redshifts`
(SDSS `SpecObjAll.z`/`zErr`), cross-matched against real DES or Pan-STARRS
multi-band photometry via `crossmatch.match_catalogs` (reused unchanged,
crossmatch.py:211) -- a real calibration sample, not synthetic. TNS
(credential-gated) and the ZTF alert stream are not photometric-redshift
sources and are not used here.

`photo_z_nmad` is the standard photo-z normalised-median-absolute-deviation
metric (`1.4826 * median(|z_pred - z_true| / (1 + z_true))`), new to this
codebase (confirmed no NMAD helper existed; `alerce.py`'s own `1.4826` MAD
scaling constant is precedent for the scaling factor, not this formula).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .research import stats as research_stats

from .crossmatch import match_catalogs
from .surveys.base import SourceRef

MIN_CALIBRATION_ROWS = 10


class PhotoZError(ValueError):
    """A photo-z calibration, fit, or evaluation could not be completed."""


def build_calibration_sample(photometry_sources: list[SourceRef], redshift_sources: list[SourceRef],
                             band_keys: tuple[str, ...], *, radius_arcsec: float = 2.0) -> list[dict]:
    """Cross-matches real multi-band photometry against real spectroscopic
    redshifts, via `crossmatch.match_catalogs` unchanged.

    `band_keys` names the `SourceRef.extra` fields to pull magnitudes from
    (e.g. DES's `("g_mean", "r_mean", "i_mean", "z_mean")` or Pan-STARRS's
    `("gMeanPSFMag", ..., "yMeanPSFMag")`) -- this function is band-set
    agnostic, per each connector's own real column names.
    """
    if not band_keys:
        raise PhotoZError("band_keys must be non-empty")
    matches = match_catalogs(photometry_sources, redshift_sources, radius_arcsec=radius_arcsec)
    rows: list[dict] = []
    for match in matches:
        z = match.counterpart.extra.get("z")
        if z is None:
            continue
        try:
            z = float(z)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(z) or z <= 0:
            continue
        magnitudes = []
        valid = True
        for key in band_keys:
            value = match.source.extra.get(key)
            try:
                magnitudes.append(float(value))
            except (TypeError, ValueError):
                valid = False
                break
        if not valid or not all(np.isfinite(m) for m in magnitudes):
            continue
        rows.append({"magnitudes": magnitudes, "z_true": z,
                     "separation_arcsec": match.separation_arcsec})
    return rows


@dataclass(frozen=True)
class PhotoZModel:
    band_keys: tuple[str, ...]
    k: int
    _regressor: object
    _scaler: object

    def predict(self, magnitudes: list[list[float]]) -> np.ndarray:
        scaled = self._scaler.transform(np.asarray(magnitudes, dtype=np.float64))
        return np.asarray(self._regressor.predict(scaled), dtype=np.float64)


def fit_photo_z_knn(magnitudes: np.ndarray, redshifts: np.ndarray, *,
                    band_keys: tuple[str, ...] = (), k: int = 10) -> PhotoZModel:
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    redshifts = np.asarray(redshifts, dtype=np.float64)
    if len(magnitudes) < MIN_CALIBRATION_ROWS:
        raise PhotoZError(f"need at least {MIN_CALIBRATION_ROWS} calibration rows, got {len(magnitudes)}")
    if len(magnitudes) != len(redshifts):
        raise PhotoZError("magnitudes and redshifts must be the same length")
    if k < 1:
        raise PhotoZError("k must be at least 1")
    k = min(k, len(magnitudes))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(magnitudes)
    regressor = KNeighborsRegressor(n_neighbors=k)
    regressor.fit(scaled, redshifts)
    return PhotoZModel(band_keys=band_keys, k=k, _regressor=regressor, _scaler=scaler)


def photo_z_nmad(z_true, z_pred) -> float:
    """`1.4826 * median(|z_pred - z_true| / (1 + z_true))` -- the standard
    photo-z NMAD definition."""
    z_true = np.asarray(z_true, dtype=np.float64)
    z_pred = np.asarray(z_pred, dtype=np.float64)
    if len(z_true) == 0:
        raise PhotoZError("need at least one point to compute NMAD")
    if np.any(z_true <= -1.0):
        raise PhotoZError("z_true must be > -1 (1 + z_true must be positive)")
    delta_z = (z_pred - z_true) / (1.0 + z_true)
    return float(1.4826 * np.median(np.abs(delta_z)))


def _summary(values: list[float]) -> dict | None:
    """Delegates to `research.stats.summary` -- see that module's docstring
    for why this shape (mean/std/ci95 over repeated seeds, not object-group
    bootstrap) is the right one here. Was this module's own local
    reimplementation; migrated per docs/LIMITATIONS.md's tracked debt."""
    return research_stats.summary(values)


def evaluate_photo_z(calibration_sample: list[dict], *, k: int = 10,
                     test_fraction: float = 0.3, n_seeds: int = 5, seed: int = 42) -> dict:
    """Held-out NMAD on a real (or synthetic, for mechanism validation)
    calibration sample, multi-seed mean/std/ci95 summary
    (`sweep.TrialResult`-style). Reports only; never corrects a score."""
    if len(calibration_sample) < MIN_CALIBRATION_ROWS:
        raise PhotoZError(f"need at least {MIN_CALIBRATION_ROWS} calibration rows, "
                          f"got {len(calibration_sample)}")
    magnitudes = np.array([row["magnitudes"] for row in calibration_sample], dtype=np.float64)
    redshifts = np.array([row["z_true"] for row in calibration_sample], dtype=np.float64)
    n = len(calibration_sample)

    nmad_values: list[float] = []
    for trial in range(n_seeds):
        rng = np.random.default_rng(seed + trial)
        order = rng.permutation(n)
        cut = max(MIN_CALIBRATION_ROWS - 1, int(round(n * (1.0 - test_fraction))))
        cut = min(cut, n - 1) if n > 1 else 0
        train_idx, test_idx = order[:cut], order[cut:]
        if len(train_idx) < MIN_CALIBRATION_ROWS or len(test_idx) == 0:
            continue
        model = fit_photo_z_knn(magnitudes[train_idx], redshifts[train_idx], k=k)
        predicted = model.predict(magnitudes[test_idx])
        nmad_values.append(photo_z_nmad(redshifts[test_idx], predicted))

    return {"nmad": _summary(nmad_values), "n_calibration_rows": n, "k": k,
           "n_seeds_used": len(nmad_values)}


__all__ = [
    "PhotoZError", "MIN_CALIBRATION_ROWS", "build_calibration_sample",
    "PhotoZModel", "fit_photo_z_knn", "photo_z_nmad", "evaluate_photo_z",
]
