"""Turning irregular light curves into fixed-length tensors (plan phase 5).

Three problems have to be solved before a neural network can see this data:

1. Lengths vary enormously — 234 points for a ZTF curve against 19,499 for a
   TESS sector — but a convolutional encoder needs a fixed input size.
2. Units differ. ZTF reports magnitudes (smaller is brighter), TESS reports
   electron flux. Feeding both raw would teach the model to separate surveys
   rather than behaviours, which is the survey bias plan section 36 warns
   about and which was already measured in the Phase 4 baselines.
3. Sampling is irregular and gappy. Ground-based surveys have seasonal gaps of
   200 days; interpolating straight across one invents data that was never
   observed.

The representation here is a 2-channel sequence: channel 0 is the resampled,
per-curve normalised brightness; channel 1 is a validity mask that is 0
wherever the value had to be interpolated across a real gap. The mask means
the model can learn to ignore invented points instead of treating them as
observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import store
from .surveys.base import LightCurve

DEFAULT_LENGTH = 256

# A grid point is "valid" only if a real observation lies within this many grid
# spacings of it. Beyond that the value is an interpolation across a gap.
VALIDITY_RADIUS = 1.0

# Below this a curve cannot be resampled meaningfully.
MIN_POINTS = 16

# A gap this many times the median spacing starts a new observing season.
# Ground-based surveys stop observing a field for months when it sets, and a
# uniform grid spends most of its points inside those gaps: measured mean
# validity across resampled ZTF curves is about 0.40, so roughly 60% of every
# sequence is masked-out interpolation carrying no information.
SEASON_GAP_FACTOR = 30.0

# A season needs enough points of its own to be worth its share of the grid.
MIN_SEASON_POINTS = 8

# Phase resampling is only meaningful when the period is real.
MIN_PHASE_PERIOD_SNR = 5.0

# How a curve is laid onto the fixed grid.
#   time   uniform over the full baseline (the original, and still the default)
#   season segmented on observing gaps, each season given its share of the grid
#   phase  folded on a credible period, resampled in phase rather than time
RESAMPLE_MODES = ("time", "season", "phase")


@dataclass
class SequenceBatch:
    """Fixed-length sequences plus the identity of each row."""

    values: np.ndarray        # (n, 2, length) float32
    identities: list[dict]
    length: int = DEFAULT_LENGTH
    mode: str = "time"

    def __len__(self) -> int:
        return len(self.identities)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def to_dict(self) -> dict:
        coverage = (float(np.mean(self.values[:, 1, :])) if len(self) else 0.0)
        return {
            "rows": len(self),
            "length": self.length,
            "channels": 2,
            "mode": self.mode,
            # The fraction of grid points backed by a real observation. This is
            # the number the season and phase modes exist to move: the uniform
            # time grid measured about 0.40 on real ZTF data.
            "mean_coverage": round(coverage, 4),
        }


def normalise(value: np.ndarray) -> np.ndarray:
    """Centre on the median and scale by the MAD.

    Median and MAD rather than mean and standard deviation because a single
    bad epoch would otherwise set the scale for the whole curve. This discards
    absolute brightness deliberately — that information already lives in the
    feature vector, and keeping it here would let the model separate bright
    surveys from faint ones instead of learning shape.
    """
    median = float(np.median(value))
    mad = float(np.median(np.abs(value - median))) * 1.4826
    if mad <= 0:
        spread = float(np.std(value))
        if spread <= 0:
            return np.zeros_like(value, dtype=np.float32)
        return ((value - median) / spread).astype(np.float32)
    return ((value - median) / mad).astype(np.float32)


def resample(curve: LightCurve, length: int = DEFAULT_LENGTH) -> np.ndarray | None:
    """Resample onto a uniform time grid, returning a (2, length) array.

    Returns None when the curve is too short or has no time span, so callers
    can skip it rather than train on a degenerate row.
    """
    tidy = curve.dropna().sorted_by_time()
    if len(tidy) < MIN_POINTS:
        return None

    start = float(tidy.time[0])
    end = float(tidy.time[-1])
    if end <= start:
        return None

    grid = np.linspace(start, end, length)
    spacing = (end - start) / (length - 1)

    normalised = normalise(tidy.value)
    resampled = np.interp(grid, tidy.time, normalised).astype(np.float32)

    # Distance from each grid point to the nearest real observation, in units
    # of grid spacing. This is what distinguishes an interpolated point inside
    # a well-sampled stretch from one invented across a seasonal gap.
    insert = np.searchsorted(tidy.time, grid)
    left = np.clip(insert - 1, 0, len(tidy) - 1)
    right = np.clip(insert, 0, len(tidy) - 1)
    distance = np.minimum(np.abs(grid - tidy.time[left]),
                          np.abs(grid - tidy.time[right]))
    mask = (distance <= VALIDITY_RADIUS * spacing).astype(np.float32)

    # Zero out invented values so the model never sees them as signal; the
    # mask channel tells it those positions carry no information.
    resampled = resampled * mask

    return np.stack([resampled, mask], axis=0)


def season_bounds(time: np.ndarray,
                  gap_factor: float = SEASON_GAP_FACTOR) -> list[tuple[int, int]]:
    """Split an observation time array on its real observing gaps.

    The threshold is derived from the curve's own median spacing rather than
    fixed in days, because a 2-minute-cadence TESS sector and a nightly ZTF
    field have nothing in common except that a gap is large relative to how
    often that instrument normally looks.
    """
    if len(time) < 2:
        return [(0, len(time))]

    spacing = np.diff(time)
    median = float(np.median(spacing))
    if not np.isfinite(median) or median <= 0:
        return [(0, len(time))]

    breaks = np.flatnonzero(spacing > gap_factor * median) + 1
    edges = [0, *breaks.tolist(), len(time)]
    return [(start, end) for start, end in zip(edges, edges[1:]) if end > start]


def resample_by_season(curve: LightCurve, length: int = DEFAULT_LENGTH,
                       gap_factor: float = SEASON_GAP_FACTOR
                       ) -> np.ndarray | None:
    """Resample each observing season onto its own share of the grid.

    Grid points are allotted in proportion to how many observations a season
    actually contains, so a well-sampled season is not compressed to make room
    for a sparse one. Nothing is interpolated across a gap, which is the whole
    point: those positions previously consumed most of the sequence while
    carrying no information.

    Seasons too short to resample are dropped rather than padded. A season with
    three points cannot support a shape, and inventing one would put fabricated
    structure into the training data.
    """
    tidy = curve.dropna().sorted_by_time()
    if len(tidy) < MIN_POINTS:
        return None

    seasons = [(start, end) for start, end in season_bounds(tidy.time, gap_factor)
               if end - start >= MIN_SEASON_POINTS
               and tidy.time[end - 1] > tidy.time[start]]
    if not seasons:
        return None
    if len(seasons) == 1:
        return resample(curve, length)

    # Normalise once, over the whole curve, so seasons stay on a common scale.
    # Normalising each season separately would erase real season-to-season
    # brightness changes, which for a long-period variable is the signal.
    normalised = normalise(tidy.value)

    counts = np.array([end - start for start, end in seasons], dtype=float)
    shares = np.maximum(2, np.floor(counts / counts.sum() * length).astype(int))
    # Give any remainder to the best-sampled season.
    while shares.sum() > length:
        shares[int(np.argmax(shares))] -= 1
    shares[int(np.argmax(counts))] += length - int(shares.sum())

    values, masks = [], []
    for (start, end), share in zip(seasons, shares):
        if share <= 0:
            continue
        times = tidy.time[start:end]
        grid = np.linspace(float(times[0]), float(times[-1]), share)
        spacing = ((float(times[-1]) - float(times[0])) / (share - 1)
                   if share > 1 else float("inf"))
        block = np.interp(grid, times, normalised[start:end])

        insert = np.searchsorted(times, grid)
        left = np.clip(insert - 1, 0, len(times) - 1)
        right = np.clip(insert, 0, len(times) - 1)
        distance = np.minimum(np.abs(grid - times[left]), np.abs(grid - times[right]))
        mask = (distance <= VALIDITY_RADIUS * spacing).astype(np.float32)

        values.append(block.astype(np.float32) * mask)
        masks.append(mask)

    resampled = np.concatenate(values)
    mask = np.concatenate(masks)
    if len(resampled) != length:
        return None
    return np.stack([resampled, mask], axis=0)


def resample_by_phase(curve: LightCurve, period_days: float,
                      length: int = DEFAULT_LENGTH) -> np.ndarray | None:
    """Fold on a period and resample in phase rather than in time.

    For a periodic source this is strictly more informative than a time grid:
    every observation lands somewhere in one cycle, so a curve with 60% of its
    time axis inside seasonal gaps can still cover phase densely.

    It is also destructive, and deliberately not the default. Folding discards
    everything aperiodic — a flare, a fade, a one-off transient — so it is only
    appropriate for a curve whose period is already established.
    """
    if not np.isfinite(period_days) or period_days <= 0:
        return None

    tidy = curve.dropna().sorted_by_time()
    if len(tidy) < MIN_POINTS:
        return None

    phase = np.mod(tidy.time - float(tidy.time[0]), period_days) / period_days
    order = np.argsort(phase)
    phase, values = phase[order], normalise(tidy.value)[order]

    grid = np.linspace(0.0, 1.0, length, endpoint=False)
    spacing = 1.0 / length
    resampled = np.interp(grid, phase, values).astype(np.float32)

    insert = np.searchsorted(phase, grid)
    left = np.clip(insert - 1, 0, len(phase) - 1)
    right = np.clip(insert, 0, len(phase) - 1)
    distance = np.minimum(np.abs(grid - phase[left]), np.abs(grid - phase[right]))
    mask = (distance <= VALIDITY_RADIUS * spacing).astype(np.float32)

    return np.stack([resampled * mask, mask], axis=0)


def _period_for(path: Path) -> float | None:
    """A credible period for phase folding, from the feature cache only.

    Deliberately does not run a period search. Doing so here would put a
    ~1 second Lomb-Scargle per curve back into sequence building, which is the
    exact cost feature caching was introduced to remove. A curve whose features
    have not been extracted simply falls back to time resampling.
    """
    from . import featurecache
    from .features import FEATURE_NAMES

    row = featurecache.load().get(path)
    if row is None:
        return None

    try:
        period = float(row[FEATURE_NAMES.index("best_period_days")])
        snr = float(row[FEATURE_NAMES.index("period_snr")])
    except (IndexError, ValueError):
        return None

    if not (np.isfinite(period) and period > 0):
        return None
    if not np.isfinite(snr) or snr < MIN_PHASE_PERIOD_SNR:
        return None
    return period


def resample_curve(curve: LightCurve, length: int = DEFAULT_LENGTH,
                   mode: str = "time",
                   period_days: float | None = None) -> np.ndarray | None:
    """Dispatch to one resampling mode, falling back to time.

    Falling back rather than failing keeps a batch homogeneous: a mode that
    cannot apply to one curve (a single season, no credible period) must not
    silently drop that curve and change the population being trained on.
    """
    if mode not in RESAMPLE_MODES:
        raise ValueError(f"unknown resample mode: {mode!r}")

    if mode == "season":
        seasonal = resample_by_season(curve, length)
        if seasonal is not None:
            return seasonal
    if mode == "phase" and period_days:
        folded = resample_by_phase(curve, period_days, length)
        if folded is not None:
            return folded
    return resample(curve, length)


def build(survey: str | None = None, length: int = DEFAULT_LENGTH,
          limit: int = 10_000, root: Path | None = None,
          mode: str = "time") -> SequenceBatch:
    """Build a sequence batch from the canonical store, streaming one at a time.

    `mode` selects the representation; see RESAMPLE_MODES. The default is
    unchanged, and deliberately so: sequences built under different modes are
    not comparable, so switching must be an explicit choice recorded in an
    experiment's preprocessing version rather than a quiet default change.
    """
    from . import config

    if mode not in RESAMPLE_MODES:
        raise ValueError(f"unknown resample mode: {mode!r}")

    root = root or config.PATHS.datasets
    search_root = root / survey.upper() if survey else root

    rows: list[np.ndarray] = []
    identities: list[dict] = []

    if not search_root.exists():
        return SequenceBatch(values=np.empty((0, 2, length), dtype=np.float32),
                             identities=[], length=length, mode=mode)

    # Phase mode needs periods; read the feature cache once for the whole
    # build rather than reopening it per curve.
    periods: dict[str, float] = {}
    if mode == "phase":
        from . import featurecache
        from .features import FEATURE_NAMES

        cache = featurecache.load()
        period_column = FEATURE_NAMES.index("best_period_days")
        snr_column = FEATURE_NAMES.index("period_snr")
        for path in sorted(search_root.rglob("*.parquet")):
            row = cache.get(path)
            if row is None:
                continue
            period, snr = float(row[period_column]), float(row[snr_column])
            if (np.isfinite(period) and period > 0
                    and np.isfinite(snr) and snr >= MIN_PHASE_PERIOD_SNR):
                periods[str(path)] = period

    for path in sorted(search_root.rglob("*.parquet")):
        if len(rows) >= limit:
            break
        try:
            curve = store.read_curve(path)
        except Exception:  # noqa: BLE001 - a corrupt file must not stop the build
            continue

        sequence = resample_curve(curve, length, mode, periods.get(str(path)))
        if sequence is None:
            continue

        rows.append(sequence)
        identities.append({
            "object_id": curve.source.object_id,
            "survey": curve.source.survey,
            "band": curve.band,
            "path": str(path),
        })

    values = (np.stack(rows) if rows
              else np.empty((0, 2, length), dtype=np.float32))
    return SequenceBatch(values=values.astype(np.float32),
                         identities=identities, length=length, mode=mode)


def train_test_split(batch: SequenceBatch, test_fraction: float = 0.2,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    """Split by row index, returning (train, test, train_idx, test_idx).

    Plan section 36 requires training and evaluation data to be separated
    correctly. Indices are returned so a caller can map scores back to the
    objects they came from without re-deriving the split.
    """
    n = len(batch)
    if n == 0:
        empty = np.empty((0, 2, batch.length), dtype=np.float32)
        return empty, empty, np.empty(0, dtype=int), np.empty(0, dtype=int)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = max(1, int(round(n * (1.0 - test_fraction)))) if n > 1 else 1

    train_idx, test_idx = order[:cut], order[cut:]
    return batch.values[train_idx], batch.values[test_idx], train_idx, test_idx
