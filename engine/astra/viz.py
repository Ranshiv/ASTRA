"""Serving stored light curves to the interface (plan section 29, phase 3).

A single TESS sector holds ~18,000 points and a plot is ~1,000 pixels wide, so
sending every point would move megabytes per redraw to draw the same picture.
Downsampling happens here, using largest-triangle-three-buckets rather than
naive striding: LTTB keeps the extrema that make a flare or eclipse visible,
which uniform sampling drops precisely because they are brief.

Nothing in this module modifies the store; it is a read path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, store, timeframe
from .surveys.base import LightCurve

# Roughly two points per horizontal pixel on a wide plot: dense enough that
# the curve looks continuous, small enough to stay responsive.
DEFAULT_MAX_POINTS = 2000


@dataclass(frozen=True)
class CurveSummary:
    """Enough to populate a list without reading the whole file."""

    path: str
    survey: str
    release: str
    object_id: str
    band: str
    value_kind: str
    time_system: str
    points: int
    time_span_days: float
    mean_value: float
    std_value: float

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "survey": self.survey,
            "release": self.release,
            "object_id": self.object_id,
            "band": self.band,
            "value_kind": self.value_kind,
            "time_system": self.time_system,
            "points": self.points,
            "time_span_days": round(self.time_span_days, 4),
            "mean_value": round(self.mean_value, 6),
            "std_value": round(self.std_value, 6),
        }


def list_curves(survey: str | None = None, limit: int = 500,
                root: Path | None = None) -> list[dict]:
    """Summarise stored curves, optionally filtered to one survey."""
    root = root or config.PATHS.datasets
    if not root.exists():
        return []

    search_root = root / survey.upper() if survey else root
    if not search_root.exists():
        return []

    summaries: list[dict] = []
    for path in sorted(search_root.rglob("*.parquet")):
        if len(summaries) >= limit:
            break
        try:
            summaries.append(summarise(store.read_curve(path), path).to_dict())
        except Exception:  # noqa: BLE001 - a corrupt file must not hide the rest
            continue
    return summaries


def summarise(curve: LightCurve, path: Path) -> CurveSummary:
    finite = curve.value[np.isfinite(curve.value)]
    return CurveSummary(
        path=str(path),
        survey=curve.source.survey,
        release=curve.release,
        object_id=curve.source.object_id,
        band=curve.band,
        value_kind=curve.value_kind,
        time_system=curve.time_system,
        points=len(curve),
        time_span_days=curve.time_span_days(),
        mean_value=float(np.mean(finite)) if finite.size else float("nan"),
        std_value=float(np.std(finite)) if finite.size else float("nan"),
    )


def lttb(time: np.ndarray, value: np.ndarray,
         target: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest-triangle-three-buckets downsampling.

    Keeps the point in each bucket forming the largest triangle with the
    previous kept point and the next bucket's mean, which preserves peaks and
    troughs. First and last points are always retained so the time span of the
    plot matches the time span of the data.
    """
    count = len(time)
    if target >= count or target < 3:
        return time, value

    # The first and last points are always kept, so the interior points are
    # distributed across `target - 2` buckets spanning indices 1..count-2.
    bucket_size = (count - 2) / (target - 2)
    selected = [0]
    previous = 0

    for i in range(target - 2):
        start = int(np.floor(i * bucket_size)) + 1
        end = min(int(np.floor((i + 1) * bucket_size)) + 1, count - 1)
        if start >= end:
            # A bucket narrower than one sample still contributes a point, or
            # the output would be shorter than requested.
            end = min(start + 1, count - 1)
            start = end - 1

        next_start = end
        next_end = min(int(np.floor((i + 2) * bucket_size)) + 1, count)
        if next_start >= next_end:
            avg_time = float(time[-1])
            avg_value = float(value[-1])
        else:
            avg_time = float(np.mean(time[next_start:next_end]))
            avg_value = float(np.mean(value[next_start:next_end]))

        window_time = time[start:end]
        window_value = value[start:end]
        areas = np.abs(
            (time[previous] - avg_time) * (window_value - value[previous])
            - (time[previous] - window_time) * (avg_value - value[previous])
        )
        chosen = start + int(np.argmax(areas))
        selected.append(chosen)
        previous = chosen

    selected.append(count - 1)
    index = np.asarray(selected, dtype=np.int64)
    return time[index], value[index]


def curve_payload(path: str | Path, max_points: int = DEFAULT_MAX_POINTS,
                  frame: str | None = None) -> dict:
    """Plot-ready series for one stored curve."""
    curve = store.read_curve(Path(path)).dropna().sorted_by_time()
    if frame == "BJD_TDB":
        curve = timeframe.align(curve)
    if len(curve) == 0:
        return {"points": 0, "time": [], "value": [], "value_err": [],
                "downsampled": False,
                **summarise(curve, Path(path)).to_dict()}

    time, value = lttb(curve.time, curve.value, max_points)
    downsampled = len(time) < len(curve)

    # Error bars are only meaningful alongside their own point, so they are
    # carried through the same selection rather than resampled separately.
    if downsampled:
        index = np.searchsorted(curve.time, time)
        index = np.clip(index, 0, len(curve) - 1)
        err = curve.value_err[index]
    else:
        err = curve.value_err

    payload = summarise(curve, Path(path)).to_dict()
    if frame == "BJD_TDB":
        payload["time_system"] = "BJD_TDB"
    payload.update({
        "time": [float(t) for t in time],
        "value": [float(v) for v in value],
        "value_err": [float(e) for e in err],
        "downsampled": downsampled,
        "shown_points": int(len(time)),
    })
    return payload


def fold(path: str | Path, period_days: float, epoch: float | None = None,
         max_points: int = DEFAULT_MAX_POINTS) -> dict:
    """Phase-fold a curve, which is how periodic variability becomes visible."""
    if period_days <= 0:
        raise ValueError(f"period_days must be positive: {period_days}")

    curve = store.read_curve(Path(path)).dropna().sorted_by_time()
    if len(curve) == 0:
        return {"points": 0, "phase": [], "value": [], "period_days": period_days}

    reference = epoch if epoch is not None else float(curve.time[0])
    phase = np.mod(curve.time - reference, period_days) / period_days

    order = np.argsort(phase)
    phase_sorted = phase[order]
    value_sorted = curve.value[order]

    phase_out, value_out = lttb(phase_sorted, value_sorted, max_points)

    return {
        "points": int(len(curve)),
        "shown_points": int(len(phase_out)),
        "period_days": period_days,
        "epoch": reference,
        "phase": [float(p) for p in phase_out],
        "value": [float(v) for v in value_out],
        "value_kind": curve.value_kind,
        "band": curve.band,
    }


def bin_curve(path: str | Path, bin_days: float) -> dict:
    """Average into fixed time bins, to see slow trends under fast noise."""
    if bin_days <= 0:
        raise ValueError(f"bin_days must be positive: {bin_days}")

    curve = store.read_curve(Path(path)).dropna().sorted_by_time()
    if len(curve) == 0:
        return {"bins": 0, "time": [], "value": [], "value_err": []}

    start = float(curve.time[0])
    index = ((curve.time - start) // bin_days).astype(np.int64)

    times, values, errors = [], [], []
    for bucket in np.unique(index):
        mask = index == bucket
        count = int(np.count_nonzero(mask))
        times.append(float(np.mean(curve.time[mask])))
        values.append(float(np.mean(curve.value[mask])))
        # Standard error of the mean: binning reduces noise by sqrt(N).
        errors.append(float(np.std(curve.value[mask]) / np.sqrt(count))
                      if count > 1 else float(curve.value_err[mask][0]))

    return {
        "bins": len(times),
        "bin_days": bin_days,
        "time": times,
        "value": values,
        "value_err": errors,
        "value_kind": curve.value_kind,
        "band": curve.band,
    }
