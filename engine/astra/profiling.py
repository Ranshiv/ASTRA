"""Measuring where the time actually goes (plan phase 9).

Phase 5 measured the deep models peaking at 38 MB of VRAM against ~2.3 GB
free — about 1.5% of the budget. So on this machine GPU memory is not the
constraint, and optimising it would be optimising the wrong thing.

This module exists to find the real constraint before any code is made faster.
Every function here measures; none of them optimise. Timings are wall-clock and
per-item, because throughput per curve is what determines whether a Stage C run
takes an afternoon or a week.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Timing:
    name: str
    seconds: float
    items: int = 0

    @property
    def per_item_ms(self) -> float:
        return (self.seconds / self.items * 1000.0) if self.items else 0.0

    @property
    def items_per_second(self) -> float:
        return (self.items / self.seconds) if self.seconds > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seconds": round(self.seconds, 4),
            "items": self.items,
            "per_item_ms": round(self.per_item_ms, 3),
            "items_per_second": round(self.items_per_second, 2),
        }


@dataclass
class Profile:
    timings: list[Timing] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, timing: Timing) -> None:
        self.timings.append(timing)

    @property
    def total_seconds(self) -> float:
        return sum(t.seconds for t in self.timings)

    def hotspots(self, count: int = 5) -> list[Timing]:
        return sorted(self.timings, key=lambda t: -t.seconds)[:count]

    def to_dict(self) -> dict:
        total = self.total_seconds
        return {
            "total_seconds": round(total, 3),
            "stages": [
                {**t.to_dict(),
                 "share": round(t.seconds / total, 4) if total > 0 else 0.0}
                for t in sorted(self.timings, key=lambda t: -t.seconds)
            ],
            "notes": self.notes,
        }


@contextmanager
def measure(profile: Profile, name: str, items: int = 0):
    """Time a block and record it."""
    started = time.perf_counter()
    try:
        yield
    finally:
        profile.add(Timing(name, time.perf_counter() - started, items))


def profile_feature_extraction(limit: int = 100,
                               root: Path | None = None) -> Profile:
    """Break feature extraction into its component costs.

    Lomb-Scargle is the obvious suspect: it evaluates a periodogram over a
    frequency grid for every curve, whereas the photometric statistics are a
    handful of passes over the array. Splitting them apart says whether that
    suspicion is right before anything is rewritten.
    """
    from . import config, features as features_mod, store

    root = root or config.PATHS.datasets
    profile = Profile()

    paths = sorted(root.rglob("*.parquet"))[:limit] if root.exists() else []
    if not paths:
        profile.notes.append("no stored curves to profile")
        return profile

    curves = []
    with measure(profile, "read_parquet", len(paths)):
        for path in paths:
            try:
                curves.append(store.read_curve(path))
            except Exception:  # noqa: BLE001
                continue

    if not curves:
        profile.notes.append("no readable curves")
        return profile

    prepared = []
    with measure(profile, "clean_and_sort", len(curves)):
        for curve in curves:
            prepared.append(curve.dropna().sorted_by_time())

    usable = [c for c in prepared if len(c) >= features_mod.MIN_POINTS]

    with measure(profile, "photometric_features", len(usable)):
        for curve in usable:
            features_mod.photometric_features(curve.value, curve.value_err)

    with measure(profile, "variability_indices", len(usable)):
        for curve in usable:
            features_mod.variability_indices(curve.value, curve.value_err)

    with measure(profile, "temporal_features", len(usable)):
        for curve in usable:
            features_mod.temporal_features(curve.time, curve.value)

    periodic = [c for c in usable
                if len(c) >= features_mod.MIN_POINTS_FOR_PERIOD]
    with measure(profile, "periodic_features_lombscargle", len(periodic)):
        for curve in periodic:
            features_mod.periodic_features(curve.time, curve.value,
                                           curve.value_err)

    # bocpd is a pure-Python per-observation loop (unlike the vectorised
    # numpy passes above), so it is measured separately rather than assumed
    # cheap just because it isn't Lomb-Scargle.
    with measure(profile, "bocpd", len(usable)):
        for curve in usable:
            features_mod.bocpd(curve.time, curve.value)

    total_points = sum(len(c) for c in usable)
    profile.notes.append(f"{len(usable)} curves, {total_points} total points, "
                         f"{len(periodic)} eligible for a period search")
    return profile


def profile_pipeline_stages(root: Path | None = None) -> Profile:
    """Time the end-to-end stages on the stored data."""
    from . import anomaly, featurematrix, tensors

    profile = Profile()

    matrix = None
    with measure(profile, "featurematrix_build"):
        matrix = featurematrix.build(root=root)
    if len(matrix):
        profile.timings[-1].items = len(matrix)

    if len(matrix) >= 10:
        with measure(profile, "anomaly_detect", len(matrix)):
            anomaly.detect(matrix)

    batch = None
    with measure(profile, "sequence_build"):
        batch = tensors.build(root=root)
    if len(batch):
        profile.timings[-1].items = len(batch)

    return profile


def benchmark_array_ops(size: int = 2_000_000, repeats: int = 3) -> dict:
    """Compare CPU and GPU on the array work the engine actually does.

    Plan section 26 lists array processing among the GPU tasks. This checks
    that claim on this hardware instead of assuming it: the card has ~192 GB/s
    of bandwidth over PCIe 3.0, and bandwidth-bound work plus a host-device
    round trip is often slower on the GPU than well-vectorised NumPy.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(size=size).astype(np.float32)

    def time_it(fn, n=repeats) -> float:
        fn()  # warm up
        started = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - started) / n

    results: dict = {"size": size, "cpu": {}, "gpu": {}}

    results["cpu"]["normalise"] = time_it(
        lambda: (data - np.median(data)) / (np.std(data) + 1e-12))
    results["cpu"]["moments"] = time_it(
        lambda: (float(np.mean(data)), float(np.std(data)),
                 float(np.mean(data ** 3))))
    results["cpu"]["diff"] = time_it(lambda: np.abs(np.diff(data)))

    try:
        import cupy as cp
    except Exception as exc:  # noqa: BLE001
        results["gpu"]["error"] = f"CuPy unavailable: {exc}"
        return results

    try:
        def to_gpu_and_back():
            gpu = cp.asarray(data)
            out = (gpu - cp.median(gpu)) / (cp.std(gpu) + 1e-12)
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(out)

        results["gpu"]["normalise_with_transfer"] = time_it(to_gpu_and_back)

        resident = cp.asarray(data)
        cp.cuda.Stream.null.synchronize()

        def resident_normalise():
            out = (resident - cp.median(resident)) / (cp.std(resident) + 1e-12)
            cp.cuda.Stream.null.synchronize()
            return out

        results["gpu"]["normalise_resident"] = time_it(resident_normalise)

        def transfer_only():
            gpu = cp.asarray(data)
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(gpu)

        results["gpu"]["transfer_round_trip"] = time_it(transfer_only)
    except Exception as exc:  # noqa: BLE001
        results["gpu"]["error"] = str(exc)

    cpu = results["cpu"]["normalise"]
    gpu_total = results["gpu"].get("normalise_with_transfer")
    if gpu_total:
        results["verdict"] = (
            f"CPU {cpu * 1000:.1f} ms vs GPU-with-transfer "
            f"{gpu_total * 1000:.1f} ms: "
            + ("GPU wins" if gpu_total < cpu else "CPU wins; the host-device "
               "transfer costs more than the computation saves")
        )
    return results


def benchmark_periodogram(n: int = 350, baseline_days: float = 2740.0,
                          repeats: int = 1) -> dict:
    """Compare the CPU (approximate) and GPU (exact) periodogram backends.

    Sized to the real ZTF-scale case measured elsewhere: ~350 points over a
    2740-day baseline searches roughly 270,000 frequencies, and Lomb-Scargle
    is 98.3% of feature-extraction time on it. Unlike `benchmark_array_ops`,
    which correctly found bandwidth-bound array work loses to the PCIe round
    trip, a periodogram expands a few hundred points into hundreds of
    thousands of frequency evaluations -- compute-bound, not transfer-bound --
    which is why this one is expected to win.
    """
    from astropy.timeseries import LombScargle

    from . import features as features_mod
    from . import gpu_periodogram

    rng = np.random.default_rng(0)
    curve_time = np.sort(rng.uniform(0, baseline_days, n))
    curve_value = 18.0 + 0.4 * np.sin(2 * np.pi * curve_time / 2.5)         + rng.normal(0, 0.05, n)
    curve_err = np.full(n, 0.05)

    model = LombScargle(curve_time, curve_value, np.clip(curve_err, 1e-12, None))
    max_period = baseline_days * features_mod.MAX_PERIOD_FRACTION
    frequency = model.autofrequency(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / features_mod.MIN_PERIOD_DAYS,
        samples_per_peak=features_mod.SAMPLES_PER_PEAK,
    )

    def time_it(fn) -> float:
        fn()  # warm up (also pays one-time NVRTC compilation for the GPU path)
        started = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - started) / repeats

    results: dict = {"n": n, "baseline_days": baseline_days,
                     "frequencies": int(frequency.size), "cpu": {}, "gpu": {}}
    results["cpu"]["fast_seconds"] = time_it(
        lambda: model.power(frequency, method="fast"))

    ok, reason = gpu_periodogram.available()
    if not ok:
        results["gpu"]["error"] = reason
        return results

    try:
        results["gpu"]["exact_seconds"] = time_it(
            lambda: gpu_periodogram.power(curve_time, curve_value, curve_err,
                                          frequency))
    except Exception as exc:  # noqa: BLE001 - a benchmark must not crash a probe
        results["gpu"]["error"] = str(exc)
        return results

    cpu_seconds = results["cpu"]["fast_seconds"]
    gpu_seconds = results["gpu"]["exact_seconds"]
    results["speedup"] = round(cpu_seconds / gpu_seconds, 2) if gpu_seconds > 0 else None
    results["verdict"] = (
        f"CPU (approximate) {cpu_seconds * 1000:.1f} ms vs "
        f"GPU (exact) {gpu_seconds * 1000:.1f} ms"
    )
    return results


def gpu_memory_report() -> dict:
    """What the GPU is actually holding, if there is one."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}

    if not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device"}

    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "total_mb": round(total / 1024 ** 2, 1),
        "free_mb": round(free / 1024 ** 2, 1),
        "torch_allocated_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
        "torch_reserved_mb": round(torch.cuda.memory_reserved() / 1024 ** 2, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
    }


def run_all(limit: int = 100) -> dict:
    """Full profile: stages, feature breakdown, array benchmark, GPU state."""
    return {
        "feature_extraction": profile_feature_extraction(limit).to_dict(),
        "pipeline_stages": profile_pipeline_stages().to_dict(),
        "array_ops": benchmark_array_ops(),
        "periodogram": benchmark_periodogram(),
        "gpu": gpu_memory_report(),
    }
