"""Optional CUDA-accelerated BOCPD, batched across curves.

`profiling.profile_feature_extraction` measured `features.bocpd` as the
second-heaviest per-curve cost after the Lomb-Scargle period search (16.5%
of feature-extraction time on a real 200-curve local sample, at 75 ms/curve
-- see the measured table this entry adds to `docs/DEFERRED.txt`). Unlike
Lomb-Scargle, BOCPD's recursion is inherently serial *within* one curve: each
observation's posterior depends on the previous one. There is no frequency
axis to parallelise the way `gpu_periodogram.py` parallelises across
frequency bins.

The parallelism this module exploits instead is across CURVES: one thread
per curve, each running the exact same serial recursion `features.bocpd`
runs on CPU, entirely independently of every other thread. This is the
"streaming batches" half of backlog item 41 as much as the "custom kernel"
half -- the natural unit of GPU work here is a batch of curves, not one.

To keep the GPU kernel numerically identical to the CPU reference, the one
step that needs a full-array reduction over the whole curve (the `sigma`
robust-noise estimate, from two medians) is computed on the HOST in NumPy,
exactly as `features.bocpd` computes it, and passed into the kernel as a
precomputed per-curve scalar. The kernel then performs only the fixed-width
(`MAX_RUN_LENGTH`-bounded) recursion, reading `time`/`value` sequentially
from global memory -- no per-thread sort, no per-thread array proportional to
curve length, so a 19,499-point curve costs the kernel the same local memory
as a 300-point one.

Nothing here is imported eagerly; see `gpu_periodogram.py`'s docstring for
why (no CuPy in a released installer, CUDA present without headers, etc).
`available()` is shared in spirit with `gpu_periodogram.available()` but
kept as its own probe: compiling one kernel proves nothing about the other
compiling, and a caller must not assume they succeed or fail together.
"""

from __future__ import annotations

import math

import numpy as np

from . import config

MAX_RUN_LENGTH = 512
DEFAULT_HAZARD = 1 / 200.0

_KERNEL_SOURCE = r"""
extern "C" __global__
void bocpd_batch(const double* __restrict__ times,
                 const double* __restrict__ values,
                 const long long* __restrict__ offsets,
                 const int* __restrict__ lengths,
                 const double* __restrict__ sigmas,
                 double* __restrict__ out_change_probability,
                 double* __restrict__ out_max_probability,
                 double* __restrict__ out_change_index,
                 double* __restrict__ out_change_time,
                 const int n_curves, const double hazard, const int max_run_length)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_curves) return;

    const int n = lengths[c];
    const long long offset = offsets[c];
    const double variance = sigmas[c] * sigmas[c];
    const double norm_const = sqrt(2.0 * 3.14159265358979323846 * variance);

    // Two fixed-size buffers per state array, ping-ponged each step so a
    // write to "next" never clobbers a value "cur" still needs later in the
    // same step -- mirrors the CPU reference's separate next_run/next_means/
    // next_counts arrays exactly, just without the numpy allocation per step.
    double run_a[512], means_a[512], counts_a[512];
    double run_b[512], means_b[512], counts_b[512];
    double* run = run_a; double* means = means_a; double* counts = counts_a;
    double* next_run = run_b; double* next_means = means_b; double* next_counts = counts_b;

    int run_len = 1;
    run[0] = 1.0;
    means[0] = values[offset];
    counts[0] = 1.0;

    double best_prob = -1.0;
    long long best_index = 0;
    double last_prob = 0.0;

    for (int t = 1; t < n; ++t) {
        const double obs = values[offset + t];
        int max_len = run_len + 1;
        if (max_len > max_run_length) max_len = max_run_length;

        double change = 0.0;
        for (int j = 0; j < max_len; ++j) next_run[j] = 0.0;
        for (int k = 0; k < run_len; ++k) {
            const double d = obs - means[k];
            const double predictive = exp(-0.5 * d * d / variance) / norm_const;
            const double contribution = run[k] * predictive;
            change += contribution * hazard;
            if (k + 1 < max_len) next_run[k + 1] += contribution * (1.0 - hazard);
        }
        next_run[0] = change;

        double normalizer = 0.0;
        for (int j = 0; j < max_len; ++j) normalizer += next_run[j];
        if (!isfinite(normalizer) || normalizer <= 0.0) {
            next_run[0] = 1.0;
            for (int j = 1; j < max_len; ++j) next_run[j] = 0.0;
        } else {
            for (int j = 0; j < max_len; ++j) next_run[j] /= normalizer;
        }

        next_means[0] = obs;
        next_counts[0] = 1.0;
        int carry = run_len;
        if (carry > max_len - 1) carry = max_len - 1;
        for (int k = 0; k < carry; ++k) {
            next_counts[k + 1] = counts[k] + 1.0;
            next_means[k + 1] = (means[k] * counts[k] + obs) / next_counts[k + 1];
        }

        double* swap_run = run; run = next_run; next_run = swap_run;
        double* swap_means = means; means = next_means; next_means = swap_means;
        double* swap_counts = counts; counts = next_counts; next_counts = swap_counts;
        run_len = max_len;

        last_prob = run[0];
        if (last_prob > best_prob) { best_prob = last_prob; best_index = t; }
    }

    out_change_probability[c] = last_prob;
    out_max_probability[c] = best_prob;
    out_change_index[c] = (double) best_index;
    out_change_time[c] = times[offset + best_index];
}
"""

_KERNEL_NAME = "bocpd_batch"
_BLOCK_SIZE = 64  # small: each thread holds ~24 KB of local state (6 x 512 doubles)

_kernel_cache: "object | None" = None
_availability_cache: tuple[bool, str] | None = None


def _kernel():
    global _kernel_cache
    if _kernel_cache is None:
        import cupy as cp

        _kernel_cache = cp.RawKernel(_KERNEL_SOURCE, _KERNEL_NAME)
    return _kernel_cache


def available(force: bool = False) -> tuple[bool, str]:
    """Same three-mode probe as `gpu_periodogram.available` -- see there."""
    global _availability_cache
    if _availability_cache is not None and not force:
        return _availability_cache

    try:
        import cupy as cp
    except Exception as exc:  # noqa: BLE001 - absence is expected, not fatal
        _availability_cache = (False, f"CuPy is not installed ({exc}).")
        return _availability_cache

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            _availability_cache = (False, "No CUDA device visible to CuPy.")
            return _availability_cache
    except Exception as exc:  # noqa: BLE001 - a broken driver is not fatal
        _availability_cache = (False, f"CUDA device query failed ({exc}).")
        return _availability_cache

    config.ensure_cuda_path()
    try:
        global _kernel_cache
        if force:
            _kernel_cache = None
        _kernel()
    except Exception as exc:  # noqa: BLE001 - compilation failure is data
        _availability_cache = (
            False,
            "CUDA headers unavailable for kernel compilation "
            f"(CUDA_PATH may be stale): {exc}",
        )
        return _availability_cache

    _availability_cache = (True, "")
    return _availability_cache


def _sigma(values: np.ndarray) -> float:
    """Exactly `features.bocpd`'s robust noise estimate -- kept in one place
    conceptually, duplicated here (not imported) because `features.bocpd`
    computes it inline rather than as a separate function; a future refactor
    that extracts it there should update this copy too."""
    differences = np.diff(values)
    sigma = 1.4826 * float(np.median(np.abs(differences))) / math.sqrt(2.0)
    sigma = max(sigma, 1e-6 * max(float(np.median(np.abs(values))), 1.0))
    return sigma


def compute_batch(curves: list[tuple[np.ndarray, np.ndarray]],
                  hazard: float = DEFAULT_HAZARD,
                  max_run_length: int = MAX_RUN_LENGTH) -> list[dict[str, float]]:
    """BOCPD for a batch of (time, value) curves, one CUDA thread each.

    Every entry of `curves` must already be cleaned (finite, sorted by time)
    the way `features.bocpd` expects its caller to have done -- this function
    does not repeat that filtering. A curve with fewer than 3 points gets the
    same NaN result `features.bocpd` returns for it, without occupying a
    thread. Callers must check `available()` first; this raises on any GPU
    failure rather than degrading silently, matching `gpu_periodogram.power`.
    """
    if max_run_length > MAX_RUN_LENGTH:
        raise ValueError(
            f"max_run_length {max_run_length} exceeds the kernel's fixed "
            f"local-array size ({MAX_RUN_LENGTH}); recompile with a larger "
            "buffer to raise this.")

    import cupy as cp

    kernel = _kernel()
    nan_result = {"change_probability": float("nan"), "max_probability": float("nan"),
                 "change_index": float("nan"), "change_time": float("nan")}

    results: list[dict[str, float] | None] = [None] * len(curves)
    usable_indices: list[int] = []
    flat_times: list[float] = []
    flat_values: list[float] = []
    offsets: list[int] = []
    lengths: list[int] = []
    sigmas: list[float] = []

    for i, (time, value) in enumerate(curves):
        time = np.asarray(time, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        if len(value) < 3:
            results[i] = dict(nan_result)
            continue
        usable_indices.append(i)
        offsets.append(len(flat_values))
        lengths.append(len(value))
        sigmas.append(_sigma(value))
        flat_times.extend(time.tolist())
        flat_values.extend(value.tolist())

    if not usable_indices:
        return [r if r is not None else dict(nan_result) for r in results]

    n_curves = len(usable_indices)
    device_times = cp.asarray(np.asarray(flat_times, dtype=np.float64))
    device_values = cp.asarray(np.asarray(flat_values, dtype=np.float64))
    device_offsets = cp.asarray(np.asarray(offsets, dtype=np.int64))
    device_lengths = cp.asarray(np.asarray(lengths, dtype=np.int32))
    device_sigmas = cp.asarray(np.asarray(sigmas, dtype=np.float64))

    out_change_probability = cp.empty(n_curves, dtype=cp.float64)
    out_max_probability = cp.empty(n_curves, dtype=cp.float64)
    out_change_index = cp.empty(n_curves, dtype=cp.float64)
    out_change_time = cp.empty(n_curves, dtype=cp.float64)

    blocks = (n_curves + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    kernel((blocks,), (_BLOCK_SIZE,), (
        device_times, device_values, device_offsets, device_lengths, device_sigmas,
        out_change_probability, out_max_probability, out_change_index, out_change_time,
        np.int32(n_curves), np.float64(hazard), np.int32(max_run_length),
    ))
    cp.cuda.Stream.null.synchronize()

    change_probability = cp.asnumpy(out_change_probability)
    max_probability = cp.asnumpy(out_max_probability)
    change_index = cp.asnumpy(out_change_index)
    change_time = cp.asnumpy(out_change_time)

    for row, original_index in enumerate(usable_indices):
        results[original_index] = {
            "change_probability": float(change_probability[row]),
            "max_probability": float(max_probability[row]),
            "change_index": float(change_index[row]),
            "change_time": float(change_time[row]),
        }

    return [r if r is not None else dict(nan_result) for r in results]
