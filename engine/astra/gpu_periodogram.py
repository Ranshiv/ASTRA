"""Optional CUDA-accelerated Lomb-Scargle, exact rather than approximate.

`features.periodic_features` calls astropy with `method="fast"` -- Press &
Rybicki / Ruiz-Antolin & Townsend extirpolation, O[N log M]. That is the right
default: it is what makes a ~274,000-frequency grid affordable at all on CPU,
and it is an approximation. This module is the other option: a direct O[NM]
sum on GPU, computed in double precision throughout. `nifty-ls`'s own paper is
explicit that float32 is not recommended for Lomb-Scargle -- the problem's
condition number amplifies timing errors by O(N) -- so there is no float32
fast path here, and no trigonometric recurrence either (recurrence degrades at
the frequency counts this grid reaches).

The two paths are not required to agree bit-for-bit with each other, only each
with itself. Parity is tested against astropy's own exact path
(`method="cython"`), and measured to agree to about 1e-12 relative -- machine
precision for float64, not a tolerance chosen to make a test pass. See
`tests/test_gpu_periodogram.py`.

Nothing here is imported eagerly. `available()` is the only entry point most
callers need, and it never raises: a released installer ships no CuPy at all
(see astra-engine.spec), and a machine that has CuPy can still lack CUDA
headers to JIT the kernel against -- exactly what CUDA_PATH pointing at an
uninstalled toolkit does. `config.ensure_cuda_path()` is called before
probing, since RawKernel compilation needs headers, not just a driver.
"""

from __future__ import annotations

import numpy as np

from . import config

# Generalised (floating-mean) Lomb-Scargle, standard normalisation, matching
# astropy's LombScargle(fit_mean=True, center_data=True,
# normalization="standard") model. Zechmeister & Kuester (2009) formulation:
# weights are the per-point inverse-variance weights normalised to sum to 1,
# and power = (chi2_ref - chi2(f)) / chi2_ref computed via the 2x2 normal
# equations rather than forming chi2 directly.
_KERNEL_SOURCE = r"""
extern "C" __global__
void gls_standard(const double* __restrict__ t,
                  const double* __restrict__ y,
                  const double* __restrict__ w,
                  const double* __restrict__ freq,
                  double* __restrict__ power,
                  const int n, const int nf,
                  const double Y, const double YY)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nf) return;

    const double om = 6.283185307179586476 * freq[f];
    double C = 0.0, S = 0.0, YC = 0.0, YS = 0.0, CC = 0.0, CS = 0.0;
    for (int i = 0; i < n; ++i) {
        double phase = om * t[i];
        double c = cos(phase), s = sin(phase);
        double wi = w[i], yi = y[i];
        C  += wi * c;       S  += wi * s;
        YC += wi * yi * c;  YS += wi * yi * s;
        CC += wi * c * c;   CS += wi * c * s;
    }
    double SS = 1.0 - CC;

    // Floating-mean correction: subtract the mean's contribution from each
    // moment before solving the 2x2 system.
    double yc = YC - Y * C;
    double ys = YS - Y * S;
    double cc = CC - C * C;
    double ss = SS - S * S;
    double cs = CS - C * S;

    double det = cc * ss - cs * cs;
    if (det <= 0.0 || YY <= 0.0) {
        power[f] = 0.0;
        return;
    }
    power[f] = (ss * yc * yc - 2.0 * cs * yc * ys + cc * ys * ys) / (YY * det);
}
"""

_KERNEL_NAME = "gls_standard"
_BLOCK_SIZE = 256

# Populated lazily by `_kernel()`. A module-level cache means NVRTC
# compilation is paid once per process, not once per curve.
_kernel_cache: "object | None" = None
_availability_cache: tuple[bool, str] | None = None


def _kernel():
    """Compile (once) and return the RawKernel, raising on any failure.

    Callers must go through `available()` first in production code; this is
    the low-level entry point `available()` itself uses to prove the kernel
    actually builds, not just that CuPy imports.
    """
    global _kernel_cache
    if _kernel_cache is None:
        import cupy as cp

        _kernel_cache = cp.RawKernel(_KERNEL_SOURCE, _KERNEL_NAME)
    return _kernel_cache


def available(force: bool = False) -> tuple[bool, str]:
    """Whether the GPU periodogram can actually run here, and why not if not.

    Three distinct failure modes are checked, because they need different
    fixes: CuPy not installed (a released, CPU-only build), no CUDA device
    (no card, or the driver is unusable), and CUDA present but headers
    missing so NVRTC cannot compile (`config.ensure_cuda_path()` exists
    precisely because this machine's CUDA_PATH pointed at an uninstalled
    toolkit). A device being present does not prove a kernel can build, so
    this compiles a trivial kernel rather than only checking device count.

    Cached after the first call; pass `force=True` to re-probe after an
    environment change (used by tests).
    """
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
            _kernel_cache = None  # re-probe against a possibly-changed path
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


def power(time: np.ndarray, value: np.ndarray, value_err: np.ndarray,
         frequency: np.ndarray) -> np.ndarray:
    """Exact generalised Lomb-Scargle power at each given frequency.

    Same contract as `astropy.timeseries.LombScargle(...).power(frequency)`
    under `fit_mean=True, center_data=True, normalization="standard"` -- the
    astropy defaults `periodic_features` relies on. Callers must check
    `available()` first; this raises on any GPU failure rather than
    degrading, so a caller cannot mistake a silently-empty result for a real
    periodogram.
    """
    import cupy as cp

    kernel = _kernel()
    t = np.ascontiguousarray(time, dtype=np.float64)
    y = np.ascontiguousarray(value, dtype=np.float64)
    err = np.clip(np.ascontiguousarray(value_err, dtype=np.float64), 1e-12, None)
    freq = np.ascontiguousarray(frequency, dtype=np.float64)

    w = 1.0 / err**2
    w = w / w.sum()
    mean_y = float(np.dot(w, y))
    variance_y = float(np.dot(w, y**2) - mean_y**2)

    device_t = cp.asarray(t)
    device_y = cp.asarray(y)
    device_w = cp.asarray(w)
    device_freq = cp.asarray(freq)
    device_power = cp.empty(freq.size, dtype=cp.float64)

    blocks = (freq.size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    kernel((blocks,), (_BLOCK_SIZE,), (
        device_t, device_y, device_w, device_freq, device_power,
        np.int32(t.size), np.int32(freq.size),
        np.float64(mean_y), np.float64(variance_y),
    ))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(device_power)
