"""Narrowband drift (de-Doppler) technosignature search on a caller-
supplied dynamic spectrum (roadmap: astrophysics & extraterrestrial-
study feature pass).

There is no filterbank/dynamic-spectrum reader anywhere in this
codebase and none is added here. `DynamicSpectrum` is a plain
`(n_time, n_freq)` power array plus its time/frequency axes -- the
search operates on WHATEVER array a caller supplies. Two provenance
paths exist: (1) `synthesize_waterfall`, a synthetic injection generator
(drifting tone + chi-squared-with-2-dof detector noise, the correct
statistic for a square-law-detected spectrometer, not Gaussian) used for
every validated number in this module and in `technosignature_eval.py`;
(2) real Breakthrough Listen data, which this module does NOT read --
see the `[GAP]` below.

De-Doppler search: a narrowband transmitter drifts in the observer frame
at up to a few Hz/s from Earth's own rotation/orbital acceleration.
`dedrift_bruteforce` sums power along linear tracks
`f(t) = f0 + d*t` for a grid of trial drift rates `d`, the reference
algorithm every real pipeline's fast path (Taylor 1974's incoherent
tree) is checked against. Enriquez et al. (2017, ApJ 849, 104) -- the
paper behind Breakthrough Listen's `turboSETI` pipeline -- searched
`+/-4 Hz/s` at an S/N threshold of 10; both are used here as the
defaults, confirmed against a secondary source quoting that paper's own
search parameters this session.

[GAP] The O(N log N) Taylor-tree fast path (Taylor 1974, A&AS 15, 367)
is DELIBERATELY NOT IMPLEMENTED. This codebase's own standard is that a
fast path must be checked bit-for-bit against the brute-force reference
before being trusted (see the module-docstring requirement above); the
Taylor tree's edge-of-band combination rule could not be verified
bit-exact against `dedrift_bruteforce` in the time available for this
build, and shipping an unverified "fast path" that silently disagrees
with the reference implementation would be worse than not having one.
`dedrift_bruteforce` is therefore the only search path
(`O(n_drift * n_time * n_freq)`, vectorised per drift trial over numpy,
fine for the array sizes this module is validated at). A future
Taylor-tree implementation must ship its own bit-exact-agreement test
before replacing this note.

[GAP] No real Breakthrough Listen data path. `h5py` is already a core
dependency (`gw.py` uses it for GW posterior-sample files), so BL's HDF5
filterbank format is reachable without a new dependency in principle --
but the exact BL HDF5 `/data` layout and header-attribute schema has not
been verified against a real file this session, so this is a stated
live-contract gap, the same class as the Chandra `packageset` gap
already recorded in `docs/DEFERRED.txt`, not a shipped capability.
SIGPROC `.fil` filterbank is not readable at all. Also: linear drift
only (no acceleration term), incoherent summation only, single-dish only
(no interferometric coincidence), and no signal classification -- a
surviving hit is an unexplained narrowband detection, nothing more.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_MAX_DRIFT_HZ_S = 4.0
DEFAULT_SNR_THRESHOLD = 10.0


class TechnosignatureError(ValueError):
    """Raised when a dynamic spectrum or drift-search request is inadmissible."""


@dataclass(frozen=True)
class DynamicSpectrum:
    """A `(n_time, n_freq)` power waterfall plus its time/frequency axes.

    `freq_hz` may be increasing or decreasing (SIGPROC/filterbank
    convention is often descending-frequency) -- `channel_width_hz`
    carries the signed step so drift-rate sign conventions stay
    consistent regardless of axis orientation.
    """

    time_s: np.ndarray
    freq_hz: np.ndarray
    power: np.ndarray

    def __post_init__(self) -> None:
        time_s = np.asarray(self.time_s, dtype=np.float64)
        freq_hz = np.asarray(self.freq_hz, dtype=np.float64)
        power = np.asarray(self.power, dtype=np.float64)
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "freq_hz", freq_hz)
        object.__setattr__(self, "power", power)
        if time_s.ndim != 1 or freq_hz.ndim != 1:
            raise TechnosignatureError("time_s and freq_hz must be one-dimensional")
        if power.ndim != 2:
            raise TechnosignatureError("power must be two-dimensional (n_time, n_freq)")
        if power.shape != (len(time_s), len(freq_hz)):
            raise TechnosignatureError("power shape must match (len(time_s), len(freq_hz))")
        if len(time_s) < 2 or len(freq_hz) < 2:
            raise TechnosignatureError("need at least two time samples and two frequency channels")
        if not np.all(np.isfinite(power)):
            raise TechnosignatureError("power must be finite everywhere")
        # Relative tolerance, not a fixed-decimal round: `freq_hz` values
        # can sit at ~1e9 Hz (an L-band centre frequency) where absolute
        # float64 spacing already exceeds a channel width in the last few
        # decimal digits, so a fixed-decimal uniqueness check spuriously
        # rejects a perfectly uniform grid built as `f0 + arange(n)*df`.
        if not np.allclose(np.diff(time_s), time_s[1] - time_s[0], rtol=1e-9, atol=1e-12):
            raise TechnosignatureError("time_s must be uniformly sampled")
        if not np.allclose(np.diff(freq_hz), freq_hz[1] - freq_hz[0], rtol=1e-9, atol=1e-6):
            raise TechnosignatureError("freq_hz must be uniformly sampled")

    @property
    def dt_s(self) -> float:
        return float(self.time_s[1] - self.time_s[0])

    @property
    def channel_width_hz(self) -> float:
        return float(self.freq_hz[1] - self.freq_hz[0])

    def to_dict(self) -> dict[str, Any]:
        return {"time_s": self.time_s.tolist(), "freq_hz": self.freq_hz.tolist(),
               "power": self.power.tolist()}


@dataclass(frozen=True)
class TechnosignatureHit:
    frequency_hz: float
    drift_rate_hz_s: float
    snr: float
    freq_channel_index: int
    drift_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def drift_rate_grid(spectrum: DynamicSpectrum, *,
                    max_drift_hz_s: float = DEFAULT_MAX_DRIFT_HZ_S,
                    n_drift: int | None = None) -> np.ndarray:
    """Trial drift rates. Default `n_drift` gives a one-channel-over-the-
    full-integration step, the natural resolution of a linear drift
    search (Enriquez et al. 2017's own stated grid convention)."""
    if max_drift_hz_s <= 0:
        raise TechnosignatureError("max_drift_hz_s must be positive")
    total_time_s = spectrum.time_s[-1] - spectrum.time_s[0]
    if total_time_s <= 0:
        raise TechnosignatureError("spectrum must span a positive time range")
    natural_step = abs(spectrum.channel_width_hz) / total_time_s
    if n_drift is None:
        n_drift = max(3, int(round(2.0 * max_drift_hz_s / natural_step)) | 1)  # force odd -> includes 0
    return np.linspace(-max_drift_hz_s, max_drift_hz_s, int(n_drift))


def dedrift_bruteforce(spectrum: DynamicSpectrum, drift_rates_hz_s: np.ndarray) -> np.ndarray:
    """Brute-force shift-and-sum de-Doppler plane, shape `(n_drift, n_freq)`.

    For trial drift `d`, row `t`'s channels are shifted by
    `round(d * t_seconds / channel_width_hz)` before summing -- edge
    positions with no valid shifted channel are excluded from that
    trial's sum entirely (never wrapped: wrapping would fabricate signal
    at the band edge from the opposite edge, a real correctness bug this
    module's own edge tests guard against).
    """
    drift_rates_hz_s = np.asarray(drift_rates_hz_s, dtype=np.float64)
    n_time, n_freq = spectrum.power.shape
    df = spectrum.channel_width_hz
    if df == 0:
        raise TechnosignatureError("channel_width_hz must be nonzero")
    freq_idx = np.arange(n_freq)
    plane = np.empty((len(drift_rates_hz_s), n_freq), dtype=np.float64)
    for i, drift in enumerate(drift_rates_hz_s):
        shift_per_row = np.round(drift * spectrum.time_s / df).astype(np.int64)
        cols = freq_idx[None, :] + shift_per_row[:, None]
        valid = (cols >= 0) & (cols < n_freq)
        clipped = np.clip(cols, 0, n_freq - 1)
        gathered = np.take_along_axis(spectrum.power, clipped, axis=1)
        plane[i] = np.sum(np.where(valid, gathered, 0.0), axis=0)
    return plane


def find_hits(dedrift_plane: np.ndarray, spectrum: DynamicSpectrum, drift_rates_hz_s: np.ndarray, *,
             snr_threshold: float = DEFAULT_SNR_THRESHOLD, n_neighbour: int = 2
             ) -> list[TechnosignatureHit]:
    """Robust-baseline SNR thresholding with greedy non-max suppression.

    Baseline/scale via `1.4826 * MAD`, the same robust-statistics idiom
    `spectral_features.py` already uses elsewhere in this codebase --
    correct for the plane's non-Gaussian (summed chi-squared) statistics,
    which a plain mean/std baseline would be biased by by outlier hits.
    """
    plane = np.asarray(dedrift_plane, dtype=np.float64)
    median = float(np.median(plane))
    mad = float(np.median(np.abs(plane - median)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        return []
    snr_plane = (plane - median) / sigma

    candidates = np.argwhere(snr_plane >= snr_threshold)
    if candidates.size == 0:
        return []
    order = np.argsort(-snr_plane[candidates[:, 0], candidates[:, 1]])
    accepted: list[TechnosignatureHit] = []
    suppressed = np.zeros(plane.shape, dtype=bool)
    for idx in order:
        drift_idx, freq_idx = candidates[idx]
        if suppressed[drift_idx, freq_idx]:
            continue
        accepted.append(TechnosignatureHit(
            frequency_hz=float(spectrum.freq_hz[freq_idx]),
            drift_rate_hz_s=float(drift_rates_hz_s[drift_idx]),
            snr=float(snr_plane[drift_idx, freq_idx]),
            freq_channel_index=int(freq_idx), drift_index=int(drift_idx)))
        d_lo, d_hi = max(0, drift_idx - n_neighbour), min(plane.shape[0], drift_idx + n_neighbour + 1)
        f_lo, f_hi = max(0, freq_idx - n_neighbour), min(plane.shape[1], freq_idx + n_neighbour + 1)
        suppressed[d_lo:d_hi, f_lo:f_hi] = True
    return accepted


def search(spectrum: DynamicSpectrum, *, max_drift_hz_s: float = DEFAULT_MAX_DRIFT_HZ_S,
          n_drift: int | None = None, snr_threshold: float = DEFAULT_SNR_THRESHOLD,
          n_neighbour: int = 2) -> dict[str, Any]:
    """End-to-end: drift grid -> de-Doppler plane -> hits."""
    drift_rates = drift_rate_grid(spectrum, max_drift_hz_s=max_drift_hz_s, n_drift=n_drift)
    plane = dedrift_bruteforce(spectrum, drift_rates)
    hits = find_hits(plane, spectrum, drift_rates, snr_threshold=snr_threshold,
                     n_neighbour=n_neighbour)
    return {"schema_version": SCHEMA_VERSION, "n_drift_trials": len(drift_rates),
           "max_drift_hz_s": float(max_drift_hz_s), "snr_threshold": float(snr_threshold),
           "hits": [hit.to_dict() for hit in hits]}


def cadence_filter(on_hit_lists: list[list[TechnosignatureHit]],
                   off_hit_lists: list[list[TechnosignatureHit]], *,
                   frequency_tolerance_hz: float, drift_tolerance_hz_s: float
                   ) -> list[TechnosignatureHit]:
    """ON/OFF cadence RFI rejection: keep a hit only when a matching hit
    (within tolerance) is present in EVERY ON scan and absent from EVERY
    OFF scan (Enriquez et al. 2017 / Price et al. 2020's cadence
    criterion) -- the single function that turns a hit list into a
    technosignature CANDIDATE list; everything else here is noise-
    statistics bookkeeping."""
    if not on_hit_lists:
        raise TechnosignatureError("at least one ON scan is required")

    def _matches(a: TechnosignatureHit, b: TechnosignatureHit) -> bool:
        return (abs(a.frequency_hz - b.frequency_hz) <= frequency_tolerance_hz
               and abs(a.drift_rate_hz_s - b.drift_rate_hz_s) <= drift_tolerance_hz_s)

    survivors: list[TechnosignatureHit] = []
    for hit in on_hit_lists[0]:
        in_every_on = all(any(_matches(hit, other) for other in scan) for scan in on_hit_lists[1:])
        in_any_off = any(any(_matches(hit, other) for other in scan) for scan in off_hit_lists)
        if in_every_on and not in_any_off:
            survivors.append(hit)
    return survivors


def synthesize_waterfall(*, n_time: int = 16, n_freq: int = 1024, f0_hz: float = 1.4e9,
                         channel_width_hz: float = 2.7939677, dt_s: float = 18.25,
                         drift_rate_hz_s: float = 0.0, snr: float = 0.0,
                         start_channel: int | None = None, seed: int = 42) -> dict[str, Any]:
    """Synthetic drifting-tone injection on chi-squared-with-2-dof
    detector noise (the correct statistic for a square-law-detected
    spectrometer, per the module docstring). `n_freq`/`channel_width_hz`/
    `dt_s` defaults approximate a Breakthrough Listen GBT high-spectral-
    resolution data product (Lebofsky et al. 2019, PASP 131, 064505),
    confirmed against a secondary source this session; used only to make
    the synthetic array's scale realistic, not as a live-data claim.
    """
    if n_time < 2 or n_freq < 2:
        raise TechnosignatureError("n_time and n_freq must each be at least 2")
    rng = np.random.default_rng(seed)
    # A square-law detector's power is chi-squared with 2 dof (i.e.
    # exponential with mean 2) -- NOT Gaussian; this matters for the
    # false-alarm calibration in `technosignature_eval.py`.
    power = rng.exponential(scale=1.0, size=(n_time, n_freq))

    time_s = np.arange(n_time, dtype=np.float64) * dt_s
    freq_hz = f0_hz + np.arange(n_freq, dtype=np.float64) * channel_width_hz
    if snr > 0:
        start_channel = n_freq // 2 if start_channel is None else int(start_channel)
        for t_idx, t in enumerate(time_s):
            shift = drift_rate_hz_s * t / channel_width_hz
            channel = start_channel + int(round(shift))
            if 0 <= channel < n_freq:
                power[t_idx, channel] += snr

    spectrum = DynamicSpectrum(time_s=time_s, freq_hz=freq_hz, power=power)
    return {"spectrum": spectrum, "truth": {"drift_rate_hz_s": drift_rate_hz_s, "snr": snr,
                                            "start_channel": start_channel}}


__all__ = [
    "TechnosignatureError", "DynamicSpectrum", "TechnosignatureHit",
    "DEFAULT_MAX_DRIFT_HZ_S", "DEFAULT_SNR_THRESHOLD",
    "drift_rate_grid", "dedrift_bruteforce", "find_hits", "search",
    "cadence_filter", "synthesize_waterfall",
]
