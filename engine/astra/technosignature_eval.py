"""technosignature_eval.py: injection-recovery completeness vs. SNR and
drift rate, false-alarm rate on pure-noise spectra, and RFI-rejection
efficiency of `cadence_filter` -- all measured on `synthesize_waterfall`'s
chi-squared-with-2-dof synthetic noise (see `technosignature.py`'s module
docstring for why that statistic, not Gaussian).

Every number in this module is measured on SYNTHETIC data. There is no
real Breakthrough Listen data path (see `technosignature.py`'s `[GAP]`),
so the false-alarm rate reported here is a floor for an idealised
detector noise model, not an estimate of real-world RFI-dominated
performance -- real data's statistics are nothing like chi-squared-with-
2-dof once RFI is present, and this eval says so rather than implying
otherwise.

Not registered in `rpc.py` -- see `test_not_referenced_by_rpc` in
`tests/test_technosignature_eval.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import significance
from . import technosignature as tech


class TechnosignatureEvalError(ValueError):
    """A technosignature validation study could not be run."""


def false_alarm_rate(*, n_trials: int = 100, snr_threshold: float = tech.DEFAULT_SNR_THRESHOLD,
                     n_time: int = 16, n_freq: int = 1024, seed: int = 42) -> dict[str, Any]:
    """Hits-per-trial on pure-noise spectra at one SNR threshold, with a
    Wilson interval on the any-hit rate."""
    if n_trials <= 0:
        raise TechnosignatureEvalError("n_trials must be positive")
    n_with_hit = 0
    total_hits = 0
    for trial in range(n_trials):
        spectrum = tech.synthesize_waterfall(n_time=n_time, n_freq=n_freq, snr=0.0,
                                             seed=seed + trial)["spectrum"]
        result = tech.search(spectrum, snr_threshold=snr_threshold)
        n_hits = len(result["hits"])
        total_hits += n_hits
        if n_hits > 0:
            n_with_hit += 1
    rate = n_with_hit / n_trials
    ci95 = significance._ci_binomial(n_with_hit, n_trials)
    return {"n_trials": int(n_trials), "snr_threshold": float(snr_threshold),
           "n_trials_with_any_hit": int(n_with_hit), "any_hit_rate": round(rate, 6),
           "ci95": ci95, "total_hits": int(total_hits),
           "mean_hits_per_trial": round(total_hits / n_trials, 6)}


def false_alarm_rate_vs_threshold(*, thresholds: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0, 15.0),
                                  n_trials: int = 50, seed: int = 42) -> dict[str, Any]:
    """False-alarm rate at several thresholds -- must be monotonically
    non-increasing as the threshold rises; this module's own test asserts
    that monotonicity rather than assuming it."""
    rows = [false_positive_row(threshold, n_trials=n_trials, seed=seed) for threshold in thresholds]
    return {"thresholds": list(thresholds), "rows": rows}


def false_positive_row(threshold: float, *, n_trials: int, seed: int) -> dict[str, Any]:
    result = false_alarm_rate(n_trials=n_trials, snr_threshold=threshold, seed=seed)
    return {"snr_threshold": threshold, "any_hit_rate": result["any_hit_rate"]}


def completeness_vs_snr_and_drift(*, snr_grid: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0),
                                  drift_grid: tuple[float, ...] = (0.0, 1.0, 2.0, 3.5),
                                  n_trials_per_cell: int = 10, snr_threshold: float = tech.DEFAULT_SNR_THRESHOLD,
                                  tolerance_channels: int = 2, tolerance_drift_hz_s: float = 0.5,
                                  seed: int = 42) -> dict[str, Any]:
    """Recovery-fraction grid over injected SNR and drift rate. A
    recovery counts when a returned hit lands within `tolerance_channels`
    of the injected channel AND `tolerance_drift_hz_s` of the injected
    drift -- both tolerances stated explicitly since "recovered" is
    otherwise an arbitrary judgment call."""
    if n_trials_per_cell <= 0:
        raise TechnosignatureEvalError("n_trials_per_cell must be positive")
    rows = []
    start_channel = 512
    for injected_snr in snr_grid:
        for injected_drift in drift_grid:
            n_recovered = 0
            for trial in range(n_trials_per_cell):
                spectrum = tech.synthesize_waterfall(
                    n_time=16, n_freq=1024, drift_rate_hz_s=injected_drift, snr=injected_snr,
                    start_channel=start_channel, seed=seed + trial)["spectrum"]
                result = tech.search(spectrum, max_drift_hz_s=max(4.0, abs(injected_drift) + 1.0),
                                    snr_threshold=snr_threshold)
                recovered = any(
                    abs(hit["freq_channel_index"] - start_channel) <= tolerance_channels
                    and abs(hit["drift_rate_hz_s"] - injected_drift) <= tolerance_drift_hz_s
                    for hit in result["hits"])
                if recovered:
                    n_recovered += 1
            completeness = n_recovered / n_trials_per_cell
            ci95 = significance._ci_binomial(n_recovered, n_trials_per_cell)
            rows.append({"injected_snr": injected_snr, "injected_drift_hz_s": injected_drift,
                        "n_recovered": n_recovered, "n_trials": n_trials_per_cell,
                        "completeness": round(completeness, 6), "ci95": ci95})
    return {"rows": rows}


def cadence_rejection_efficiency(*, n_trials: int = 50, seed: int = 42) -> dict[str, Any]:
    """Inject a zero-drift RFI comb present in BOTH an ON and an OFF scan,
    plus one genuine ON-only signal, and report the fraction of trials
    where `cadence_filter` correctly keeps the genuine signal and rejects
    the RFI -- the efficiency of the single function that matters most in
    this module (see `technosignature.py`'s `cadence_filter` docstring)."""
    if n_trials <= 0:
        raise TechnosignatureEvalError("n_trials must be positive")
    n_rfi_rejected = 0
    n_signal_kept = 0
    for trial in range(n_trials):
        rfi_hit = tech.TechnosignatureHit(frequency_hz=1_400_000_000.0, drift_rate_hz_s=0.0,
                                          snr=30.0, freq_channel_index=100, drift_index=8)
        signal_hit = tech.TechnosignatureHit(frequency_hz=1_400_100_000.0, drift_rate_hz_s=1.5,
                                             snr=25.0, freq_channel_index=600, drift_index=10)
        on_scan_1 = [rfi_hit, signal_hit]
        on_scan_2 = [rfi_hit, signal_hit]
        off_scan = [rfi_hit]
        survivors = tech.cadence_filter([on_scan_1, on_scan_2], [off_scan],
                                        frequency_tolerance_hz=10.0, drift_tolerance_hz_s=0.1)
        survivor_freqs = {round(h.frequency_hz) for h in survivors}
        if round(rfi_hit.frequency_hz) not in survivor_freqs:
            n_rfi_rejected += 1
        if round(signal_hit.frequency_hz) in survivor_freqs:
            n_signal_kept += 1
    return {"n_trials": int(n_trials),
           "rfi_rejection_rate": round(n_rfi_rejected / n_trials, 6),
           "signal_retention_rate": round(n_signal_kept / n_trials, 6)}


def run_validation_study(*, n_trials: int = 30, seed: int = 42) -> dict[str, Any]:
    return {
        "false_alarm_rate": false_alarm_rate(n_trials=n_trials, seed=seed),
        "false_alarm_rate_vs_threshold": false_alarm_rate_vs_threshold(n_trials=min(n_trials, 20), seed=seed),
        "completeness": completeness_vs_snr_and_drift(n_trials_per_cell=min(n_trials, 5), seed=seed),
        "cadence_rejection_efficiency": cadence_rejection_efficiency(n_trials=n_trials, seed=seed),
    }


__all__ = [
    "TechnosignatureEvalError", "false_alarm_rate", "false_alarm_rate_vs_threshold",
    "completeness_vs_snr_and_drift", "cadence_rejection_efficiency", "run_validation_study",
]
