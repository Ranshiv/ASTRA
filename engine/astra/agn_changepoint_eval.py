"""Lead-time study: how early a real TDE/AGN change-point is correctly
flagged relative to the true flare peak.

Split from `agn_changepoint.py` purely to keep each file under this
project's 500-line guideline (same `stellar_manifold.py`/
`stellar_manifold_eval.py` split rationale, not an independent module).

Reuses `sn_classification.truncate_light_curve` UNCHANGED for the
cutoff-grid early-alert simulation -- the identical client-side mechanism
item 19 built for early SN classification, now applied to flagging a
change-point instead of a class label. `_summary()`'s mean/std/ci95 shape delegates to `research.stats.summary`
(migrated per docs/LIMITATIONS.md's tracked _summary-duplication debt --
this and the other eval modules in this family used to each carry their
own copy of this shape; now they share one implementation).

"Lead time" has no prior definition in this codebase or, to this module's
knowledge, one single universally agreed literature definition; the one
used here (true flare peak time MINUS the earliest cutoff time at which
the change-point is correctly flagged and remains flagged at every later
grid point) is stated explicitly as this module's own convention, the same
honesty discipline item 19's time-to-classification definition used.
Validated on SYNTHETIC injected flares atop a real DRW realization (real
cadence/noise, the mechanism-validation discipline every eval module in
this family uses) -- not yet run against a real ZTF/NEOWISE AGN sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .research import stats as research_stats

from .agn_changepoint import (
    AGNChangepointError, calibrate_changepoint_significance, changepoint_evidence,
    default_flare_guess, fit_drw, tde_flare_model,
)
from .sn_classification import truncate_light_curve


def _summary(values: list[float]) -> dict | None:
    """Delegates to `research.stats.summary` -- see that module's docstring
    for why this shape (mean/std/ci95 over repeated seeds, not object-group
    bootstrap) is the right one here. Was this module's own local
    reimplementation; migrated per docs/LIMITATIONS.md's tracked debt."""
    return research_stats.summary(values)


@dataclass(frozen=True)
class LeadTimeResult:
    cutoff_days_since_first: list[float]
    n_trials: int
    n_detected: int
    lead_time_days: dict | None

    def to_dict(self) -> dict:
        return {
            "cutoff_days_since_first": self.cutoff_days_since_first,
            "n_trials": self.n_trials, "n_detected": self.n_detected,
            "detection_rate": round(self.n_detected / self.n_trials, 4) if self.n_trials else 0.0,
            "lead_time_days": self.lead_time_days,
        }


def evaluate_lead_time(*, drw_sigma: float, drw_tau: float, flare_amplitude: float,
                       flare_rise_sigma: float, flare_t_decay_ref: float,
                       cutoff_grid_days: list[float], n_trials: int = 10,
                       span_days: float = 200.0, cadence_days: float = 1.0,
                       noise_sigma: float = 0.02, target_fpr: float = 0.01,
                       n_calibration_realizations: int = 30, seed: int = 42) -> LeadTimeResult:
    """Injects a known TDE flare (fixed shape, `t0` at 60% of `span_days`)
    onto `n_trials` independent synthetic DRW realizations, truncates each
    to every `cutoff_grid_days` value via `truncate_light_curve`, and finds
    the earliest cutoff at which `changepoint_evidence` clears a
    significance threshold calibrated (per trial) on that same DRW
    background. Reports the lead time's mean/std/ci95 across trials that
    detected the flare at all."""
    if not cutoff_grid_days:
        raise AGNChangepointError("cutoff_grid_days must be non-empty")
    if n_trials < 1:
        raise AGNChangepointError("n_trials must be at least 1")
    cutoff_grid_days = sorted(cutoff_grid_days)

    from celerite2 import GaussianProcess, terms  # local: only needed to build synthetic realizations

    time = np.arange(0.0, span_days, cadence_days)
    t0_true = 0.6 * span_days
    term = terms.RealTerm(a=drw_sigma ** 2, c=1.0 / drw_tau)
    gp = GaussianProcess(term, mean=0.0)
    gp.compute(time, diag=noise_sigma ** 2 * np.ones_like(time))
    err = np.full_like(time, noise_sigma)

    lead_times: list[float] = []
    n_detected = 0
    for trial in range(n_trials):
        np.random.seed(seed + trial)  # celerite2.sample() has no explicit-generator kwarg
        baseline = gp.sample()
        flare = tde_flare_model(time, t0_true, flare_amplitude, flare_rise_sigma, flare_t_decay_ref)
        value = baseline + flare

        # Per-cutoff (flagged, cutoff_time) pairs; `None` for a cutoff that
        # could not even be fit (too few points), never a fabricated flag.
        flags: list[tuple[bool, float] | None] = []
        for cutoff in cutoff_grid_days:
            t, v, e = truncate_light_curve(time, value, err, cutoff)
            if len(t) < 25:  # need enough points for both fit_drw and changepoint_evidence
                flags.append(None)
                continue
            try:
                drw_fit = fit_drw(t, v, e, sigma_guess=drw_sigma, tau_guess=drw_tau)
                threshold = calibrate_changepoint_significance(
                    drw_fit, t, e, n_realizations=n_calibration_realizations,
                    target_fpr=target_fpr, seed=seed + trial)
                flare_guess = default_flare_guess(t, v, drw_fit)
                evidence = changepoint_evidence(t, v, e, drw_fit, flare_guess)
            except AGNChangepointError:
                flags.append(None)
                continue
            flags.append((evidence.delta_bic <= threshold, float(t[-1])))

        # Earliest cutoff flagged AND every later cutoff also flagged --
        # the same "does not drop below at any later grid point" discipline
        # item 19's time-to-classification definition uses.
        detected_cutoff = None
        for i, flag in enumerate(flags):
            if flag is None or not flag[0]:
                continue
            if all(f is not None and f[0] for f in flags[i:]):
                detected_cutoff = flag[1]
                break

        if detected_cutoff is not None:
            n_detected += 1
            lead_times.append(t0_true - detected_cutoff)

    return LeadTimeResult(
        cutoff_days_since_first=cutoff_grid_days, n_trials=n_trials, n_detected=n_detected,
        lead_time_days=_summary(lead_times),
    )


__all__ = ["LeadTimeResult", "evaluate_lead_time"]
