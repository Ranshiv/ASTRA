"""Reviewer human-factors experiment (Direction 6 of the research plan
adopted 2026-08-29: "the review UI as a controlled experiment").

The claim to establish: displaying ASTRA's own score measurably biases a
human reviewer's vetting decision, and correcting for that bias changes
what the human+model ensemble is actually worth. `active_review.py`
already tracks reviewer agreement as a quality signal; this module is the
first to treat what a reviewer SEES as an experimental variable rather than
a constant.

Three arms, assigned per (reviewer, candidate) pair:

- `score_shown`   -- the real ASTRA score is displayed, today's behaviour.
- `score_blinded` -- no score is displayed at all.
- `score_shuffled` -- a DECOY score, real but belonging to a different
  candidate, is displayed. This is the arm that actually separates
  "the reviewer anchors on whatever number they see" from "the number is
  informative": if `score_shuffled` reviewers agree with `score_blinded`
  reviewers but not with `score_shown` reviewers, that is anchoring on a
  number, not signal from that number.

Assignment is a deterministic seeded hash of `(seed, reviewer_id,
candidate_id)`, never a stored random draw -- the same pair always gets the
same arm on every call, so re-opening a candidate does not silently
reshuffle a reviewer already mid-decision, and the assignment is fully
reconstructible from the vote row alone for later analysis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from . import candidates as candidates_mod

Arm = Literal["score_shown", "score_blinded", "score_shuffled"]

ARMS: tuple[Arm, ...] = ("score_shown", "score_blinded", "score_shuffled")

DEFAULT_ARM_SEED = 0

# Declared before any experimental label is collected, per the research
# plan's own requirement -- a human-subjects claim is only defensible if
# the analysis was fixed in advance, not selected after seeing the data.
# `save_preregistration` refuses to silently overwrite this with a
# DIFFERENT plan once one has been saved (see that function's docstring).
PREREGISTERED_ANALYSIS_PLAN: dict = {
    "hypothesis": (
        "Displaying ASTRA's own score measurably biases a human reviewer's "
        "vetting decision, relative to a blinded reviewer with no score."
    ),
    "arms": list(ARMS),
    "primary_metric": "anchoring_signature_detected",
    "primary_analysis": "review_experiment_eval.anchoring_effect_size",
    "secondary_analyses": [
        "review_experiment_eval.calibration_curve",
        "review_experiment_eval.ensemble_accounting",
    ],
    "minimum_votes_per_arm": 30,
    "exclusion_criteria": [
        "votes with arm=None (cast outside the experiment)",
        "votes for a candidate with no resolved ground truth",
        "votes with a label outside review.POSITIVE | review.NEGATIVE",
    ],
    "stopping_rule": (
        "Collect until every arm reaches minimum_votes_per_arm; do not stop "
        "early based on an interim look at anchoring_signature_detected."
    ),
}


def _content_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _preregistration_path() -> Path:
    from .research.store import research_root
    directory = research_root() / "experiments" / "preregistrations"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "review_experiment.json"


def save_preregistration() -> dict:
    """Write `PREREGISTERED_ANALYSIS_PLAN` to disk with its content hash,
    once, before any experimental label is collected.

    Refuses to overwrite an existing file whose hash differs -- silently
    replacing an already-registered plan after labels may already have
    been collected under it would defeat the entire point of pre-
    registration. Calling this again with the SAME plan (the normal case:
    every session that starts the experiment calls it) is a no-op that
    returns the existing record unchanged.
    """
    path = _preregistration_path()
    content_hash = _content_hash(PREREGISTERED_ANALYSIS_PLAN)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("content_hash") != content_hash:
            raise ValueError(
                "a different review-experiment analysis plan is already "
                f"registered at {path} (hash {existing.get('content_hash')}); "
                "refusing to overwrite a pre-registered plan")
        return existing
    record = {"plan": PREREGISTERED_ANALYSIS_PLAN, "content_hash": content_hash}
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def load_preregistration() -> dict | None:
    path = _preregistration_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_index(*parts: str, modulus: int) -> int:
    """A deterministic index in `[0, modulus)` from the given parts, stable
    across processes and Python versions (unlike the builtin `hash()`,
    which is salted per-process for strings)."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def assign_arm(reviewer_id: str, candidate_id: str, *, seed: int = DEFAULT_ARM_SEED) -> Arm:
    """The experimental condition this reviewer sees for this candidate.

    Deterministic and reproducible: calling this twice with the same
    inputs always returns the same arm, so the assignment can be recorded
    once (in the vote row's `arm` column) and independently re-derived
    later from the same three inputs as a consistency check.
    """
    index = _stable_index(str(seed), reviewer_id, candidate_id, modulus=len(ARMS))
    return ARMS[index]


def pick_decoy_candidate_id(reviewer_id: str, candidate_id: str,
                            pool_candidate_ids: list[str], *,
                            seed: int = DEFAULT_ARM_SEED) -> str | None:
    """The OTHER candidate whose score is shown to this reviewer for this
    candidate, in the `score_shuffled` arm.

    Deterministic given the same `(reviewer_id, candidate_id, seed)` and
    pool -- but the pool itself will grow over a real review campaign, so
    the caller must pass `displayed_score` (the actual number shown) back
    into `candidates.cast_label_vote` rather than relying on this function
    being re-derivable after the pool has changed. Returns `None` when no
    other candidate is available to serve as a decoy.
    """
    others = [cid for cid in pool_candidate_ids if cid != candidate_id]
    if not others:
        return None
    index = _stable_index(str(seed), reviewer_id, candidate_id, "decoy", modulus=len(others))
    return sorted(others)[index]


def displayed_score_for(arm: Arm, real_score: float | None,
                        decoy_score: float | None) -> float | None:
    """What the reviewer actually sees on screen for this arm."""
    if arm == "score_shown":
        return real_score
    if arm == "score_shuffled":
        return decoy_score
    return None  # score_blinded


def cast_experimental_vote(candidate_id: str, reviewer_id: str, label: str, *,
                           score_lookup: dict[str, float],
                           decision_latency_ms: int | None = None,
                           self_reported_confidence: float | None = None,
                           presentation_index: int | None = None,
                           note: str = "", root=None,
                           seed: int = DEFAULT_ARM_SEED) -> dict:
    """Cast one vote under this experiment: resolve the reviewer's arm and
    the score they actually saw, then record both alongside the vote via
    `candidates.cast_label_vote`'s additive experiment columns.

    `score_lookup` maps every candidate ID in the reviewer's current pool
    (this candidate included) to its real ASTRA score -- the pool a decoy
    can be drawn from for the `score_shuffled` arm. A candidate missing
    from `score_lookup` yields a `None` score, exactly `score_blinded`'s
    own "nothing displayed" value, rather than raising.

    This is the ONLY intended way to cast a vote that participates in the
    experiment -- `candidates.cast_label_vote` itself is left completely
    unchanged for ordinary, non-experimental use, and a vote cast directly
    through it (arm=None) is correctly excluded from every analysis in
    `review_experiment_eval.py`.
    """
    arm = assign_arm(reviewer_id, candidate_id, seed=seed)
    real_score = score_lookup.get(candidate_id)
    decoy_id = None
    decoy_score = None
    if arm == "score_shuffled":
        decoy_id = pick_decoy_candidate_id(
            reviewer_id, candidate_id, list(score_lookup.keys()), seed=seed)
        if decoy_id is not None:
            decoy_score = score_lookup.get(decoy_id)

    displayed = displayed_score_for(arm, real_score, decoy_score)
    vote = candidates_mod.cast_label_vote(
        candidate_id, reviewer_id, label, note, root, arm=arm,
        displayed_score=displayed, decision_latency_ms=decision_latency_ms,
        self_reported_confidence=self_reported_confidence,
        presentation_index=presentation_index)
    vote["decoy_candidate_id"] = decoy_id
    return vote
