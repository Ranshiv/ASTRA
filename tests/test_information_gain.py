"""Binary-entropy arithmetic, ranking, and `followup.plan` integration
correctness for `information_gain.py`. No `research` extra needed (no
new optional dependency).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astra import information_gain as ig


# ---------------------------------------------------------------------------
# posterior_entropy / entropy_from_tail_probability
# ---------------------------------------------------------------------------

def test_posterior_entropy_is_zero_at_the_extremes():
    assert ig.posterior_entropy(0.0) == 0.0
    assert ig.posterior_entropy(1.0) == 0.0


def test_posterior_entropy_is_one_bit_at_one_half():
    assert ig.posterior_entropy(0.5) == pytest.approx(1.0)


def test_posterior_entropy_rejects_out_of_range_probability():
    with pytest.raises(ig.InformationGainError):
        ig.posterior_entropy(1.5)
    with pytest.raises(ig.InformationGainError):
        ig.posterior_entropy(-0.1)


def test_entropy_from_tail_probability_matches_posterior_entropy():
    assert ig.entropy_from_tail_probability(0.3) == ig.posterior_entropy(0.3)


# ---------------------------------------------------------------------------
# rank_by_information_gain
# ---------------------------------------------------------------------------

def _permissive_kwargs(**overrides):
    kwargs = {"duration_hours": 24, "min_altitude_deg": 20.0,
             "twilight_sun_altitude_deg": 10.0, "cadence_minutes": 30}
    kwargs.update(overrides)
    return kwargs


def test_rank_by_information_gain_ranks_feasible_before_infeasible():
    items = [
        {"candidate_id": "hidden", "ra_deg": 180.0, "dec_deg": -85.0, "tail_probability": 0.5},
        {"candidate_id": "visible", "ra_deg": 180.0, "dec_deg": 80.0, "tail_probability": 0.5},
    ]
    ranked = ig.rank_by_information_gain(items, followup_kwargs=_permissive_kwargs())
    assert ranked[0].candidate_id == "visible"
    assert ranked[0].observable is True
    assert ranked[1].candidate_id == "hidden"
    assert ranked[1].observable is False
    assert ranked[1].value_per_hour is None


def test_rank_by_information_gain_orders_by_entropy_per_hour():
    items = [
        {"candidate_id": "confident", "ra_deg": 180.0, "dec_deg": 80.0, "tail_probability": 0.02},
        {"candidate_id": "ambiguous", "ra_deg": 180.0, "dec_deg": 80.0, "tail_probability": 0.5},
    ]
    ranked = ig.rank_by_information_gain(items, followup_kwargs=_permissive_kwargs())
    assert ranked[0].candidate_id == "ambiguous"
    assert ranked[0].value_per_hour > ranked[1].value_per_hour


def test_rank_by_information_gain_divides_by_exposure_hours():
    item = {"candidate_id": "a", "ra_deg": 180.0, "dec_deg": 80.0, "tail_probability": 0.5}
    cheap = ig.rank_by_information_gain([item], exposure_hours=1.0, followup_kwargs=_permissive_kwargs())
    expensive = ig.rank_by_information_gain([item], exposure_hours=4.0, followup_kwargs=_permissive_kwargs())
    assert cheap[0].value_per_hour == pytest.approx(expensive[0].value_per_hour * 4.0)


def test_rank_by_information_gain_rejects_bad_exposure_hours():
    with pytest.raises(ig.InformationGainError):
        ig.rank_by_information_gain([{"candidate_id": "a", "ra_deg": 0, "dec_deg": 0,
                                      "tail_probability": 0.5}], exposure_hours=0.0)


def test_rank_by_information_gain_requires_candidate_id_and_posterior():
    with pytest.raises(ig.InformationGainError):
        ig.rank_by_information_gain([{"ra_deg": 0, "dec_deg": 0, "tail_probability": 0.5}])
    with pytest.raises(ig.InformationGainError):
        ig.rank_by_information_gain([{"candidate_id": "a", "ra_deg": 0, "dec_deg": 0}])


def test_rank_by_information_gain_handles_empty_items():
    assert ig.rank_by_information_gain([]) == []


def test_rank_by_information_gain_accepts_conformal_p_value_key():
    items = [{"candidate_id": "a", "ra_deg": 180.0, "dec_deg": 80.0, "p_value": 0.3}]
    ranked = ig.rank_by_information_gain(items, followup_kwargs=_permissive_kwargs())
    assert ranked[0].value_per_hour == pytest.approx(ig.posterior_entropy(0.3))


def test_information_gain_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "information_gain" not in rpc_source
