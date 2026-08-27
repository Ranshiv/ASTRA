"""Offset-likelihood normalization, Schechter-prior/distance-modulus
arithmetic, and Bayesian-combine correctness for `host_association.py`.

No `research` extra needed (no new optional dependency), so no
`pytest.importorskip` gate -- unlike `test_agn_changepoint.py`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

from astra import host_association as ha
from astra.surveys.base import SourceRef


# ---------------------------------------------------------------------------
# exponential_offset_likelihood
# ---------------------------------------------------------------------------

def test_exponential_offset_likelihood_integrates_to_one():
    # f(r) = (r / r_e**2) * exp(-r / r_e) is a proper radial density --
    # verifying the closed-form normalization claim in the module docstring
    # numerically, not just asserting it.
    r_e = 2.3
    area, _ = integrate.quad(lambda r: ha.exponential_offset_likelihood(r, r_e), 0.0, 200.0)
    assert area == pytest.approx(1.0, abs=1e-6)


def test_exponential_offset_likelihood_is_zero_at_the_origin_and_peaks_at_r_e():
    r_e = 1.0
    assert ha.exponential_offset_likelihood(0.0, r_e) == 0.0
    grid = np.linspace(0.01, 10.0, 2000)
    densities = [ha.exponential_offset_likelihood(r, r_e) for r in grid]
    assert grid[int(np.argmax(densities))] == pytest.approx(r_e, abs=0.02)


def test_exponential_offset_likelihood_rejects_bad_inputs():
    with pytest.raises(ha.HostAssociationError):
        ha.exponential_offset_likelihood(1.0, 0.0)
    with pytest.raises(ha.HostAssociationError):
        ha.exponential_offset_likelihood(-1.0, 1.0)


# ---------------------------------------------------------------------------
# schechter_luminosity_prior / absolute_magnitude
# ---------------------------------------------------------------------------

def test_schechter_luminosity_prior_peaks_near_m_star_and_falls_off_at_the_bright_end():
    # For alpha=-1.05 (alpha+1 < 0), the un-normalized density x**(alpha+1)*exp(-x)
    # rises steeply as M gets brighter than M* (x -> large) then is cut off
    # by the exponential -- a brighter-than-M* galaxy should score lower
    # than one near M*.
    near_star = ha.schechter_luminosity_prior(ha.DEFAULT_M_STAR)
    much_brighter = ha.schechter_luminosity_prior(ha.DEFAULT_M_STAR - 5.0)
    assert much_brighter < near_star


def test_schechter_luminosity_prior_rejects_non_finite():
    with pytest.raises(ha.HostAssociationError):
        ha.schechter_luminosity_prior(float("nan"))


def test_absolute_magnitude_matches_hand_computed_distance_modulus():
    from astropy.cosmology import Planck18

    apparent_mag, z = 18.2, 0.05
    expected = apparent_mag - Planck18.distmod(z).value
    assert ha.absolute_magnitude(apparent_mag, z) == pytest.approx(expected)


def test_absolute_magnitude_rejects_non_positive_redshift():
    with pytest.raises(ha.HostAssociationError):
        ha.absolute_magnitude(18.0, 0.0)
    with pytest.raises(ha.HostAssociationError):
        ha.absolute_magnitude(18.0, -0.1)


# ---------------------------------------------------------------------------
# associate_host
# ---------------------------------------------------------------------------

def _transient() -> SourceRef:
    return SourceRef(survey="ALeRCE", object_id="ZTF00abc", ra_deg=180.0, dec_deg=0.0)


def test_associate_host_ranks_a_close_bright_correct_z_candidate_above_a_distant_faint_one():
    transient = _transient()
    close_bright = SourceRef(survey="DES", object_id="close", ra_deg=180.0002, dec_deg=0.0,
                             extra={"r_mean": 17.0})
    far_faint = SourceRef(survey="DES", object_id="far", ra_deg=180.01, dec_deg=0.0,
                          extra={"r_mean": 22.0})

    result = ha.associate_host(
        transient, [close_bright, far_faint],
        redshifts={"close": (0.03, "sdss"), "far": (0.03, "sdss")},
        r_e_arcsec={"close": 2.0, "far": 2.0},
    )

    assert result.candidates[0].source.object_id == "close"
    assert result.candidates[0].posterior_probability > result.candidates[1].posterior_probability
    assert result.no_host_probability == pytest.approx(0.05)
    total = result.no_host_probability + sum(c.posterior_probability for c in result.candidates)
    assert total == pytest.approx(1.0)


def test_associate_host_degrades_to_offset_only_without_a_redshift():
    transient = _transient()
    candidate = SourceRef(survey="DES", object_id="only", ra_deg=180.0002, dec_deg=0.0,
                          extra={"r_mean": 17.0})
    result = ha.associate_host(transient, [candidate], r_e_arcsec={"only": 2.0})
    assert result.candidates[0].redshift is None
    assert result.candidates[0].abs_mag is None
    assert result.candidates[0].posterior_probability > 0.0


def test_associate_host_uses_the_fallback_radius_when_size_is_missing():
    transient = _transient()
    candidate = SourceRef(survey="DES", object_id="nosize", ra_deg=180.0002, dec_deg=0.0)
    result = ha.associate_host(transient, [candidate])
    assert result.candidates[0].r_e_arcsec is None  # measured size stays reported as missing
    assert result.candidates[0].posterior_probability > 0.0  # but a score was still computed


def test_associate_host_vetoes_a_flagged_foreground_star():
    transient = _transient()
    star = SourceRef(survey="DES", object_id="star", ra_deg=180.0001, dec_deg=0.0,
                     extra={"r_mean": 15.0})
    galaxy = SourceRef(survey="DES", object_id="galaxy", ra_deg=180.005, dec_deg=0.0,
                       extra={"r_mean": 20.0})
    result = ha.associate_host(
        transient, [star, galaxy], r_e_arcsec={"star": 1.0, "galaxy": 2.0},
        foreground_flags={"star": True})
    scores = {c.source.object_id: c.posterior_probability for c in result.candidates}
    assert scores["star"] == 0.0
    assert scores["galaxy"] > 0.0


def test_associate_host_favors_no_host_when_every_candidate_is_a_flagged_star():
    transient = _transient()
    star = SourceRef(survey="DES", object_id="star", ra_deg=180.0001, dec_deg=0.0)
    result = ha.associate_host(transient, [star], foreground_flags={"star": True})
    assert result.no_host_probability == pytest.approx(1.0)
    assert result.candidates[0].posterior_probability == 0.0


def test_associate_host_handles_no_candidates():
    result = ha.associate_host(_transient(), [])
    assert result.candidates == []
    assert result.no_host_probability == pytest.approx(1.0)


def test_associate_host_rejects_bad_prior_no_host():
    with pytest.raises(ha.HostAssociationError):
        ha.associate_host(_transient(), [], prior_no_host=1.0)
    with pytest.raises(ha.HostAssociationError):
        ha.associate_host(_transient(), [], prior_no_host=-0.1)


def test_host_candidate_and_result_to_dict_shape():
    transient = _transient()
    candidate = SourceRef(survey="DES", object_id="c1", ra_deg=180.0002, dec_deg=0.0,
                          extra={"r_mean": 18.0})
    result = ha.associate_host(transient, [candidate], redshifts={"c1": (0.05, "sdss")},
                               r_e_arcsec={"c1": 2.0})
    payload = result.to_dict()
    assert payload["candidates"][0]["object_id"] == "c1"
    assert payload["candidates"][0]["redshift_source"] == "sdss"
    assert "no_host_probability" in payload


def test_host_association_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "host_association" not in rpc_source
