"""Tracklet linking and preliminary orbit determination (moving_objects.py).

The Gauss IOD/element-conversion tests use a synthetic, known-truth orbit
(propagated by this module's own two-body Kepler propagator and observed
from Earth's real ephemeris position via astropy's builtin, no-network
solar-system model) -- ground truth by construction, the same discipline
`artifact.calibrate_from_injection`/`tess_psf.injected_source_recovery` use
elsewhere in this codebase. MPC network access is monkeypatched, never live.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astra import moving_objects as mo

MU = mo.MU_SUN_AU3_PER_DAY2

TRUTH_ELEMENTS = {
    "semi_major_axis_au": 2.5,
    "eccentricity": 0.15,
    "inclination_deg": 10.0,
    "raan_deg": 80.0,
    "argument_of_perihelion_deg": 40.0,
    "mean_anomaly_deg": 30.0,
    "epoch_mjd": 60000.0,
}


def _observation_from_truth(mjd: float) -> mo.Observation:
    elements_t = mo.two_body_propagate(TRUTH_ELEMENTS, mjd, mu=MU)
    r_t, _ = mo.elements_to_state_vector(elements_t, mu=MU)
    earth = mo.earth_heliocentric_position_au(mjd)
    ra, dec = mo.unit_vector_to_radec(r_t - earth)
    return mo.Observation(mjd=mjd, ra_deg=ra, dec_deg=dec, observer_position_au=earth)


class TestUnitVectorRoundTrip:
    def test_radec_round_trips_through_unit_vector(self):
        vector = mo.radec_to_unit_vector(123.4, -25.6)
        ra, dec = mo.unit_vector_to_radec(vector)
        assert ra == pytest.approx(123.4, abs=1e-9)
        assert dec == pytest.approx(-25.6, abs=1e-9)

    def test_zero_vector_raises(self):
        with pytest.raises(mo.MovingObjectError):
            mo.unit_vector_to_radec(np.zeros(3))


class TestElementsStateVectorRoundTrip:
    def test_round_trips_within_tolerance(self):
        r_vec, v_vec = mo.elements_to_state_vector(TRUTH_ELEMENTS, mu=MU)
        recovered = mo.state_vector_to_elements(r_vec, v_vec, mu=MU, epoch_mjd=60000.0)
        assert recovered["semi_major_axis_au"] == pytest.approx(2.5, abs=1e-8)
        assert recovered["eccentricity"] == pytest.approx(0.15, abs=1e-8)
        assert recovered["inclination_deg"] == pytest.approx(10.0, abs=1e-6)
        assert recovered["raan_deg"] == pytest.approx(80.0, abs=1e-6)
        assert recovered["argument_of_perihelion_deg"] == pytest.approx(40.0, abs=1e-6)
        assert recovered["mean_anomaly_deg"] == pytest.approx(30.0, abs=1e-6)


class TestTwoBodyPropagate:
    def test_mean_anomaly_advances_and_wraps(self):
        propagated = mo.two_body_propagate(TRUTH_ELEMENTS, 60000.0 + 10000.0, mu=MU)
        assert 0.0 <= propagated["mean_anomaly_deg"] < 360.0
        assert propagated["epoch_mjd"] == pytest.approx(70000.0)

    def test_missing_epoch_raises(self):
        elements = dict(TRUTH_ELEMENTS)
        elements["epoch_mjd"] = None
        with pytest.raises(mo.MovingObjectError):
            mo.two_body_propagate(elements, 60001.0, mu=MU)


class TestGaussPreliminaryOrbit:
    def test_recovers_known_orbit_from_synthetic_observations(self):
        times = [60000.0, 60000.3, 60000.6]
        observations = [_observation_from_truth(t) for t in times]

        fit = mo.gauss_preliminary_orbit(observations, mu=MU)
        elements = mo.state_vector_to_elements(
            fit["r2_au"], fit["v2_au_per_day"], mu=MU, epoch_mjd=times[1])
        truth_at_mid = mo.two_body_propagate(TRUTH_ELEMENTS, times[1], mu=MU)

        assert elements["semi_major_axis_au"] == pytest.approx(
            truth_at_mid["semi_major_axis_au"], rel=1e-3)
        assert elements["eccentricity"] == pytest.approx(
            truth_at_mid["eccentricity"], abs=2e-3)
        assert elements["inclination_deg"] == pytest.approx(
            truth_at_mid["inclination_deg"], abs=1e-2)
        assert elements["raan_deg"] == pytest.approx(
            truth_at_mid["raan_deg"], abs=1e-2)
        assert elements["mean_anomaly_deg"] == pytest.approx(
            truth_at_mid["mean_anomaly_deg"], abs=1e-2)

    def test_requires_exactly_three_observations(self):
        times = [60000.0, 60000.3]
        observations = [_observation_from_truth(t) for t in times]
        with pytest.raises(mo.MovingObjectError):
            mo.gauss_preliminary_orbit(observations, mu=MU)

    def test_duplicate_times_raise(self):
        obs = _observation_from_truth(60000.0)
        with pytest.raises(mo.MovingObjectError):
            mo.gauss_preliminary_orbit([obs, obs, obs], mu=MU)


class TestAssembleTracklets:
    def _detections(self) -> list[dict]:
        times = [60000.0, 60000.1, 60000.2, 60000.3]
        rows = []
        for t in times:
            obs = _observation_from_truth(t)
            rows.append({"ra_deg": obs.ra_deg, "dec_deg": obs.dec_deg, "mjd": t,
                        "survey": "synthetic"})
        return rows

    def test_consistent_motion_is_accepted(self):
        tracklets = mo.assemble_tracklets(self._detections())
        assert len(tracklets) == 1
        assert tracklets[0]["accepted"]
        assert tracklets[0]["n_detections"] == 4

    def test_inconsistent_motion_is_rejected(self):
        rows = self._detections()
        rows[2]["ra_deg"] += 5.0  # a large, non-linear jump -- not real asteroid motion
        tracklets = mo.assemble_tracklets(rows)
        assert len(tracklets) == 1
        assert not tracklets[0]["accepted"]

    def test_a_wide_time_gap_splits_into_separate_groups(self):
        rows = self._detections()
        rows.append({"ra_deg": rows[-1]["ra_deg"], "dec_deg": rows[-1]["dec_deg"],
                    "mjd": rows[-1]["mjd"] + 30.0, "survey": "synthetic"})
        tracklets = mo.assemble_tracklets(rows)
        # The lone far-future point cannot form its own tracklet (needs >= 3).
        assert len(tracklets) == 1

    def test_fewer_than_three_detections_yields_no_tracklets(self):
        rows = self._detections()[:2]
        assert mo.assemble_tracklets(rows) == []


class TestOrbitFromTracklet:
    def test_fits_an_orbit_from_an_accepted_tracklet(self):
        times = [60000.0, 60000.1, 60000.2, 60000.3, 60000.4]
        rows = []
        for t in times:
            obs = _observation_from_truth(t)
            rows.append({"ra_deg": obs.ra_deg, "dec_deg": obs.dec_deg, "mjd": t,
                        "survey": "synthetic"})
        tracklets = mo.assemble_tracklets(rows)
        assert tracklets[0]["accepted"]

        result = mo.orbit_from_tracklet(tracklets[0])
        assert result["elements"]["semi_major_axis_au"] == pytest.approx(2.5, rel=1e-2)
        assert result["n_detections_used"] == 3
        assert result["n_detections_available"] == 5

    def test_too_few_detections_raises(self):
        with pytest.raises(mo.MovingObjectError):
            mo.orbit_from_tracklet({"detections": [{"ra_deg": 1, "dec_deg": 1, "mjd": 1}]})


class TestMpcSearchOrbits:
    def test_parses_a_list_response(self, monkeypatch):
        class _FakeResponse:
            @staticmethod
            def json():
                return [{"a": 2.77, "e": 0.076, "i": 10.6}]

        captured = {}

        def fake_get(url, params, timeout, provider, headers=None):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return _FakeResponse()

        monkeypatch.setattr(mo.netclient, "get", fake_get)
        rows = mo.mpc_search_orbits({"name": "Ceres"})

        assert rows == [{"a": 2.77, "e": 0.076, "i": 10.6}]
        assert captured["url"] == mo.MPC_SEARCH_ORBITS_URL
        assert "Authorization" in captured["headers"]
        assert captured["params"]["name"] == "Ceres"

    def test_wraps_a_dict_response_in_a_list(self, monkeypatch):
        class _FakeResponse:
            @staticmethod
            def json():
                return {"a": 2.77}

        monkeypatch.setattr(mo.netclient, "get",
                            lambda url, params, timeout, provider, headers=None: _FakeResponse())
        rows = mo.mpc_search_orbits({"name": "Ceres"})
        assert rows == [{"a": 2.77}]


class TestOrbitalElementResiduals:
    def test_computes_residuals_against_published_fields(self):
        fitted = {"semi_major_axis_au": 2.51, "eccentricity": 0.151,
                 "inclination_deg": 10.05, "raan_deg": 80.1,
                 "argument_of_perihelion_deg": 40.2}
        published = {"a": 2.5, "e": 0.15, "i": 10.0, "node": 80.0, "argper": 40.0}
        report = mo.orbital_element_residuals(fitted, published)
        assert report["residuals"]["semi_major_axis_au"] == pytest.approx(0.01)
        assert report["residuals"]["eccentricity"] == pytest.approx(0.001)

    def test_missing_field_reports_none_not_zero(self):
        report = mo.orbital_element_residuals({"semi_major_axis_au": 2.5}, {})
        assert report["residuals"]["semi_major_axis_au"] is None
        assert report["residuals"]["eccentricity"] is None
