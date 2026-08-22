"""GW coincidence evidence: skymap construction from posterior samples,
credible-region membership, time filtering, and the never-touch-the-score
contract. No network in this suite -- events/skymaps are monkeypatched or
built from synthetic in-memory HDF5 posteriors, following the house
convention already established for the catalogue cross-reference tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import candidates as candidates_mod
from astra import gw, store
from astra.surveys.base import LightCurve, SourceRef


def _write_hdf5_posterior(path, ra_deg, dec_deg, group="synth_posterior"):
    import h5py

    dtype = np.dtype([("right_ascension", "f8"), ("declination", "f8")])
    rows = np.zeros(len(ra_deg), dtype=dtype)
    rows["right_ascension"] = np.radians(ra_deg)
    rows["declination"] = np.radians(dec_deg)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(group, data=rows)


def _scattered_ra_dec(rng, center_ra, center_dec, n=20000, spread_deg=2.0):
    ra = (center_ra + rng.normal(0, spread_deg, n)) % 360
    dec = np.clip(center_dec + rng.normal(0, spread_deg, n), -89.9, 89.9)
    return ra, dec


class TestBuildSkymapFromSamples:
    def test_scattered_posterior_is_reported_as_gw_posterior(self, tmp_path):
        rng = np.random.default_rng(1)
        ra, dec = _scattered_ra_dec(rng, 180.0, 10.0)
        path = tmp_path / "posterior.hdf5"
        _write_hdf5_posterior(path, ra, dec)

        result = gw.build_skymap_from_samples(path, nside=64)
        assert result is not None
        probability, source = result
        assert source == "gw_posterior"
        assert probability.sum() == pytest.approx(1.0, abs=1e-9)

    def test_fixed_position_is_flagged_em_counterpart(self, tmp_path):
        """A near-zero-spread posterior is conditioned on a known position,
        not a GW-only localization -- see GW170817's real public release."""
        ra = np.full(5000, 197.45)
        dec = np.full(5000, -23.38)
        path = tmp_path / "posterior.hdf5"
        _write_hdf5_posterior(path, ra, dec)

        result = gw.build_skymap_from_samples(path, nside=64)
        assert result is not None
        _, source = result
        assert source == "em_counterpart_fixed"

    def test_missing_position_columns_yield_none(self, tmp_path):
        import h5py

        path = tmp_path / "posterior.hdf5"
        with h5py.File(path, "w") as handle:
            dtype = np.dtype([("mass_1", "f8")])
            handle.create_dataset("x_posterior", data=np.zeros(10, dtype=dtype))

        assert gw.build_skymap_from_samples(path, nside=64) is None

    def test_no_posterior_group_yields_none(self, tmp_path):
        import h5py

        path = tmp_path / "posterior.hdf5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("not_a_posterior", data=np.zeros(10))

        assert gw.build_skymap_from_samples(path, nside=64) is None


class TestCredibleMembership:
    """Verified directly against a real published posterior during planning:
    a true-position pixel came back at credible_level 0.032 (well inside the
    90% region), a 1-sigma offset at 0.486, and a far-away point at ~1.0."""

    def _skymap_event(self, tmp_path, monkeypatch, ra_deg, dec_deg):
        rng = np.random.default_rng(2)
        ra, dec = _scattered_ra_dec(rng, ra_deg, dec_deg)
        path = tmp_path / "posterior.hdf5"
        _write_hdf5_posterior(path, ra, dec)
        monkeypatch.setattr(gw, "skymap_path", lambda *a, **k: path)
        return ra_deg, dec_deg

    def test_true_position_is_well_inside_the_90pct_region(self, tmp_path, monkeypatch):
        ra_deg, dec_deg = self._skymap_event(tmp_path, monkeypatch, 180.0, 10.0)
        result = gw.credible_membership("EVT", "CAT", ra_deg, dec_deg, root=tmp_path)

        assert result["position_source"] == "gw_posterior"
        assert result["credible_level"] < 0.5
        assert result["in_90pct_region"] is True

    def test_far_position_is_outside_the_90pct_region(self, tmp_path, monkeypatch):
        self._skymap_event(tmp_path, monkeypatch, 180.0, 10.0)
        result = gw.credible_membership("EVT", "CAT", 0.0, -80.0, root=tmp_path)

        assert result["credible_level"] > 0.9
        assert result["in_90pct_region"] is False

    def test_missing_skymap_returns_none_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "skymap_path", lambda *a, **k: None)
        assert gw.credible_membership("EVT", "CAT", 180.0, 10.0, root=tmp_path) is None


class TestEventCatalogCache:
    def test_cache_hit_avoids_a_second_fetch(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        class _FakeDatasets:
            @staticmethod
            def find_datasets(type, catalog):
                calls["n"] += 1
                return ["GW1-v1"]

            @staticmethod
            def event_gps(name, catalog):
                return 1000.0

        import sys
        monkeypatch.setitem(sys.modules, "gwosc.datasets", _FakeDatasets)

        first = gw.fetch_event_catalog("CAT", root=tmp_path)
        second = gw.fetch_event_catalog("CAT", root=tmp_path)

        assert calls["n"] == 1
        assert [e.name for e in first] == [e.name for e in second] == ["GW1-v1"]

    def test_offline_with_no_cache_yields_an_empty_list_not_an_error(self, tmp_path):
        assert gw.fetch_event_catalog("CAT", offline=True, root=tmp_path) == []


class TestTemporalCoincidence:
    def _curve_at(self, tmp_path, jd_start, jd_end, n=30):
        source = SourceRef(survey="ZTF", object_id="obj1", ra_deg=180.0, dec_deg=10.0)
        curve = LightCurve(
            source=source, release="dr24", band="g", value_kind="mag",
            time=np.linspace(jd_start, jd_end, n), value=np.full(n, 18.0),
            value_err=np.full(n, 0.02), time_system="JD_UTC",
        )
        return store.write_curve(curve, tmp_path).path

    def test_event_inside_the_observation_span_is_coincident(self, tmp_path):
        path = self._curve_at(tmp_path, 2458000.0, 2458100.0)
        bounds = gw._candidate_time_bounds_jd(path)
        # An event's GPS time converts to some real JD; pick one landing
        # inside [2458000, 2458100] by constructing gw.GwEvent directly with
        # a GPS time whose UTC JD falls in range (verified via the real
        # conversion rather than asserted from a table).
        from astropy.time import Time
        gps = Time(2458050.0, format="jd", scale="utc").gps
        event = gw.GwEvent(name="E", catalog="C", gps_time=gps)

        assert gw._temporally_coincident(bounds, event, window_days=1.0) is True

    def test_event_far_outside_the_window_is_not_coincident(self, tmp_path):
        path = self._curve_at(tmp_path, 2458000.0, 2458100.0)
        bounds = gw._candidate_time_bounds_jd(path)
        from astropy.time import Time
        gps = Time(2458300.0, format="jd", scale="utc").gps
        event = gw.GwEvent(name="E", catalog="C", gps_time=gps)

        assert gw._temporally_coincident(bounds, event, window_days=1.0) is False

    def test_window_extends_the_boundary(self, tmp_path):
        path = self._curve_at(tmp_path, 2458000.0, 2458100.0)
        bounds = gw._candidate_time_bounds_jd(path)
        from astropy.time import Time
        gps = Time(2458105.0, format="jd", scale="utc").gps
        event = gw.GwEvent(name="E", catalog="C", gps_time=gps)

        assert gw._temporally_coincident(bounds, event, window_days=1.0) is False
        assert gw._temporally_coincident(bounds, event, window_days=10.0) is True

    def test_unreadable_curve_yields_no_bounds(self, tmp_path):
        assert gw._candidate_time_bounds_jd(tmp_path / "missing.parquet") is None


class TestApplyToCandidateNeverMovesTheScore:
    """The whole point of not wiring this into scoring.WEIGHTS: verify it."""

    def _candidate(self):
        return candidates_mod.Candidate(
            candidate_id="cand1", object_id="obj1", survey="ZTF", band="g",
            ra_deg=180.0, dec_deg=10.0,
            score={"total": 0.62, "components": {"statistical_rarity": 0.9}},
        )

    def test_score_is_byte_identical_after_a_match(self):
        candidate = self._candidate()
        before = dict(candidate.score)
        evidence = {
            "checked_events": 3, "temporally_coincident": 1,
            "coincident": [{"event": "GW1", "catalog": "C", "gps_time": 1.0,
                           "probability_density": 0.1, "credible_level": 0.2,
                           "in_90pct_region": True, "position_source": "gw_posterior"}],
            "state": "match", "window_days": 30.0,
        }

        gw._apply_to_candidate(candidate, evidence)

        assert candidate.score == before
        assert candidate.gw == evidence
        assert "GW1" in candidate.explanation["gw_coincidence"]

    def test_score_is_byte_identical_after_no_match(self):
        candidate = self._candidate()
        before = dict(candidate.score)
        evidence = {"checked_events": 5, "temporally_coincident": 0,
                   "coincident": [], "state": "no_match", "window_days": 30.0}

        gw._apply_to_candidate(candidate, evidence)

        assert candidate.score == before
        assert "No GW event coincidence" in candidate.explanation["gw_coincidence"]

    def test_unavailable_state_says_why_not_silently_zero(self):
        candidate = self._candidate()
        evidence = {"checked_events": 0, "coincident": [], "state": "unavailable",
                   "reason": "candidate light curve unreadable or empty"}

        gw._apply_to_candidate(candidate, evidence)

        assert "not checked" in candidate.explanation["gw_coincidence"]


class TestEnrichCandidateGw:
    def test_skips_the_skymap_lookup_when_temporally_incoincident(self, tmp_path, monkeypatch):
        """The time filter must run before the expensive spatial one."""
        source = SourceRef(survey="ZTF", object_id="obj1", ra_deg=180.0, dec_deg=10.0)
        curve = LightCurve(
            source=source, release="dr24", band="g", value_kind="mag",
            time=np.linspace(2458000.0, 2458010.0, 30), value=np.full(30, 18.0),
            value_err=np.full(30, 0.02), time_system="JD_UTC",
        )
        result = store.write_curve(curve, tmp_path)
        candidate = candidates_mod.Candidate(
            candidate_id="cand1", object_id="obj1", survey="ZTF", band="g",
            ra_deg=180.0, dec_deg=10.0, path=str(result.path),
        )

        calls = {"n": 0}

        def fake_membership(*a, **k):
            calls["n"] += 1
            return None

        monkeypatch.setattr(gw, "credible_membership", fake_membership)

        from astropy.time import Time
        far_gps = Time(2459000.0, format="jd", scale="utc").gps
        events = [gw.GwEvent(name="FAR", catalog="C", gps_time=far_gps)]

        evidence = gw.enrich_candidate_gw(candidate, events, window_days=1.0)

        assert calls["n"] == 0
        assert evidence["state"] == "no_match"
        assert evidence["checked_events"] == 1
        assert evidence["temporally_coincident"] == 0
