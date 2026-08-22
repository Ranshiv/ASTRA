"""FRB coincidence evidence: catalogue parsing, error-ellipse membership,
sparse-HEALPix localization membership, time filtering, and the
never-touch-the-score contract. No network in this suite -- bursts/maps are
synthetic or monkeypatched, following the house convention established for
gw.py and catalogs.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import candidates as candidates_mod
from astra import frb, store
from astra.surveys.base import LightCurve, SourceRef


def _burst(tns_name="FRB1", ra_deg=180.0, ra_err_deg=0.05, dec_deg=10.0,
          dec_err_deg=0.03, mjd_400=58800.0, localization_id=None,
          repeater_name=""):
    return frb.FrbBurst(tns_name=tns_name, repeater_name=repeater_name,
                        ra_deg=ra_deg, ra_err_deg=ra_err_deg, dec_deg=dec_deg,
                        dec_err_deg=dec_err_deg, mjd_400=mjd_400,
                        localization_id=localization_id)


class TestParseCatalogCsv:
    VALID_CSV = (
        "tns_name,repeater_name,ra,ra_err,dec,dec_err,mjd_400,excluded_flag\n"
        "FRB20200101A,,180.0,0.05,10.0,0.03,58800.0,0\n"
        "FRB20200102A,,181.0,0.05,11.0,0.03,58801.0,1\n"
        "FRB20200103A,,182.0,bad,12.0,0.03,58802.0,0\n"
    )

    def test_excludes_flagged_and_malformed_rows(self):
        bursts = frb._parse_catalog_csv(self.VALID_CSV)
        assert [b.tns_name for b in bursts] == ["FRB20200101A"]

    def test_empty_payload_yields_no_bursts(self):
        assert frb._parse_catalog_csv("") == []

    def test_missing_column_is_skipped_not_fatal(self):
        csv_text = "tns_name,ra,dec,mjd_400\nFRB1,180.0,10.0,58800.0\n"
        assert frb._parse_catalog_csv(csv_text) == []  # no ra_err/dec_err


class TestWithinErrorEllipse:
    def test_exact_position_is_inside(self):
        inside, offset = frb.within_error_ellipse(180.0, 10.0, _burst())
        assert inside is True
        assert offset == pytest.approx(0.0, abs=1e-9)

    def test_one_sigma_offset_is_inside_the_default_threshold(self):
        burst = _burst()
        inside, offset = frb.within_error_ellipse(
            180.0 + burst.ra_err_deg / np.cos(np.radians(10.0)), 10.0, burst)
        assert inside is True
        assert offset == pytest.approx(1.0, abs=1e-3)

    def test_far_position_is_outside(self):
        inside, offset = frb.within_error_ellipse(190.0, 10.0, _burst())
        assert inside is False
        assert offset > frb.DEFAULT_SIGMA_THRESHOLD

    def test_ra_offset_is_scaled_by_cos_dec(self):
        """Same raw RA offset reads as a smaller physical distance at high dec."""
        low_dec_burst = _burst(dec_deg=0.0)
        high_dec_burst = _burst(dec_deg=80.0)
        _, offset_low = frb.within_error_ellipse(180.1, 0.0, low_dec_burst)
        _, offset_high = frb.within_error_ellipse(180.1, 80.0, high_dec_burst)
        assert offset_high < offset_low


class TestLocalizationMembership:
    def _write_sparse_map(self, root, localization_id, pixels, levels, nside=4096):
        import h5py

        path = root / "chimefrb" / f"loc_{localization_id}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("ipix", data=np.array(pixels, dtype=np.int64))
            handle.create_dataset("CL", data=np.array(levels, dtype=np.float32))
        return path

    def _true_pixel(self, ra_deg, dec_deg, nside=4096):
        import astropy.units as u
        from astropy_healpix import HEALPix

        healpix = HEALPix(nside=nside, order="nested")
        return int(healpix.lonlat_to_healpix(ra_deg * u.deg, dec_deg * u.deg))

    def test_true_pixel_reports_its_confidence_level(self, tmp_path):
        pixel = self._true_pixel(180.0, 10.0)
        self._write_sparse_map(tmp_path, "FRB1", [pixel, pixel + 1], [0.1, 0.5])
        burst = _burst(localization_id="FRB1")

        result = frb.localization_membership(burst, 180.0, 10.0, root=tmp_path)

        assert result["confidence_level"] == pytest.approx(0.1, abs=1e-4)
        assert result["in_90pct_region"] is True

    def test_position_outside_the_sparse_map_is_least_confident(self, tmp_path):
        pixel = self._true_pixel(180.0, 10.0)
        self._write_sparse_map(tmp_path, "FRB1", [pixel], [0.1])
        burst = _burst(localization_id="FRB1")

        result = frb.localization_membership(burst, 0.0, -80.0, root=tmp_path)

        assert result["confidence_level"] == 1.0
        assert result["in_90pct_region"] is False

    def test_burst_with_no_localization_product_returns_none(self, tmp_path):
        burst = _burst(localization_id=None)
        assert frb.localization_membership(burst, 180.0, 10.0, root=tmp_path) is None

    def test_missing_map_file_returns_none_not_a_crash(self, tmp_path, monkeypatch):
        burst = _burst(localization_id="NEVER_DOWNLOADED")
        monkeypatch.setattr(frb.netclient, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))
        assert frb.localization_membership(burst, 180.0, 10.0, root=tmp_path) is None


class TestEventCatalogCache:
    def test_offline_with_no_cache_yields_an_empty_list_not_an_error(self, tmp_path):
        assert frb.fetch_burst_catalog(offline=True, root=tmp_path) == []

    def test_cache_hit_avoids_a_second_fetch(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        class _FakeResponse:
            text = TestParseCatalogCsv.VALID_CSV

        def fake_get(url, params, timeout, provider):
            calls["n"] += 1
            return _FakeResponse()

        monkeypatch.setattr(frb.netclient, "get", fake_get)

        first = frb.fetch_burst_catalog(root=tmp_path)
        second = frb.fetch_burst_catalog(root=tmp_path)

        assert calls["n"] == 1
        assert len(first) == len(second) == 1


class TestTemporalCoincidence:
    def _curve_at(self, tmp_path, jd_start, jd_end, n=30):
        source = SourceRef(survey="ZTF", object_id="obj1", ra_deg=180.0, dec_deg=10.0)
        curve = LightCurve(
            source=source, release="dr24", band="g", value_kind="mag",
            time=np.linspace(jd_start, jd_end, n), value=np.full(n, 18.0),
            value_err=np.full(n, 0.02), time_system="JD_UTC",
        )
        return store.write_curve(curve, tmp_path).path

    def test_burst_inside_the_observation_span_is_coincident(self, tmp_path):
        path = self._curve_at(tmp_path, 2458000.0, 2458100.0)
        bounds = frb._candidate_time_bounds_jd(path)
        burst = _burst(mjd_400=2458050.0 - 2_400_000.5)

        assert frb._temporally_coincident(bounds, burst, window_days=1.0) is True

    def test_burst_far_outside_the_window_is_not_coincident(self, tmp_path):
        path = self._curve_at(tmp_path, 2458000.0, 2458100.0)
        bounds = frb._candidate_time_bounds_jd(path)
        burst = _burst(mjd_400=2458300.0 - 2_400_000.5)

        assert frb._temporally_coincident(bounds, burst, window_days=1.0) is False

    def test_window_extends_the_boundary(self, tmp_path):
        path = self._curve_at(tmp_path, 2458000.0, 2458100.0)
        bounds = frb._candidate_time_bounds_jd(path)
        burst = _burst(mjd_400=2458105.0 - 2_400_000.5)

        assert frb._temporally_coincident(bounds, burst, window_days=1.0) is False
        assert frb._temporally_coincident(bounds, burst, window_days=10.0) is True

    def test_unreadable_curve_yields_no_bounds(self, tmp_path):
        assert frb._candidate_time_bounds_jd(tmp_path / "missing.parquet") is None


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
            "checked_bursts": 3, "temporally_coincident": 1,
            "coincident": [{"burst": "FRB1", "repeater_name": "", "mjd_400": 58800.0,
                           "sigma_offset": 0.5, "sigma_threshold": 3.0,
                           "position_source": "ellipse"}],
            "state": "match", "window_days": 1.0, "sigma_threshold": 3.0,
        }

        frb._apply_to_candidate(candidate, evidence)

        assert candidate.score == before
        assert candidate.frb == evidence
        assert "FRB1" in candidate.explanation["frb_coincidence"]

    def test_score_is_byte_identical_after_no_match(self):
        candidate = self._candidate()
        before = dict(candidate.score)
        evidence = {"checked_bursts": 5, "temporally_coincident": 0,
                   "coincident": [], "state": "no_match", "window_days": 1.0,
                   "sigma_threshold": 3.0}

        frb._apply_to_candidate(candidate, evidence)

        assert candidate.score == before
        assert "No FRB coincidence" in candidate.explanation["frb_coincidence"]

    def test_unavailable_state_says_why_not_silently_zero(self):
        candidate = self._candidate()
        evidence = {"checked_bursts": 0, "coincident": [], "state": "unavailable",
                   "reason": "candidate light curve unreadable or empty"}

        frb._apply_to_candidate(candidate, evidence)

        assert "not checked" in candidate.explanation["frb_coincidence"]


class TestTopKCounterpartRecall:
    def test_true_counterpart_within_k_counts_as_a_hit(self):
        target = _burst(tns_name="TRUE", ra_deg=180.0, dec_deg=10.0)
        distractor_far = _burst(tns_name="FAR", ra_deg=200.0, dec_deg=-30.0)
        query = {"ra_deg": 180.0, "dec_deg": 10.0, "true_tns_name": "TRUE"}

        result = frb.top_k_counterpart_recall([query], [target, distractor_far], k=2)

        assert result["queries"] == 1
        assert result["hits"] == 1
        assert result["recall_at_k"] == pytest.approx(1.0)
        assert result["per_query"][0]["rank"] == 1

    def test_true_counterpart_ranked_outside_k_is_a_miss(self):
        true_burst = _burst(tns_name="TRUE", ra_deg=180.05, dec_deg=10.0)
        closer_bursts = [
            _burst(tns_name=f"CLOSER{i}", ra_deg=180.0, dec_deg=10.0) for i in range(3)
        ]
        query = {"ra_deg": 180.0, "dec_deg": 10.0, "true_tns_name": "TRUE"}

        result = frb.top_k_counterpart_recall(
            [query], closer_bursts + [true_burst], k=1)

        assert result["hits"] == 0
        assert result["recall_at_k"] == pytest.approx(0.0)
        assert result["per_query"][0]["rank"] is None

    def test_queries_without_a_known_truth_are_skipped(self):
        result = frb.top_k_counterpart_recall(
            [{"ra_deg": 180.0, "dec_deg": 10.0}], [_burst()], k=1)
        assert result["queries"] == 0
        assert result["recall_at_k"] != result["recall_at_k"]  # NaN

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError):
            frb.top_k_counterpart_recall([], [], k=0)


class TestCredibleRegionContainment:
    def test_empirical_containment_tracks_the_nominal_level(self, tmp_path):
        # sigma_deg is chosen well above nside=256's ~0.23 deg pixel scale so
        # the Gaussian resolves into many pixels rather than collapsing into
        # one (a sigma much smaller than the pixel size would make every
        # trial land in the same single pixel and trivially fail).
        result = frb.credible_region_containment(
            nside=256, trials=400, levels=(0.5, 0.9), sigma_deg=1.0,
            seed=7, root=tmp_path)

        assert result["trials"] == 400
        containment = result["empirical_containment"]
        # A well-calibrated map should land within a generous tolerance of
        # the nominal level; this is a statistical check, not exact equality.
        assert abs(containment[0.5] - 0.5) < 0.15
        assert abs(containment[0.9] - 0.9) < 0.10
        assert containment[0.5] <= containment[0.9] + 1e-9

    def test_containment_cleans_up_its_synthetic_map(self, tmp_path):
        frb.credible_region_containment(nside=64, trials=20, root=tmp_path)
        assert not (tmp_path / "chimefrb" / "loc_containment_check.h5").exists()


class TestEnrichCandidateFrb:
    def test_skips_the_localization_lookup_when_temporally_incoincident(
            self, tmp_path, monkeypatch):
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

        monkeypatch.setattr(frb, "localization_membership", fake_membership)

        far_burst = _burst(mjd_400=2459000.0 - 2_400_000.5)

        evidence = frb.enrich_candidate_frb(candidate, [far_burst], window_days=1.0)

        assert calls["n"] == 0
        assert evidence["state"] == "no_match"
        assert evidence["checked_bursts"] == 1
        assert evidence["temporally_coincident"] == 0

    def test_a_match_prefers_the_healpix_result_when_available(self, tmp_path, monkeypatch):
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
        near_burst = _burst(mjd_400=2458005.0 - 2_400_000.5, localization_id="FRB1")

        monkeypatch.setattr(frb, "localization_membership",
                            lambda *a, **k: {"confidence_level": 0.2, "in_90pct_region": True})

        evidence = frb.enrich_candidate_frb(candidate, [near_burst], window_days=1.0)

        assert evidence["state"] == "match"
        assert evidence["coincident"][0]["position_source"] == "healpix"
        assert evidence["coincident"][0]["confidence_level"] == pytest.approx(0.2)
