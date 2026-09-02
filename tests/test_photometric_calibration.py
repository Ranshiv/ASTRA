"""Cross-survey zero-point/color-term calibration (photometric_calibration.py).
No network -- matched pairs are built from synthetic SourceRef objects with a
known, injected zero-point/color-term relationship, recoverable by
construction (same discipline `artifact.calibrate_from_injection` uses).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astra import photometric_calibration as calib
from astra.surveys import gaia as gaia_survey
from astra.surveys.base import SourceRef


def _gaia_source(object_id: str, ra: float, dec: float, *, g: float, bp: float, rp: float,
                 flux_error_fraction: float = 0.01) -> SourceRef:
    # Gaia flux ~ 10**(-0.4*mag) up to a constant; only the ratio matters for
    # magnitude_error_from_flux, so an arbitrary flux scale is fine.
    def flux_pair(mag: float) -> tuple[float, float]:
        flux = 10 ** (-0.4 * mag)
        return flux, flux * flux_error_fraction

    g_flux, g_flux_err = flux_pair(g)
    bp_flux, bp_flux_err = flux_pair(bp)
    rp_flux, rp_flux_err = flux_pair(rp)
    return SourceRef(survey="Gaia", object_id=object_id, ra_deg=ra, dec_deg=dec, extra={
        "phot_g_mean_mag": g, "phot_bp_mean_mag": bp, "phot_rp_mean_mag": rp,
        "phot_g_mean_flux": g_flux, "phot_g_mean_flux_error": g_flux_err,
        "phot_bp_mean_flux": bp_flux, "phot_bp_mean_flux_error": bp_flux_err,
        "phot_rp_mean_flux": rp_flux, "phot_rp_mean_flux_error": rp_flux_err,
    })


def _panstarrs_source(object_id: str, ra: float, dec: float, *, g: float,
                      g_error: float = 0.02) -> SourceRef:
    return SourceRef(survey="Pan-STARRS", object_id=object_id, ra_deg=ra, dec_deg=dec,
                     extra={"g_mean": g, "g_mean_error": g_error})


class TestSourceMagnitude:
    def test_gaia_error_is_derived_from_flux(self):
        source = _gaia_source("1", 10.0, 20.0, g=15.0, bp=15.5, rp=14.5)
        mag, error = calib.source_magnitude(source, "G")
        assert mag == pytest.approx(15.0)
        assert error is not None and error > 0

    def test_panstarrs_error_is_read_directly(self):
        source = _panstarrs_source("1", 10.0, 20.0, g=15.2, g_error=0.03)
        mag, error = calib.source_magnitude(source, "g")
        assert mag == pytest.approx(15.2)
        assert error == pytest.approx(0.03)

    def test_unknown_survey_returns_none(self):
        source = SourceRef(survey="DES", object_id="1", ra_deg=10.0, dec_deg=20.0, extra={})
        mag, error = calib.source_magnitude(source, "g")
        assert mag is None and error is None

    def test_sdss_error_is_read_directly(self):
        # SDSS's ugriz photometry is joined onto SpecObjAll rows by
        # `surveys/sdss.py::cone_search` -- see that module's docstring for
        # the real gap this closes (SDSS pairs were previously unavailable
        # to this calibration at all).
        source = SourceRef(survey="SDSS", object_id="1", ra_deg=10.0, dec_deg=20.0,
                           extra={"mag_g": 18.2, "mag_g_error": 0.01})
        mag, error = calib.source_magnitude(source, "g")
        assert mag == pytest.approx(18.2)
        assert error == pytest.approx(0.01)

    def test_sdss_missing_photometry_returns_none(self):
        # bestObjID=0 (no photometric counterpart) leaves `mag_g` absent.
        source = SourceRef(survey="SDSS", object_id="1", ra_deg=10.0, dec_deg=20.0, extra={})
        mag, error = calib.source_magnitude(source, "g")
        assert mag is None and error is None


class TestBuildMatchedPairs:
    def test_matches_gaia_and_panstarrs_by_position(self):
        by_survey = {
            "Gaia": [_gaia_source("g1", 10.0, 20.0, g=15.0, bp=15.5, rp=14.5)],
            "Pan-STARRS": [_panstarrs_source("p1", 10.0001, 20.0001, g=15.3)],
        }
        rows = calib.build_matched_pairs(
            "Gaia", "Pan-STARRS", "G", "g", by_survey=by_survey,
            color_survey="Gaia", color_bands=("BP", "RP"))
        assert len(rows) == 1
        assert rows[0]["anchor_mag"] == pytest.approx(15.0)
        assert rows[0]["comparison_mag"] == pytest.approx(15.3)
        assert rows[0]["color"] == pytest.approx(1.0)

    def test_no_counterpart_within_radius_yields_no_pair(self):
        by_survey = {
            "Gaia": [_gaia_source("g1", 10.0, 20.0, g=15.0, bp=15.5, rp=14.5)],
            "Pan-STARRS": [_panstarrs_source("p1", 50.0, -10.0, g=15.3)],
        }
        rows = calib.build_matched_pairs("Gaia", "Pan-STARRS", "G", "g", by_survey=by_survey)
        assert rows == []


class TestDeriveStratumByObjectId:
    def test_reads_the_named_field_per_source(self):
        sources = [
            _panstarrs_source("p1", 10.0, 20.0, g=15.0),
            _panstarrs_source("p2", 11.0, 21.0, g=16.0),
        ]
        sources[0].extra["run2d"] = "26"
        sources[1].extra["run2d"] = "v5_13_2"
        stratum = calib.derive_stratum_by_object_id(sources, "run2d")
        assert stratum == {"p1": "26", "p2": "v5_13_2"}

    def test_missing_field_maps_to_none_not_fabricated(self):
        sources = [_panstarrs_source("p1", 10.0, 20.0, g=15.0)]
        stratum = calib.derive_stratum_by_object_id(sources, "run2d")
        assert stratum == {"p1": None}


class TestFitZeroPoint:
    def _synthetic_pairs(self, n: int, true_zero_point: float, true_color_term: float,
                         seed: int = 0) -> list[dict]:
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            anchor_mag = rng.uniform(12.0, 18.0)
            color = rng.uniform(-0.5, 2.0)
            delta = true_zero_point + true_color_term * color
            comparison_mag = anchor_mag + delta
            rows.append({
                "anchor_object_id": f"a{i}", "comparison_object_id": f"c{i}",
                "anchor_mag": anchor_mag, "anchor_mag_error": 0.01,
                "comparison_mag": comparison_mag, "comparison_mag_error": 0.01,
                "color": color, "stratum": None,
            })
        return rows

    def test_recovers_known_zero_point_and_color_term(self):
        rows = self._synthetic_pairs(200, true_zero_point=0.15, true_color_term=-0.08)
        result = calib.fit_zero_point(rows, min_pairs=10)
        entry = result["default"]
        assert entry["ready"]
        assert entry["zero_point"] == pytest.approx(0.15, abs=0.01)
        assert entry["color_term"] == pytest.approx(-0.08, abs=0.01)
        assert entry["residual_rms"] < 0.05

    def test_separates_strata(self):
        rows_a = self._synthetic_pairs(50, true_zero_point=0.1, true_color_term=0.0, seed=1)
        rows_b = self._synthetic_pairs(50, true_zero_point=0.4, true_color_term=0.0, seed=2)
        for row in rows_a:
            row["stratum"] = "night_1"
        for row in rows_b:
            row["stratum"] = "night_2"
        result = calib.fit_zero_point(rows_a + rows_b, min_pairs=10)
        assert result["night_1"]["zero_point"] == pytest.approx(0.1, abs=0.02)
        assert result["night_2"]["zero_point"] == pytest.approx(0.4, abs=0.02)

    def test_below_min_pairs_is_not_ready(self):
        rows = self._synthetic_pairs(3, true_zero_point=0.0, true_color_term=0.0)
        result = calib.fit_zero_point(rows, min_pairs=10)
        assert result["default"]["ready"] is False
        assert result["default"]["n_pairs"] == 3


class TestCalibrateEndToEnd:
    def test_full_pipeline_recovers_injected_offset(self):
        rng = np.random.default_rng(7)
        true_zero_point = 0.2
        gaia_sources = []
        panstarrs_sources = []
        for i in range(30):
            ra = i * 0.01
            dec = 20.0
            g = rng.uniform(13.0, 17.0)
            bp, rp = g + 0.3, g - 0.3
            gaia_sources.append(_gaia_source(f"g{i}", ra, dec, g=g, bp=bp, rp=rp))
            panstarrs_sources.append(_panstarrs_source(f"p{i}", ra, dec, g=g + true_zero_point))
        by_survey = {"Gaia": gaia_sources, "Pan-STARRS": panstarrs_sources}

        result = calib.calibrate("Gaia", "Pan-STARRS", "G", "g", by_survey=by_survey,
                                 min_pairs=10)

        assert result["ready"]
        assert result["n_pairs"] == 30
        assert result["strata"]["default"]["zero_point"] == pytest.approx(
            true_zero_point, abs=0.02)

    def test_no_matches_reports_not_ready(self):
        by_survey = {
            "Gaia": [_gaia_source("g1", 10.0, 20.0, g=15.0, bp=15.5, rp=14.5)],
            "Pan-STARRS": [_panstarrs_source("p1", 50.0, -10.0, g=15.3)],
        }
        result = calib.calibrate("Gaia", "Pan-STARRS", "G", "g", by_survey=by_survey)
        assert result["ready"] is False
        assert result["reason"] == "no matched photometric pairs"

    def test_stratum_field_derives_strata_from_comparison_survey_metadata(self):
        rng = np.random.default_rng(11)
        gaia_sources, panstarrs_sources = [], []
        for i in range(30):
            ra, dec = i * 0.01, 20.0
            g = rng.uniform(13.0, 17.0)
            bp, rp = g + 0.3, g - 0.3
            # Odd/even split into two "camera" strata with different offsets,
            # via a metadata field already on the Pan-STARRS SourceRef.
            camera = "cam_a" if i % 2 == 0 else "cam_b"
            offset = 0.1 if camera == "cam_a" else 0.5
            gaia_sources.append(_gaia_source(f"g{i}", ra, dec, g=g, bp=bp, rp=rp))
            p_source = _panstarrs_source(f"p{i}", ra, dec, g=g + offset)
            p_source.extra["camera"] = camera
            panstarrs_sources.append(p_source)
        by_survey = {"Gaia": gaia_sources, "Pan-STARRS": panstarrs_sources}

        result = calib.calibrate("Gaia", "Pan-STARRS", "G", "g", by_survey=by_survey,
                                 stratum_field="camera", min_pairs=10)

        assert result["ready"]
        assert result["strata"]["cam_a"]["zero_point"] == pytest.approx(0.1, abs=0.05)
        assert result["strata"]["cam_b"]["zero_point"] == pytest.approx(0.5, abs=0.05)

    def test_passing_both_stratum_kwargs_raises(self):
        by_survey = {
            "Gaia": [_gaia_source("g1", 10.0, 20.0, g=15.0, bp=15.5, rp=14.5)],
            "Pan-STARRS": [_panstarrs_source("p1", 10.0, 20.0, g=15.3)],
        }
        with pytest.raises(ValueError):
            calib.calibrate("Gaia", "Pan-STARRS", "G", "g", by_survey=by_survey,
                            stratum_by_object_id={"p1": "x"}, stratum_field="camera")


class TestSaveRoundTrip:
    def test_save_writes_json(self, tmp_path):
        payload = {"schema_version": calib.SCHEMA_VERSION, "ready": True}
        path = calib.save(payload, root=tmp_path, name="test-run")
        assert path.exists()
        assert path.parent.name == "photometric_calibration"


class TestGaiaPhotometricErrors:
    def test_missing_flux_returns_none(self):
        assert gaia_survey.magnitude_error_from_flux(None, 1.0) is None
        assert gaia_survey.magnitude_error_from_flux(1.0, None) is None

    def test_nonpositive_flux_returns_none(self):
        assert gaia_survey.magnitude_error_from_flux(0.0, 1.0) is None
        assert gaia_survey.magnitude_error_from_flux(-1.0, 1.0) is None

    def test_typical_flux_ratio_gives_a_reasonable_mag_error(self):
        error = gaia_survey.magnitude_error_from_flux(1000.0, 10.0)
        assert error is not None
        assert 0.005 < error < 0.05
