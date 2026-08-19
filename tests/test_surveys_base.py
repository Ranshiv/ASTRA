"""Normalisation contract shared by every connector."""

from __future__ import annotations

import numpy as np
import pytest

from astra.surveys import base
from astra.surveys.base import ConeQuery, LightCurve, SourceRef


class TestConeQuery:
    def test_rejects_impossible_declination(self):
        with pytest.raises(ValueError, match="dec_deg"):
            ConeQuery(ra_deg=10.0, dec_deg=91.0, radius_arcsec=5.0)

    def test_rejects_non_positive_radius(self):
        with pytest.raises(ValueError, match="radius_arcsec"):
            ConeQuery(ra_deg=10.0, dec_deg=10.0, radius_arcsec=0.0)

    def test_radius_converts_to_degrees(self, cone):
        assert cone.radius_deg == pytest.approx(10.0 / 3600.0)

    def test_key_is_stable_for_equal_queries(self):
        a = ConeQuery(180.122, 22.411, 10.0)
        b = ConeQuery(180.122, 22.411, 10.0)
        assert a.key() == b.key()


class TestSourceRef:
    def test_storage_key_is_deterministic(self, source):
        assert source.storage_key("dr24") == source.storage_key("dr24")

    def test_storage_key_separates_releases(self, source):
        assert source.storage_key("dr23") != source.storage_key("dr24")

    def test_storage_key_separates_objects(self):
        a = SourceRef(survey="ZTF", object_id="1", ra_deg=0.0, dec_deg=0.0)
        b = SourceRef(survey="ZTF", object_id="2", ra_deg=0.0, dec_deg=0.0)
        assert a.storage_key("dr24") != b.storage_key("dr24")

    def test_connector_description_declares_capabilities(self):
        from astra.surveys import gaia, tess, ztf

        assert "image" in ztf.ZTFConnector().describe()["capabilities"]
        assert "target_pixel_file" in tess.TESSConnector().describe()["capabilities"]
        assert "light_curve" not in gaia.GaiaConnector().describe()["capabilities"]
        assert tess.TESSConnector().describe()["resolution_arcsec"] == 21.0


class TestLightCurve:
    def test_time_stays_float64_and_values_become_float32(self, curve):
        assert curve.time.dtype == np.float64
        assert curve.value.dtype == np.float32
        assert curve.value_err.dtype == np.float32

    def test_mismatched_column_lengths_are_rejected(self, source):
        with pytest.raises(ValueError, match="same length"):
            LightCurve(source=source, release="dr24", band="g",
                       value_kind="mag", time=[1.0, 2.0],
                       value=[1.0], value_err=[1.0])

    def test_dropna_removes_non_finite_points(self, source):
        lc = LightCurve(source=source, release="dr24", band="g",
                        value_kind="mag",
                        time=[1.0, 2.0, 3.0],
                        value=[10.0, np.nan, 12.0],
                        value_err=[0.1, 0.1, 0.1])
        assert len(lc.dropna()) == 2

    def test_dropna_preserves_time_system(self, curve):
        assert curve.dropna().time_system == "HJD_UTC"

    def test_sorting_orders_by_time(self, source):
        lc = LightCurve(source=source, release="dr24", band="g",
                        value_kind="mag", time=[3.0, 1.0, 2.0],
                        value=[1.0, 2.0, 3.0], value_err=[0.1, 0.1, 0.1])
        assert list(lc.sorted_by_time().time) == [1.0, 2.0, 3.0]

    def test_time_span_of_short_curve_is_zero(self, source):
        lc = LightCurve(source=source, release="dr24", band="g",
                        value_kind="mag", time=[1.0], value=[1.0],
                        value_err=[0.1])
        assert lc.time_span_days() == 0.0


class TestBandNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("1", "g"), ("2", "r"), ("3", "i"), ("zg", "g"), ("zr", "r"),
    ])
    def test_ztf_filter_codes(self, raw, expected):
        assert base.normalise_band("ztf", raw) == expected

    def test_unknown_code_passes_through(self):
        assert base.normalise_band("ztf", "q") == "q"

    def test_empty_becomes_unknown(self):
        assert base.normalise_band("ztf", "") == "unknown"


class TestToArrays:
    def test_empty_input_yields_correctly_typed_empty_columns(self):
        time, value, err = base.to_arrays([])
        assert len(time) == 0
        assert time.dtype == np.float64
        assert value.dtype == np.float32

    def test_preserves_bjd_precision(self):
        # float32 would round this to roughly the nearest 0.25 day.
        time, _, _ = base.to_arrays([(2458000.123456, 18.0, 0.1)])
        assert time[0] == pytest.approx(2458000.123456, abs=1e-9)
