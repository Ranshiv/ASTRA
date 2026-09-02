"""Connector parsing and registry behaviour, without touching the network."""

from __future__ import annotations

import numpy as np
import pytest

from astra import surveys
from astra.surveys import gaia, ztf
from astra.surveys import panstarrs, sdss
from astra.surveys.base import ConeQuery, LightCurve, SourceRef, SurveyConnector

# A trimmed real IRSA response: two bands, one unparseable epoch.
ZTF_CSV = """oid,hjd,mjd,mag,magerr,catflags,filtercode,ra,dec
123,2458000.5,58000.0,18.21,0.03,0,zg,180.1,22.4
123,2458001.5,58001.0,18.25,0.04,0,zg,180.1,22.4
123,2458002.5,58002.0,17.90,0.02,0,zr,180.1,22.4
123,,58003.0,,,0,zg,180.1,22.4
"""


class TestZTFParsing:
    def test_parse_csv_reads_all_rows(self):
        assert len(ztf.parse_csv(ZTF_CSV)) == 4

    def test_parse_csv_tolerates_empty_response(self):
        assert ztf.parse_csv("") == []
        assert ztf.parse_csv("   \n") == []

    def test_rows_split_into_one_curve_per_band(self, source):
        curves = ztf.ZTFConnector()._rows_to_curves(ztf.parse_csv(ZTF_CSV), source)
        assert sorted(c.band for c in curves) == ["g", "r"]

    def test_malformed_epoch_is_skipped_not_fatal(self, source):
        curves = ztf.ZTFConnector()._rows_to_curves(ztf.parse_csv(ZTF_CSV), source)
        g_band = next(c for c in curves if c.band == "g")
        assert len(g_band) == 2  # the blank row is dropped

    def test_ztf_reports_its_time_system(self, source):
        curves = ztf.ZTFConnector()._rows_to_curves(ztf.parse_csv(ZTF_CSV), source)
        assert all(c.time_system == "HJD_UTC" for c in curves)

    def test_ztf_values_are_magnitudes(self, source):
        curves = ztf.ZTFConnector()._rows_to_curves(ztf.parse_csv(ZTF_CSV), source)
        assert all(c.value_kind == "mag" for c in curves)

    def test_catalog_name_tracks_release(self):
        assert ztf.ZTFConnector(release="dr23").catalog == "ztf_objects_dr23"


class TestGaiaDerivedProperties:
    def test_distance_from_parallax(self):
        result = gaia.derived_properties({"parallax": 10.0})
        assert result["distance_pc"] == pytest.approx(100.0)

    def test_negative_parallax_yields_no_distance(self):
        """Negative parallaxes are common in Gaia and are not distances."""
        assert gaia.derived_properties({"parallax": -0.5})["distance_pc"] is None

    def test_missing_parallax_is_handled(self):
        assert gaia.derived_properties({})["distance_pc"] is None

    def test_colour_index(self):
        result = gaia.derived_properties(
            {"phot_bp_mean_mag": 15.0, "phot_rp_mean_mag": 14.0})
        assert result["bp_rp"] == pytest.approx(1.0)

    def test_parallax_signal_to_noise(self):
        result = gaia.derived_properties({"parallax": 2.0, "parallax_error": 0.5})
        assert result["parallax_snr"] == pytest.approx(4.0)

    def test_absolute_magnitude_at_ten_parsecs_equals_apparent(self):
        result = gaia.derived_properties(
            {"parallax": 100.0, "phot_g_mean_mag": 12.0})
        assert result["distance_pc"] == pytest.approx(10.0)
        assert result["abs_g_mag"] == pytest.approx(12.0)

    def test_gaia_has_no_light_curves(self, source):
        assert gaia.GaiaConnector().fetch_light_curves(source) == []


class TestRegistry:
    def test_the_three_initial_surveys_are_registered(self):
        assert {"gaia", "tess", "ztf"}.issubset(set(surveys.available()))

    def test_sdss_and_panstarrs_are_enabled_by_default(self):
        assert surveys.available() == ["gaia", "panstarrs", "sdss", "tess", "ztf"]
        assert "chandra" not in surveys.available()
        assert {"sdss", "panstarrs", "chandra"}.issubset(set(surveys.available(True)))

    def test_lookup_is_case_insensitive(self):
        assert surveys.get("ZTF").name == "ZTF"

    def test_unknown_survey_names_the_alternatives(self):
        with pytest.raises(KeyError, match="available"):
            surveys.get("rubin")

    def test_describe_all_avoids_the_network(self):
        described = surveys.describe_all()
        assert {"ZTF", "Gaia", "TESS", "SDSS", "Pan-STARRS"}.issubset(
            {d["name"] for d in described})

    def test_a_new_survey_arrives_as_a_connector(self):
        """Plan section 7: adding a survey must not require editing the pipeline."""

        class RubinConnector(SurveyConnector):
            name = "Rubin"
            release = "dp1"

            def cone_search(self, query, limit=100):
                return []

            def fetch_light_curves(self, source):
                return []

        surveys.register("rubin", RubinConnector)
        try:
            assert surveys.get("rubin").name == "Rubin"
        finally:
            surveys._REGISTRY.pop("rubin")

    def test_registering_a_non_connector_is_rejected(self):
        with pytest.raises(TypeError):
            surveys.register("bad", dict)  # type: ignore[arg-type]

    def test_sdss_csv_fixture_is_bounded(self):
        rows = sdss.parse_csv("objID,ra,dec\n1,10,20\n2,11,21\n", limit=1)
        assert rows == [{"objID": "1", "ra": "10", "dec": "20"}]

    def test_panstarrs_json_fixture_is_bounded(self):
        # The real `mean` endpoint returns {"info": [...columns...],
        # "data": [...positional rows...]}, not a bare list of dicts --
        # see test_surveys_panstarrs.py for the full finding. This fixture
        # predated that fix and encoded the same wrong shape.
        payload = {"info": [{"name": "objID"}],
                  "data": [[1], [2]]}
        rows = panstarrs.parse_rows(payload, limit=1)
        assert rows == [{"objID": 1}]

    def test_metadata_connectors_are_not_falsified_as_light_curves(self, source):
        assert sdss.SDSSConnector().fetch_light_curves(source) == []
        assert panstarrs.PanSTARRSConnector().fetch_light_curves(source) == []


class TestTESSConversion:
    def test_btjd_is_converted_to_bjd(self):
        from astra.surveys import tess

        class FakeColumn:
            def __init__(self, values):
                self.value = np.asarray(values)

        class FakeCurve:
            time = FakeColumn([1000.0, 1001.0])
            flux = FakeColumn([1.0, 1.01])
            flux_err = FakeColumn([0.01, 0.01])
            meta = {"SECTOR": 14}

        source = SourceRef(survey="TESS", object_id="TIC 1",
                           ra_deg=0.0, dec_deg=0.0)
        curve = tess.TESSConnector()._convert(FakeCurve(), source)

        assert curve.time[0] == pytest.approx(1000.0 + tess.BTJD_OFFSET)
        assert curve.time_system == "BJD_TDB"
        assert curve.value_kind == "flux"
        assert curve.release.endswith("s14")


class TestTESSTargetSelection:
    """Regression tests for the two defects that starved TESS acquisition.

    `lk.search_lightcurve` returns one row per target x sector. The connector
    used to slice `search.table[:limit]` and then dedupe, so a single
    well-observed star consumed the whole budget; and every source inherited
    the cone centre as its position, which makes cross-matching meaningless.
    """

    class _FakeTable:
        def __init__(self, rows, colnames):
            self._rows = rows
            self.colnames = colnames

        def __iter__(self):
            return iter(self._rows)

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, item):
            if isinstance(item, slice):
                return TestTESSTargetSelection._FakeTable(
                    self._rows[item], self.colnames)
            return self._rows[item]

    class _FakeSearch:
        def __init__(self, table):
            self.table = table

        def __len__(self):
            return len(self.table)

    def _search(self, targets, sectors_each, with_coords=True):
        colnames = ["target_name", "sequence_number"]
        if with_coords:
            colnames += ["s_ra", "s_dec"]

        rows = []
        for index in range(targets):
            for sector in range(sectors_each):
                row = {"target_name": str(100 + index),
                       "sequence_number": sector + 1}
                if with_coords:
                    # Each target sits at a distinct, resolvable position.
                    row["s_ra"] = 10.0 + index * 0.001
                    row["s_dec"] = 20.0 + index * 0.001
                rows.append(row)
        return self._FakeSearch(self._FakeTable(rows, colnames))

    def _patch(self, monkeypatch, search):
        import sys
        import types

        module = types.ModuleType("lightkurve")
        module.search_lightcurve = lambda *a, **k: search
        monkeypatch.setitem(sys.modules, "lightkurve", module)

    def test_limit_counts_targets_not_sector_rows(self, monkeypatch, cone):
        """5 targets x 20 sectors with limit=5 must yield 5 sources, not 1."""
        from astra.surveys import tess

        self._patch(monkeypatch, self._search(targets=5, sectors_each=20))
        sources = tess.TESSConnector().cone_search(cone, limit=5)

        assert len(sources) == 5
        assert len({s.object_id for s in sources}) == 5

    def test_limit_is_still_respected(self, monkeypatch, cone):
        from astra.surveys import tess

        self._patch(monkeypatch, self._search(targets=12, sectors_each=3))
        assert len(tess.TESSConnector().cone_search(cone, limit=4)) == 4

    def test_per_target_coordinates_are_used(self, monkeypatch, cone):
        """Sources must not all inherit the cone centre."""
        from astra.surveys import tess

        self._patch(monkeypatch, self._search(targets=4, sectors_each=2))
        sources = tess.TESSConnector().cone_search(cone, limit=4)

        positions = {(s.ra_deg, s.dec_deg) for s in sources}
        assert len(positions) == 4
        assert all(s.ra_deg != cone.ra_deg for s in sources)

    def test_cone_centre_is_the_fallback_when_columns_absent(
            self, monkeypatch, cone):
        from astra.surveys import tess

        self._patch(monkeypatch,
                    self._search(targets=3, sectors_each=2, with_coords=False))
        sources = tess.TESSConnector().cone_search(cone, limit=3)

        assert all(s.ra_deg == cone.ra_deg for s in sources)
        assert all(s.dec_deg == cone.dec_deg for s in sources)

    def test_sectors_are_recorded(self, monkeypatch, cone):
        from astra.surveys import tess

        self._patch(monkeypatch, self._search(targets=2, sectors_each=3))
        sources = tess.TESSConnector().cone_search(cone, limit=2)

        assert all(sorted(s.extra["sectors"]) == [1, 2, 3] for s in sources)

    def test_empty_search_returns_nothing(self, monkeypatch, cone):
        from astra.surveys import tess

        self._patch(monkeypatch, self._FakeSearch(self._FakeTable([], [])))
        assert tess.TESSConnector().cone_search(cone, limit=5) == []


class TestTESSAuthorAndRelease:
    """B2: release must follow author, or QLP/SPOC data collides in the
    store (same (survey, release, object_id, band) storage key) and the
    resumable-fetch cursor treats a QLP fetch as "already done" because a
    SPOC fetch used the same release. See docs/DEFERRED.txt Phase 8."""

    def test_default_release_is_unchanged_for_spoc(self):
        from astra.surveys import tess

        connector = tess.TESSConnector()
        assert connector.author == "SPOC"
        assert connector.release == "spoc"

    def test_release_follows_a_non_default_author(self):
        from astra.surveys import tess

        assert tess.TESSConnector(author="QLP").release == "qlp"
        assert tess.TESSConnector(author="TGLC").release == "tglc"

    def test_explicit_release_still_overrides_the_author_default(self):
        from astra.surveys import tess

        connector = tess.TESSConnector(author="QLP", release="custom")
        assert connector.release == "custom"

    def test_flux_column_is_looked_up_per_author(self):
        from astra.surveys import tess

        assert tess.FLUX_COLUMNS["SPOC"] == "pdcsap_flux"
        assert tess.FLUX_COLUMNS["QLP"] == "sap_flux"
        assert tess.FLUX_COLUMNS["TGLC"] == "cal_psf_flux"
        # An author with no table entry must still resolve to something,
        # rather than KeyError deep inside fetch_light_curves.
        assert tess.FLUX_COLUMNS.get("SOMETHING_NEW", tess.DEFAULT_FLUX_COLUMN) \
            == tess.DEFAULT_FLUX_COLUMN

    def test_spoc_and_qlp_curves_for_the_same_target_do_not_collide(self):
        """The store's collision surface: two curves that would previously
        share a storage key (release was a fixed constant regardless of
        author) must now land at different paths."""
        from astra import store
        from astra.surveys import tess

        class FakeColumn:
            def __init__(self, values):
                self.value = np.asarray(values)

        class FakeCurve:
            time = FakeColumn([1000.0, 1001.0])
            flux = FakeColumn([1.0, 1.01])
            flux_err = FakeColumn([0.01, 0.01])
            meta = {"SECTOR": 14}

        source = SourceRef(survey="TESS", object_id="TIC 1",
                           ra_deg=0.0, dec_deg=0.0)
        spoc_curve = tess.TESSConnector(author="SPOC")._convert(FakeCurve(), source)
        qlp_curve = tess.TESSConnector(author="QLP")._convert(FakeCurve(), source)

        assert spoc_curve.release == "spoc-s14"
        assert qlp_curve.release == "qlp-s14"
        assert store.curve_path(spoc_curve) != store.curve_path(qlp_curve)

    def test_flux_column_failure_is_logged_not_silently_swallowed(self, monkeypatch, caplog):
        """A bad author/flux_column pairing (e.g. requesting pdcsap_flux
        against a QLP file, which has no such column) must be visible in
        the log, not indistinguishable from "no data at this position"."""
        import logging
        import sys
        import types

        from astra.surveys import tess

        class FailingSlice:
            def download(self, flux_column=None):
                raise KeyError(flux_column)

        class FakeSearch:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                return FailingSlice()

        module = types.ModuleType("lightkurve")
        module.search_lightcurve = lambda *a, **k: FakeSearch()
        monkeypatch.setitem(sys.modules, "lightkurve", module)

        source = SourceRef(survey="TESS", object_id="TIC 1",
                           ra_deg=0.0, dec_deg=0.0)
        with caplog.at_level(logging.WARNING, logger="astra.surveys.tess"):
            curves = tess.TESSConnector(author="QLP").fetch_light_curves(source)

        assert curves == []
        assert any("QLP" in record.message and "sap_flux" in record.message
                   for record in caplog.records)

    def test_tglc_without_archive_errors_gets_robust_finite_errors(self):
        from astra.surveys import tess

        class FakeColumn:
            def __init__(self, values):
                self.value = np.asarray(values)

        class FakeCurve:
            time = FakeColumn([1000.0, 1001.0, 1002.0, 1003.0])
            flux = FakeColumn([1.0, 1.01, 0.99, 1.0])
            meta = {"SECTOR": 14}

        source = SourceRef(survey="TESS", object_id="TIC 1",
                           ra_deg=0.0, dec_deg=0.0)
        curve = tess.TESSConnector(author="TGLC")._convert(FakeCurve(), source)

        assert curve is not None
        assert np.isfinite(curve.value_err).all()
        assert np.all(curve.value_err > 0)
        assert curve.source.extra["flux_error"] == "estimated_mad_differences"

    def test_gaia_epoch_fixture_validation_is_offline_and_bounded(self):
        result = gaia.GaiaEpochAdapter.validate_chunk([
            {"source_id": 1, "time": 2459000.1, "g_flux": 100.0, "g_flux_error": 2.0},
            {"source_id": 2, "time": 2459000.2, "g_flux": "bad", "g_flux_error": 2.0},
        ])
        assert result["release"] == "dr4-epoch"
        assert result["enabled"] is False
        assert result["accepted"] == 1
        assert result["rejected"] == 1
