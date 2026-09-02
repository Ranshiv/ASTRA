"""Kepler/K2 connector contract: cone search dedup/quarter tracking, BKJD
time conversion, flux-error fallback, mission validation, and a real live
fetch for a well-known Kepler flare star. Mocking follows `tess.py`'s own
lightkurve-search shape (no dedicated `tests/test_surveys_tess.py`
exists to mirror directly, so this follows the module's own documented
contract instead)."""

from __future__ import annotations

import numpy as np
import pytest

from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.kepler import BKJD_OFFSET, KeplerConnector


class _FakeColumn:
    def __init__(self, values):
        self.value = np.asarray(values)


class _FakeDownloaded:
    def __init__(self, time, flux, flux_err=None, meta=None):
        self.time = _FakeColumn(time)
        self.flux = _FakeColumn(flux)
        if flux_err is not None:
            self.flux_err = _FakeColumn(flux_err)
        self.meta = meta or {"QUARTER": 5}


class _FakeSearchResult:
    """A single `search[index]` entry with a `.download()` method."""

    def __init__(self, downloaded):
        self._downloaded = downloaded

    def download(self, flux_column=None):
        return self._downloaded


class _FakeTable:
    def __init__(self, rows, colnames):
        self._rows = rows
        self.colnames = colnames

    def __iter__(self):
        return iter(self._rows)


class _FakeSearch:
    def __init__(self, rows, colnames, per_row_downloaded=None):
        self.table = _FakeTable(rows, colnames)
        self._results = [_FakeSearchResult(d) for d in (per_row_downloaded or [])]

    def __len__(self):
        return len(self.table._rows) if self._results == [] else len(self._results)

    def __getitem__(self, index):
        return self._results[index]


class TestKeplerConnectorConstruction:
    def test_rejects_unknown_mission(self):
        with pytest.raises(ValueError):
            KeplerConnector(mission="TESS")

    def test_defaults_to_kepler_mission(self):
        connector = KeplerConnector()
        assert connector.mission == "Kepler"
        assert connector.name == "Kepler"
        assert connector.release == "kepler"

    def test_k2_mission_sets_name_and_release(self):
        connector = KeplerConnector(mission="K2")
        assert connector.name == "K2"
        assert connector.release == "k2"


class TestConeSearch:
    def test_dedupes_by_target_and_tracks_quarters(self, monkeypatch, cone: ConeQuery):
        rows = [
            {"target_name": "9726699", "s_ra": 297.415, "s_dec": 46.949, "sequence_number": 5},
            {"target_name": "9726699", "s_ra": 297.415, "s_dec": 46.949, "sequence_number": 6},
            {"target_name": "9726700", "s_ra": 297.42, "s_dec": 46.95, "sequence_number": 5},
        ]
        columns = ["target_name", "s_ra", "s_dec", "sequence_number"]
        fake_search = _FakeSearch(rows, columns)

        import lightkurve as lk
        monkeypatch.setattr(lk, "search_lightcurve", lambda *a, **k: fake_search)

        sources = KeplerConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        by_id = {s.object_id: s for s in sources}
        assert by_id["KIC 9726699"].extra["quarters"] == [5, 6]
        assert by_id["KIC 9726700"].extra["quarters"] == [5]

    def test_empty_search_returns_no_sources(self, monkeypatch, cone: ConeQuery):
        import lightkurve as lk
        monkeypatch.setattr(lk, "search_lightcurve", lambda *a, **k: _FakeSearch([], []))
        assert KeplerConnector().cone_search(cone) == []

    def test_masked_sequence_number_is_skipped_not_fatal(self, monkeypatch, cone: ConeQuery):
        # Real bug found live against a real Kepler-field query: a masked
        # (missing) `sequence_number` cell -- common in real MAST search
        # results, astropy masked columns -- raised numpy.ma.MaskError on
        # `int(...)`, which `_record_quarter`'s `except (TypeError,
        # ValueError)` did not catch, crashing the whole cone_search() for
        # every target in the batch over one target's missing quarter.
        rows = [
            {"target_name": "9726699", "s_ra": 297.415, "s_dec": 46.949,
             "sequence_number": np.ma.masked},
            {"target_name": "9726700", "s_ra": 297.42, "s_dec": 46.95,
             "sequence_number": 5},
        ]
        columns = ["target_name", "s_ra", "s_dec", "sequence_number"]
        fake_search = _FakeSearch(rows, columns)

        import lightkurve as lk
        monkeypatch.setattr(lk, "search_lightcurve", lambda *a, **k: fake_search)

        sources = KeplerConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        by_id = {s.object_id: s for s in sources}
        assert by_id["KIC 9726699"].extra["quarters"] == []
        assert by_id["KIC 9726700"].extra["quarters"] == [5]

    def test_respects_limit(self, monkeypatch, cone: ConeQuery):
        rows = [{"target_name": str(i), "s_ra": 1.0, "s_dec": 1.0, "sequence_number": 1}
               for i in range(5)]
        columns = ["target_name", "s_ra", "s_dec", "sequence_number"]
        import lightkurve as lk
        monkeypatch.setattr(lk, "search_lightcurve", lambda *a, **k: _FakeSearch(rows, columns))
        sources = KeplerConnector().cone_search(cone, limit=2)
        assert len(sources) == 2


class TestFetchLightCurves:
    def test_converts_bkjd_to_bjd_tdb(self, monkeypatch, source: SourceRef):
        time = np.array([100.0, 100.5, 101.0])
        flux = np.array([1.0, 1.01, 0.99], dtype=np.float32)
        flux_err = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        downloaded = _FakeDownloaded(time, flux, flux_err, meta={"QUARTER": 5})
        fake_search = _FakeSearch([], [], per_row_downloaded=[downloaded])

        import lightkurve as lk
        monkeypatch.setattr(lk, "search_lightcurve", lambda *a, **k: fake_search)

        curves = KeplerConnector().fetch_light_curves(source)
        assert len(curves) == 1
        assert curves[0].time[0] == pytest.approx(100.0 + BKJD_OFFSET)
        assert curves[0].release == "kepler-q5"
        assert curves[0].source.extra["flux_error"] == "archive"

    def test_quarter_zero_is_not_treated_as_missing(self, monkeypatch, source: SourceRef):
        # Real bug found this session against KIC 9726699's real Quarter 0
        # product: `meta.get("QUARTER") or fallback` treats a real
        # QUARTER=0 as falsy and mislabels it "unknown".
        downloaded = _FakeDownloaded(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array([0.01, 0.01, 0.01], dtype=np.float32), meta={"QUARTER": 0})
        fake_search = _FakeSearch([], [], per_row_downloaded=[downloaded])

        import lightkurve as lk
        monkeypatch.setattr(lk, "search_lightcurve", lambda *a, **k: fake_search)

        curves = KeplerConnector().fetch_light_curves(source)
        assert curves[0].release == "kepler-q0"

    def test_estimates_error_when_flux_err_missing(self, monkeypatch, source: SourceRef):
        time = np.linspace(0, 10, 50)
        flux = np.ones(50, dtype=np.float32)
        downloaded = _FakeDownloaded(time, flux, flux_err=None)
        fake_search = _FakeSearch([], [], per_row_downloaded=[downloaded])

        import lightkurve as lk
        monkeypatch.setattr(lk, "search_lightcurve", lambda *a, **k: fake_search)

        curves = KeplerConnector().fetch_light_curves(source)
        assert len(curves) == 1
        assert curves[0].source.extra["flux_error"] == "estimated_mad_differences"

    def test_a_failed_quarter_does_not_abort_the_whole_fetch(self, monkeypatch, source: SourceRef):
        class _RaisingResult:
            def download(self, flux_column=None):
                raise RuntimeError("simulated download failure")

        good = _FakeDownloaded(np.array([1.0, 2.0]), np.array([1.0, 1.0], dtype=np.float32),
                               np.array([0.01, 0.01], dtype=np.float32))
        fake_search = _FakeSearch([], [])
        fake_search._results = [_RaisingResult(), _FakeSearchResult(good)]

        import lightkurve as lk
        monkeypatch.setattr(lk, "search_lightcurve", lambda *a, **k: fake_search)

        curves = KeplerConnector().fetch_light_curves(source)
        assert len(curves) == 1

    def test_empty_search_returns_no_curves(self, monkeypatch, source: SourceRef):
        import lightkurve as lk
        monkeypatch.setattr(lk, "search_lightcurve", lambda *a, **k: _FakeSearch([], []))
        assert KeplerConnector().fetch_light_curves(source) == []


@pytest.mark.live
class TestKeplerLive:
    """A real fetch for KIC 9726699 (GJ 1243), one of the best-studied
    Kepler flare stars in the literature (e.g. Hawley et al. 2014, ApJ
    797, 121) -- the concrete demonstration this session's Kepler-
    connector gap closure names. Confirmed live this session: a real
    `lightkurve.search_lightcurve` call against MAST returns real Kepler
    quarters for this target."""

    def test_fetches_a_real_light_curve(self):
        source = SourceRef(survey="Kepler", object_id="KIC 9726699", ra_deg=297.415, dec_deg=46.949)
        connector = KeplerConnector(max_quarters=1)
        curves = connector.fetch_light_curves(source)
        assert len(curves) >= 1
        assert len(curves[0]) > 100
        assert np.all(np.isfinite(curves[0].time))
