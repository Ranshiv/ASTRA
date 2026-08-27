"""surveys/ztf.py: cone search, light-curve parsing, and the additive
`fetch_light_curves_with_quality` path that recovers real catflags."""

from __future__ import annotations

import numpy as np
import pytest

from astra import netclient
from astra.surveys.base import SourceRef
from astra.surveys.ztf import ZTFConnector, parse_csv


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


VALID_CSV = (
    "oid,hjd,mag,magerr,filtercode,catflags\n"
    "1,2458000.5,18.1,0.02,zg,0\n"
    "1,2458001.5,18.2,0.02,zg,0\n"
    "1,2458002.5,25.0,0.50,zg,32768\n"
    "1,2458003.5,18.3,0.02,zr,0\n"
)


@pytest.fixture
def source() -> SourceRef:
    return SourceRef(survey="ZTF", object_id="1", ra_deg=10.0, dec_deg=20.0)


class TestParseCsv:
    def test_empty_payload_yields_no_rows(self):
        assert parse_csv("") == []

    def test_parses_rows(self):
        rows = parse_csv(VALID_CSV)
        assert len(rows) == 4
        assert rows[0]["catflags"] == "0"


class TestFetchLightCurves:
    def test_default_request_masks_flagged_epochs_at_the_archive(self, monkeypatch, source):
        captured = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            return _FakeResponse(VALID_CSV)

        monkeypatch.setattr(netclient, "get", fake_get)
        ZTFConnector().fetch_light_curves(source)

        assert captured["params"]["BAD_CATFLAGS_MASK"] == 32768

    def test_returns_one_curve_per_band(self, monkeypatch, source):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_CSV))
        curves = ZTFConnector().fetch_light_curves(source)
        assert {c.band for c in curves} == {"g", "r"}


class TestFetchLightCurvesWithQuality:
    def test_requests_with_the_filter_disabled(self, monkeypatch, source):
        captured = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            return _FakeResponse(VALID_CSV)

        monkeypatch.setattr(netclient, "get", fake_get)
        ZTFConnector().fetch_light_curves_with_quality(source)

        assert captured["params"]["BAD_CATFLAGS_MASK"] == 0

    def test_does_not_change_fetch_light_curves_behaviour(self, monkeypatch, source):
        """The additive method must not affect the default path at all."""
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_CSV))
        connector = ZTFConnector()

        default_curves = connector.fetch_light_curves(source)
        # Calling the new method afterward must not have mutated any shared
        # state that changes a subsequent call to the old one.
        connector.fetch_light_curves_with_quality(source)
        default_curves_again = connector.fetch_light_curves(source)

        assert len(default_curves) == len(default_curves_again)
        assert default_curves[0].band == default_curves_again[0].band

    def test_recovers_the_real_flagged_epoch_default_fetch_would_strip(self, monkeypatch, source):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_CSV))
        pairs = ZTFConnector().fetch_light_curves_with_quality(source)

        g_curve, g_catflags = next(p for p in pairs if p[0].band == "g")
        assert len(g_curve.value) == 3  # all 3 g-band rows, including the flagged one
        assert 32768 in g_catflags
        assert int(np.count_nonzero(g_catflags)) == 1

    def test_catflags_stays_aligned_with_the_curve_it_belongs_to(self, monkeypatch, source):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_CSV))
        pairs = ZTFConnector().fetch_light_curves_with_quality(source)

        for curve, catflags in pairs:
            assert len(catflags) == len(curve.value) == len(curve.time)

    def test_a_malformed_row_is_skipped_not_fatal(self, monkeypatch, source):
        bad_csv = VALID_CSV + "1,not-a-number,18.0,0.02,zg,0\n"
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(bad_csv))
        pairs = ZTFConnector().fetch_light_curves_with_quality(source)
        g_curve, _ = next(p for p in pairs if p[0].band == "g")
        assert len(g_curve.value) == 3  # malformed row dropped, not crashed
