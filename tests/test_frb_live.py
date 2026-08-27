"""Live, on-demand contract check for the CHIME/FRB CADC/CANFAR service.

Not run by pytest's default collection (see the `live` marker registered in
engine/pyproject.toml -- `addopts = "-m 'not live'"` excludes it). Run
explicitly with:

    pytest tests/test_frb_live.py -m live

frb.py's module docstring documents that CADC's machine-readable VOSpace
backend was returning 503 across every check made while that module was
built, even though the plain browsable website worked throughout -- a real
infrastructure outage, not a wrong URL. `fetch_burst_catalog`/
`localization_membership` were written against the documented VOSpace
convention but have never been exercised against a live, healthy service.
This file turns that "re-verify once the outage clears" docstring note into
a concrete, runnable check instead of a TODO comment: it asserts the
documented CSV/HDF5 field names this module's parsers assume
(`_parse_catalog_csv`'s `tns_name`/`ra`/`ra_err`/`dec`/`dec_err`/`mjd_400`/
`localization_id`/`excluded_flag`, and `localization_membership`'s HDF5
`ipix`/`CL` datasets) actually match what CADC returns today, converting a
silent "zero bursts parsed" outcome into a loud test failure instead.
"""

from __future__ import annotations

import pytest

from astra import frb

pytestmark = pytest.mark.live


class TestLiveCatalogFetch:
    def test_fetch_burst_catalog_returns_real_parsed_bursts(self, isolated_root):
        bursts = frb.fetch_burst_catalog(refresh=True)

        assert len(bursts) > 0, (
            "fetch_burst_catalog returned zero bursts against the live CADC "
            "service -- either the service is still down, or the documented "
            "CSV column names in frb._parse_catalog_csv have drifted from "
            "the real chimefrbcat2.csv schema; check both before assuming "
            "this is a flaky network failure")
        sample = bursts[0]
        assert isinstance(sample.tns_name, str) and sample.tns_name
        assert -90.0 <= sample.dec_deg <= 90.0
        assert 0.0 <= sample.ra_deg < 360.0
        assert sample.ra_err_deg > 0 and sample.dec_err_deg > 0
        assert sample.mjd_400 > 0

    def test_a_baseband_localized_burst_has_a_readable_confidence_map(self, isolated_root):
        bursts = frb.fetch_burst_catalog(refresh=True)
        localized = next((burst for burst in bursts if burst.localization_id), None)
        if localized is None:
            pytest.skip("no baseband-localized burst in the current catalogue snapshot")

        membership = frb.localization_membership(
            localized, localized.ra_deg, localized.dec_deg)

        assert membership is not None, (
            "a burst with a localization_id produced no readable HEALPix "
            "confidence map -- check that the VOSpace path or the HDF5 "
            "ipix/CL dataset names in frb.localization_membership still "
            "match the live service")
        assert 0.0 <= membership["confidence_level"] <= 1.0
