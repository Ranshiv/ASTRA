"""ADQL construction and response parsing for the Exoplanet Archive TAP query."""

import pytest

from astra import exoplanet_archive as ea


class _Response:
    headers = {"Content-Type": "text/csv"}
    text = (
        "pl_name,hostname,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_trandur,"
        "pl_trandep,pl_rade,pl_radj,pl_tranmid\n"
        "Kepler-10 b,Kepler-10,0.837491,0.000001,-0.000001,1.79,152.0,1.47,0.131,2454964.57513\n"
    )


def _fake_get_factory(calls):
    def fake_get(url, params, timeout, provider, headers=None):
        calls.append((url, params, provider))
        return _Response()
    return fake_get


def test_query_confirmed_planets_requires_exactly_one_selector(tmp_path):
    with pytest.raises(ea.ExoplanetArchiveError):
        ea.query_confirmed_planets(root=tmp_path)
    with pytest.raises(ea.ExoplanetArchiveError):
        ea.query_confirmed_planets(host_name="Kepler-10", planet_name="Kepler-10 b", root=tmp_path)


def test_query_confirmed_planets_by_host_name_parses_rows(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ea.tap.netclient, "get", _fake_get_factory(calls))

    records = ea.query_confirmed_planets(host_name="Kepler-10", root=tmp_path)

    assert len(calls) == 1
    url, params, provider = calls[0]
    assert url == ea.EXOPLANET_ARCHIVE_TAP_URL
    assert provider == "exoplanetarchive"
    assert "hostname = 'Kepler-10'" in params["QUERY"]
    assert len(records) == 1
    record = records[0]
    assert record.name == "Kepler-10 b"
    assert record.period_days == pytest.approx(0.837491)
    assert record.depth_ppm == pytest.approx(152.0)


def test_query_confirmed_planets_by_planet_name_escapes_quotes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ea.tap.netclient, "get", _fake_get_factory(calls))

    ea.query_confirmed_planets(planet_name="O'Brien-1 b", root=tmp_path)

    _, params, _ = calls[0]
    assert "pl_name = 'O''Brien-1 b'" in params["QUERY"]


def test_query_confirmed_planets_propagates_service_failure(monkeypatch, tmp_path):
    def failing_get(url, params, timeout, provider, headers=None):
        raise RuntimeError("service unavailable")
    monkeypatch.setattr(ea.tap.netclient, "get", failing_get)

    with pytest.raises(ea.ExoplanetArchiveError):
        ea.query_confirmed_planets(host_name="Kepler-10", root=tmp_path)


def test_compare_to_published_reports_fractional_differences():
    published = ea.PlanetRecord(
        name="Kepler-10 b", host_name="Kepler-10", period_days=0.837491,
        period_err_days=1e-6, duration_hours=1.79, depth_ppm=152.0,
        radius_earth=1.47, transit_midpoint_bjd=2454964.57513,
    )
    diffs = ea.compare_to_published(fitted_period_days=0.8375, fitted_depth=0.000155,
                                    published=published)
    assert diffs["period_fractional_diff"] == pytest.approx((0.8375 - 0.837491) / 0.837491)
    assert diffs["depth_fractional_diff"] == pytest.approx((0.000155 - 152e-6) / 152e-6)


def test_compare_to_published_handles_missing_published_fields():
    published = ea.PlanetRecord(
        name="X", host_name="Y", period_days=None, period_err_days=None,
        duration_hours=None, depth_ppm=None, radius_earth=None, transit_midpoint_bjd=None,
    )
    diffs = ea.compare_to_published(fitted_period_days=1.0, fitted_depth=0.01, published=published)
    assert diffs == {"period_fractional_diff": None, "depth_fractional_diff": None}
