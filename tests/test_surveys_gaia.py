"""`gaia.query_eclipsing_binary`/`query_astrophysical_parameters` query
construction and response parsing, mocking `astroquery.gaia.Gaia.launch_job`."""

import numpy as np
import pytest
from astropy.table import Table, MaskedColumn

from astra.surveys import gaia


class _FakeJob:
    def __init__(self, table):
        self._table = table

    def get_results(self):
        return self._table


def test_query_eclipsing_binary_parses_real_column_shape(monkeypatch):
    calls = []
    table = Table({
        "source_id": [123],
        "frequency": [2.0],  # 1/d -> period_days = 0.5
        "derived_primary_ecl_phase": [0.5],
        "derived_primary_ecl_duration": [0.1],
        "derived_primary_ecl_depth": [0.3],
        "derived_secondary_ecl_phase": [0.0],
        "derived_secondary_ecl_duration": [0.12],
        "derived_secondary_ecl_depth": [0.15],
    })

    def fake_launch_job(query):
        calls.append(query)
        return _FakeJob(table)

    from astroquery.gaia import Gaia
    monkeypatch.setattr(Gaia, "launch_job", fake_launch_job)

    result = gaia.query_eclipsing_binary(123)

    assert len(calls) == 1
    assert "source_id = 123" in calls[0]
    assert result["period_days"] == pytest.approx(0.5)
    assert result["primary_eclipse_depth"] == pytest.approx(0.3)
    assert result["secondary_eclipse_depth"] == pytest.approx(0.15)


def test_query_eclipsing_binary_returns_none_for_empty_result(monkeypatch):
    def fake_launch_job(query):
        return _FakeJob(Table({"source_id": [], "frequency": [],
                               "derived_primary_ecl_phase": [], "derived_primary_ecl_duration": [],
                               "derived_primary_ecl_depth": [], "derived_secondary_ecl_phase": [],
                               "derived_secondary_ecl_duration": [], "derived_secondary_ecl_depth": []}))

    from astroquery.gaia import Gaia
    monkeypatch.setattr(Gaia, "launch_job", fake_launch_job)

    assert gaia.query_eclipsing_binary(999) is None


def test_query_eclipsing_binary_rejects_non_integer_source_id():
    with pytest.raises(ValueError):
        gaia.query_eclipsing_binary("not-an-id")


def test_query_astrophysical_parameters_parses_and_masks_missing_fields(monkeypatch):
    table = Table({
        "source_id": [123],
        "teff_gspphot": MaskedColumn([8500.0], mask=[False]),
        "radius_gspphot": MaskedColumn([1.06], mask=[False]),
        "mass_flame": MaskedColumn([np.nan], mask=[True]),
        "radius_flame": MaskedColumn([np.nan], mask=[True]),
    })

    def fake_launch_job(query):
        return _FakeJob(table)

    from astroquery.gaia import Gaia
    monkeypatch.setattr(Gaia, "launch_job", fake_launch_job)

    result = gaia.query_astrophysical_parameters(123)

    assert result["teff_gspphot"] == pytest.approx(8500.0)
    assert result["radius_gspphot"] == pytest.approx(1.06)
    assert result["mass_flame"] is None
    assert result["radius_flame"] is None


def test_query_astrophysical_parameters_returns_none_for_empty_result(monkeypatch):
    def fake_launch_job(query):
        return _FakeJob(Table({"source_id": [], "teff_gspphot": [], "radius_gspphot": [],
                               "mass_flame": [], "radius_flame": []}))

    from astroquery.gaia import Gaia
    monkeypatch.setattr(Gaia, "launch_job", fake_launch_job)

    assert gaia.query_astrophysical_parameters(1) is None


def test_query_extinction_estimate_parses_real_column_shape(monkeypatch):
    table = Table({
        "source_id": [4056453296603930624], "ra": [268.196], "dec": [-29.655],
        "parallax": [2.8194], "parallax_error": [0.126],
        "ag_gspphot": MaskedColumn([0.8337], mask=[False]),
    })

    def fake_launch_job(query):
        return _FakeJob(table)

    from astroquery.gaia import Gaia
    monkeypatch.setattr(Gaia, "launch_job", fake_launch_job)

    result = gaia.query_extinction_estimate(4056453296603930624)
    assert result["parallax_mas"] == pytest.approx(2.8194)
    assert result["ag_gspphot_mag"] == pytest.approx(0.8337)


def test_query_extinction_estimate_returns_none_for_non_positive_parallax(monkeypatch):
    table = Table({
        "source_id": [1], "ra": [0.0], "dec": [0.0],
        "parallax": [-0.5], "parallax_error": [0.1],
        "ag_gspphot": MaskedColumn([0.5], mask=[False]),
    })

    def fake_launch_job(query):
        return _FakeJob(table)

    from astroquery.gaia import Gaia
    monkeypatch.setattr(Gaia, "launch_job", fake_launch_job)

    assert gaia.query_extinction_estimate(1) is None


def test_query_extinction_estimate_returns_none_for_empty_result(monkeypatch):
    def fake_launch_job(query):
        return _FakeJob(Table({"source_id": [], "ra": [], "dec": [], "parallax": [],
                               "parallax_error": [], "ag_gspphot": []}))

    from astroquery.gaia import Gaia
    monkeypatch.setattr(Gaia, "launch_job", fake_launch_job)

    assert gaia.query_extinction_estimate(1) is None
