"""Polarization feature pipeline: query helpers, Stokes round-trip,
angle-error formula, and the class-separability mechanism (roadmap item
28). Follows tests/test_surveys_vlass.py's VOTable fixture shape."""

from __future__ import annotations

import math

import pytest

from astra import netclient, polarization_features as pf, rpc


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.headers = {"Content-Type": "application/x-votable+xml"}


def _votable(fields: list[str], rows: list[list[str]]) -> str:
    field_xml = "".join(f'<FIELD name="{name}"/>' for name in fields)
    row_xml = "".join(
        "<TR>" + "".join(f"<TD>{value}</TD>" for value in row) + "</TR>" for row in rows)
    return (
        '<?xml version="1.0"?><VOTABLE><RESOURCE><TABLE>'
        f"<DATA><TABLEDATA>{row_xml}</TABLEDATA></DATA>"
        "</TABLE></RESOURCE></VOTABLE>"
    ).replace("<TABLE>", f"<TABLE>{field_xml}")


class TestQueryOpticalPolarization:
    def test_parses_a_real_matching_row(self, monkeypatch):
        payload = _votable(["HIP", "PV", "e_PV", "PA", "e_PA"], [["1234", "1.5", "0.1", "45.0", "2.0"]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        result = pf.query_optical_polarization(180.0, 0.0)
        assert result["hip"] == 1234
        assert result["p_percent"] == pytest.approx(1.5)
        assert result["theta_deg"] == pytest.approx(45.0)

    def test_returns_none_when_no_match(self, monkeypatch):
        payload = _votable(["HIP", "PV", "PA"], [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert pf.query_optical_polarization(180.0, 0.0) is None

    def test_uses_the_vizier_provider(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["provider"] = provider
            captured["params"] = params
            return _FakeResponse(_votable(["HIP", "PV", "PA"], []))

        monkeypatch.setattr(netclient, "get", fake_get)
        pf.query_optical_polarization(180.0, 0.0)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == pf.OPTICAL_POLARIZATION_CATALOG


class TestQueryOpticalPolarizationSecondary:
    def test_parses_a_real_matching_row(self, monkeypatch):
        payload = _votable(["HR", "Pol", "e_Pol", "theta", "e_theta", "SpType"],
                           [["3982", "36.7", "0.8", "78.9", "0.8", "B7V"]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        result = pf.query_optical_polarization_secondary(180.0, 0.0)
        assert result["hr"] == 3982
        assert result["pol_raw_ppm"] == pytest.approx(36.7)
        assert result["p_percent"] == pytest.approx(36.7 * pf._PPM_TO_PERCENT)
        assert result["spectral_type"] == "B7V"

    def test_returns_none_when_no_match(self, monkeypatch):
        payload = _votable(["HR", "Pol", "theta"], [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert pf.query_optical_polarization_secondary(180.0, 0.0) is None


class TestQueryOpticalPolarizationAny:
    def test_prefers_primary_catalog_when_it_matches(self, monkeypatch):
        primary_payload = _votable(["HIP", "PV", "PA"], [["1234", "1.5", "45.0"]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(primary_payload))
        result = pf.query_optical_polarization_any(180.0, 0.0)
        assert result["source_catalog"] == pf.OPTICAL_POLARIZATION_CATALOG
        assert result["hip"] == 1234

    def test_falls_back_to_secondary_catalog(self, monkeypatch):
        empty_primary = _votable(["HIP", "PV", "PA"], [])
        secondary_payload = _votable(["HR", "Pol", "theta"], [["3982", "36.7", "78.9"]])
        calls = {"n": 0}

        def fake_get(url, params, timeout, provider):
            calls["n"] += 1
            payload = empty_primary if calls["n"] == 1 else secondary_payload
            return _FakeResponse(payload)

        monkeypatch.setattr(netclient, "get", fake_get)
        result = pf.query_optical_polarization_any(180.0, 0.0)
        assert result["source_catalog"] == pf.OPTICAL_POLARIZATION_CATALOG_SECONDARY
        assert result["hr"] == 3982
        assert result["p_percent"] == pytest.approx(36.7 * pf._PPM_TO_PERCENT)

    def test_returns_none_when_neither_catalog_matches(self, monkeypatch):
        empty = _votable(["HIP", "PV", "PA"], [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(empty))
        assert pf.query_optical_polarization_any(180.0, 0.0) is None


class TestQueryRotationMeasure:
    def test_parses_a_real_matching_row(self, monkeypatch):
        payload = _votable(["RM", "e_RM", "Tel"], [["25.4", "1.2", "VLA"]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        result = pf.query_rotation_measure(180.0, 0.0)
        assert result["rm_rad_per_m2"] == pytest.approx(25.4)
        assert result["telescope"] == "VLA"

    def test_returns_none_when_no_match(self, monkeypatch):
        payload = _votable(["RM", "e_RM"], [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert pf.query_rotation_measure(180.0, 0.0) is None


class TestQueryRotationMeasureNvss:
    def test_parses_a_real_matching_row(self, monkeypatch):
        payload = _votable(["RM", "e_RM", "Si", "Pk"], [["-18.3", "2.1", "450.0", "12.4"]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        result = pf.query_rotation_measure_nvss(180.0, 0.0)
        assert result["rm_rad_per_m2"] == pytest.approx(-18.3)
        assert result["stokes_i_flux_mjy"] == pytest.approx(450.0)

    def test_returns_none_when_no_match(self, monkeypatch):
        payload = _votable(["RM", "e_RM"], [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert pf.query_rotation_measure_nvss(180.0, 0.0) is None

    def test_uses_the_vizier_provider(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["provider"] = provider
            captured["params"] = params
            return _FakeResponse(_votable(["RM", "e_RM"], []))

        monkeypatch.setattr(netclient, "get", fake_get)
        pf.query_rotation_measure_nvss(180.0, 0.0)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == pf.ROTATION_MEASURE_CATALOG_NVSS


class TestQueryRotationMeasureAny:
    def test_prefers_nvss_catalog_when_it_matches(self, monkeypatch):
        payload = _votable(["RM", "e_RM", "Si", "Pk"], [["-18.3", "2.1", "450.0", "12.4"]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        result = pf.query_rotation_measure_any(180.0, 0.0)
        assert result["source_catalog"] == pf.ROTATION_MEASURE_CATALOG_NVSS
        assert result["rm_rad_per_m2"] == pytest.approx(-18.3)

    def test_falls_back_to_xu_catalog(self, monkeypatch):
        empty_nvss = _votable(["RM", "e_RM"], [])
        xu_payload = _votable(["RM", "e_RM", "Tel"], [["25.4", "1.2", "VLA"]])
        calls = {"n": 0}

        def fake_get(url, params, timeout, provider):
            calls["n"] += 1
            payload = empty_nvss if calls["n"] == 1 else xu_payload
            return _FakeResponse(payload)

        monkeypatch.setattr(netclient, "get", fake_get)
        result = pf.query_rotation_measure_any(180.0, 0.0)
        assert result["source_catalog"] == pf.ROTATION_MEASURE_CATALOG
        assert result["rm_rad_per_m2"] == pytest.approx(25.4)

    def test_returns_none_when_neither_catalog_matches(self, monkeypatch):
        empty = _votable(["RM", "e_RM"], [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(empty))
        assert pf.query_rotation_measure_any(180.0, 0.0) is None


class TestStokesRoundTrip:
    @pytest.mark.parametrize("p_percent,theta_deg", [(1.5, 45.0), (5.0, 10.0), (0.2, 170.0)])
    def test_round_trips_p_theta_through_stokes(self, p_percent, theta_deg):
        q, u = pf.stokes_from_p_theta(p_percent, theta_deg)
        recovered_p, recovered_theta = pf.p_theta_from_stokes(q, u)
        assert recovered_p == pytest.approx(p_percent, abs=1e-9)
        assert recovered_theta == pytest.approx(theta_deg % 180.0, abs=1e-6)

    def test_zero_polarization_gives_zero_stokes(self):
        q, u = pf.stokes_from_p_theta(0.0, 30.0)
        assert q == pytest.approx(0.0)
        assert u == pytest.approx(0.0)


class TestPolarizationAngleError:
    def test_matches_the_closed_form_serkowski_formula(self):
        # sigma_theta = 28.65 * sigma_p / p, exactly.
        result = pf.polarization_angle_error_deg(p_percent=4.0, p_error_percent=0.5)
        assert result == pytest.approx(28.65 * 0.5 / 4.0, abs=1e-3)

    def test_smaller_relative_error_gives_smaller_angle_error(self):
        precise = pf.polarization_angle_error_deg(10.0, 0.1)
        noisy = pf.polarization_angle_error_deg(10.0, 2.0)
        assert precise < noisy

    def test_returns_none_for_zero_polarization(self):
        assert pf.polarization_angle_error_deg(0.0, 0.1) is None

    def test_returns_none_for_negative_polarization(self):
        assert pf.polarization_angle_error_deg(-1.0, 0.1) is None


class TestEvaluateClassSeparability:
    def test_mechanism_separates_synthetic_classes_above_chance(self):
        result = pf.evaluate_class_separability(n_per_class=100, seed=23)
        assert result["classes"] == ["unpolarized_star", "polarized_variable", "blazar_like"]
        assert result["macro_f1"] > 0.6  # chance-level 3-class macro-F1 is ~0.33

    def test_is_reproducible_for_a_fixed_seed(self):
        first = pf.evaluate_class_separability(n_per_class=50, seed=9)
        second = pf.evaluate_class_separability(n_per_class=50, seed=9)
        assert first["macro_f1"] == second["macro_f1"]


@pytest.mark.live
class TestEvaluateRealClassSeparabilityLive:
    """Confirmed live this session (2026-08-26): ALeRCE's `class=AGN` and
    `class=YSO` filters, NVSS RM cross-matching, and ALeRCE light-curve
    fetches all reach real services. Hits multiple real, rate-limited
    services (tens of requests) -- skipped by default."""

    def test_runs_against_real_alerce_and_nvss_data(self):
        result = pf.evaluate_real_class_separability(n_per_class=10)
        assert result["n_used"] >= 0
        if result["macro_f1"] is not None:
            assert 0.0 <= result["macro_f1"] <= 1.0


class TestNotWiredIntoRpc:
    def test_polarization_features_is_not_referenced_by_rpc(self):
        import inspect

        assert "polarization_features" not in inspect.getsource(rpc)


@pytest.mark.live
class TestPolarizationLive:
    """Confirmed live this session (2026-08-25): VizieR hosts the real
    "Optical polarization for 878 Hipparcos stars" (`J/ApJ/728/104`) and
    "Rotation measures of radio point sources" (`J/other/RAA/14.942`)."""

    def test_optical_polarization_query_reaches_a_real_service(self):
        # A generic sky position with no guaranteed match -- this only
        # confirms the request/response contract, not a specific source.
        result = pf.query_optical_polarization(180.0, 0.0, radius_arcsec=3600.0)
        assert result is None or result["p_percent"] >= 0

    def test_rotation_measure_query_reaches_a_real_service(self):
        result = pf.query_rotation_measure(180.0, 0.0, radius_arcsec=3600.0)
        assert result is None or math.isfinite(result["rm_rad_per_m2"])

    def test_secondary_polarization_query_reaches_a_real_service(self):
        result = pf.query_optical_polarization_secondary(180.0, 0.0, radius_arcsec=3600.0)
        assert result is None or math.isfinite(result["pol_raw_ppm"])

    def test_nvss_rotation_measure_query_returns_real_rows(self):
        # Taylor+2009's 37,543-source catalogue is dense enough that a
        # 1-degree cone around a generic position is expected to match,
        # unlike the smaller catalogues above.
        result = pf.query_rotation_measure_nvss(180.0, 0.0, radius_arcsec=3600.0)
        assert result is not None
        assert math.isfinite(result["rm_rad_per_m2"])
