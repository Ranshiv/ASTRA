"""Kepler's-third-law arithmetic, mass-splitting, and radius-anchor fallback,
validated against a self-consistent synthetic binary and known closed forms."""

import pytest

from astra import eclipsing_binary as eb
from astra import eclipsing_binary_dimensions as ebd


def test_mass_and_radius_at_teff_matches_table_endpoints():
    mass, radius = ebd.mass_and_radius_at_teff(5860.0)  # G1V row, exact match
    assert mass == pytest.approx(1.03)
    assert radius == pytest.approx(1.060)


def test_mass_and_radius_at_teff_clamps_out_of_range():
    hot_mass, hot_radius = ebd.mass_and_radius_at_teff(50000.0)
    assert hot_mass == pytest.approx(2.75)  # clamped to the hottest row, B9V
    cool_mass, cool_radius = ebd.mass_and_radius_at_teff(1000.0)
    assert cool_mass == pytest.approx(0.090)  # clamped to the coolest row, M7V


def test_mass_and_radius_at_teff_rejects_non_positive_input():
    with pytest.raises(ebd.EclipsingBinaryDimensionsError):
        ebd.mass_and_radius_at_teff(0.0)


def test_anchor_physical_radius_prefers_gaia_gspphot():
    radius = ebd.anchor_physical_radius(radius_gspphot=1.06, bp_rp=99.0, abs_g_mag=99.0)
    assert radius == pytest.approx(1.06)


def test_anchor_physical_radius_falls_back_to_cmd_track():
    radius = ebd.anchor_physical_radius(bp_rp=0.803, abs_g_mag=4.462)  # G1V row
    assert radius == pytest.approx(1.060)


def test_anchor_physical_radius_raises_with_no_real_source():
    with pytest.raises(ebd.EclipsingBinaryDimensionsError):
        ebd.anchor_physical_radius()


def test_absolute_dimensions_recovers_self_consistent_synthetic_binary():
    # Two real rows from MASS_RADIUS_ZAMS_TRACK: G1V (star 1) and K1V (star 2).
    m1_true, r1_true_solar, teff1_true = 1.03, 1.060, 5860.0
    m2_true, r2_true_solar, teff2_true = 0.86, 0.797, 5170.0
    period_days = 5.0
    period_years = period_days / 365.25
    total_mass_true = m1_true + m2_true
    a_au_true = (total_mass_true * period_years ** 2) ** (1 / 3)
    a_solar_true = a_au_true * ebd.AU_IN_SOLAR_RADII
    r1_a_true = r1_true_solar / a_solar_true
    r2_a_true = r2_true_solar / a_solar_true
    teff_ratio_true = teff2_true / teff1_true

    fit = eb.EclipsingBinaryFit(t0=0.5, period_days=period_days, r1_a=r1_a_true, r2_a=r2_a_true,
                                inc_deg=89.0, u1_1=0.3, u2_1=0.2, u1_2=0.3, u2_2=0.2,
                                teff_ratio=teff_ratio_true, residual_rms=0.0, n_evaluations=0)

    anchor = ebd.anchor_physical_radius(radius_gspphot=r1_true_solar)
    dims = ebd.absolute_dimensions(fit, anchor, teff1_true)

    assert dims.r1_solar == pytest.approx(r1_true_solar)
    assert dims.r2_solar == pytest.approx(r2_true_solar, rel=1e-6)
    assert dims.a_au == pytest.approx(a_au_true, rel=1e-6)
    assert dims.total_mass_solar == pytest.approx(total_mass_true, rel=1e-6)
    assert dims.m1_solar == pytest.approx(m1_true, rel=1e-6)
    assert dims.m2_solar == pytest.approx(m2_true, rel=1e-6)


def test_absolute_dimensions_rejects_non_positive_anchor():
    fit = eb.EclipsingBinaryFit(t0=0.0, period_days=5.0, r1_a=0.07, r2_a=0.05, inc_deg=89.0,
                                u1_1=0.3, u2_1=0.2, u1_2=0.3, u2_2=0.2, teff_ratio=0.9,
                                residual_rms=0.0, n_evaluations=0)
    with pytest.raises(ebd.EclipsingBinaryDimensionsError):
        ebd.absolute_dimensions(fit, -1.0, 5800.0)
    with pytest.raises(ebd.EclipsingBinaryDimensionsError):
        ebd.absolute_dimensions(fit, 1.0, 0.0)


# ---------------------------------------------------------------------------
# EB mass/radius catalog cross-check (mocked VizieR)
# ---------------------------------------------------------------------------

class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def test_query_component_catalog_parses_real_column_shape(monkeypatch):
    calls = []
    row = {"Name": "V760 Sco B", "Mass": 4.609, "e_Mass": 0.073,
          "Rad": 2.642, "e_Rad": 0.066, "Teff": 16300.0}

    class _FakeVizier:
        def __init__(self, columns=None, row_limit=None):
            pass

        def query_region(self, coord, radius, catalog):
            calls.append(catalog)
            return [_FakeTable([row])]

    monkeypatch.setattr("astroquery.vizier.Vizier", _FakeVizier)

    results = ebd.query_component_catalog(246.18216, -34.89375, radius_arcsec=10.0)

    assert calls == [ebd.EB_MASS_RADIUS_CATALOG]
    assert len(results) == 1
    assert results[0]["name"] == "V760 Sco B"
    assert results[0]["mass_solar"] == pytest.approx(4.609)
    assert results[0]["radius_solar"] == pytest.approx(2.642)


def test_query_component_catalog_rejects_non_positive_radius():
    with pytest.raises(ebd.EclipsingBinaryDimensionsError):
        ebd.query_component_catalog(0.0, 0.0, radius_arcsec=0.0)


def test_mass_radius_residuals_reports_fractional_differences():
    dims = ebd.AbsoluteDimensions(r1_solar=1.06, r2_solar=0.8, a_au=0.05,
                                  total_mass_solar=1.9, m1_solar=1.03, m2_solar=0.87,
                                  teff1_k=5860.0, teff2_k=5000.0)
    component1 = {"mass_solar": 1.0, "radius_solar": 1.0}
    component2 = {"mass_solar": 0.9, "radius_solar": 0.85}

    residuals = ebd.mass_radius_residuals(dims, component1, component2)

    assert residuals["mass1_fractional_diff"] == pytest.approx((1.03 - 1.0) / 1.0)
    assert residuals["radius1_fractional_diff"] == pytest.approx((1.06 - 1.0) / 1.0)
    assert residuals["mass2_fractional_diff"] == pytest.approx((0.87 - 0.9) / 0.9)
    assert residuals["radius2_fractional_diff"] == pytest.approx((0.8 - 0.85) / 0.85)


def test_mass_radius_residuals_handles_missing_catalog_fields():
    dims = ebd.AbsoluteDimensions(r1_solar=1.06, r2_solar=0.8, a_au=0.05,
                                  total_mass_solar=1.9, m1_solar=1.03, m2_solar=0.87,
                                  teff1_k=5860.0, teff2_k=5000.0)
    residuals = ebd.mass_radius_residuals(dims, {}, {})
    assert all(value is None for value in residuals.values())
