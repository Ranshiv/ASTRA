"""habitability.py: Kopparapu HZ boundaries and Earth Similarity Index."""

from __future__ import annotations

import pytest

from astra import habitability as hab


def _sun() -> hab.StellarParameters:
    return hab.StellarParameters(teff_k=5780.0, radius_rsun=1.0, luminosity_lsun=1.0)


class TestHabitableZone:
    def test_sun_conservative_boundaries_match_paper(self):
        zone = hab.habitable_zone(_sun())
        assert zone["moist_greenhouse_au"] == pytest.approx(0.99, abs=0.02)
        assert zone["maximum_greenhouse_au"] == pytest.approx(1.67, abs=0.02)
        assert zone["conservative_inner_au"] == zone["moist_greenhouse_au"]
        assert zone["conservative_outer_au"] == zone["maximum_greenhouse_au"]

    def test_sun_optimistic_boundaries_match_paper(self):
        zone = hab.habitable_zone(_sun())
        assert zone["recent_venus_au"] == pytest.approx(0.75, abs=0.02)
        assert zone["early_mars_au"] == pytest.approx(1.77, abs=0.02)

    def test_earth_orbit_is_in_conservative_hz(self):
        planet = hab.PlanetParameters(radius_rearth=1.0, mass_mearth=1.0, semi_major_axis_au=1.0)
        result = hab.score(_sun(), planet)
        assert result["in_conservative_hz"] is True

    def test_venus_orbit_outside_conservative_and_optimistic_hz(self):
        # Venus's real orbit (0.723 AU) is closer in than even the "recent
        # Venus" empirical limit (0.75 AU) -- exactly the paper's own point
        # that Venus has been too hot for at least the last ~1 Gyr.
        planet = hab.PlanetParameters(semi_major_axis_au=0.723)
        result = hab.score(_sun(), planet)
        assert result["in_conservative_hz"] is False
        assert result["in_optimistic_hz"] is False

    def test_mars_orbit_inside_conservative_hz(self):
        # Catches an inner/outer boundary swap: Mars at 1.524 AU sits inside
        # the conservative outer edge (1.67 AU) but is often mistakenly
        # placed outside if inner/outer are transposed.
        planet = hab.PlanetParameters(semi_major_axis_au=1.524)
        result = hab.score(_sun(), planet)
        assert result["in_conservative_hz"] is True

    def test_teff_below_floor_is_extrapolated_not_raised(self):
        star = hab.StellarParameters(teff_k=2500.0, radius_rsun=0.2, luminosity_lsun=0.01)
        zone = hab.habitable_zone(star)
        assert zone["extrapolated"] is True
        result = hab.score(star, hab.PlanetParameters(semi_major_axis_au=0.05))
        assert result["quality"] == "insufficient"

    def test_teff_above_ceiling_is_extrapolated(self):
        zone = hab.habitable_zone(hab.StellarParameters(teff_k=7500.0, radius_rsun=1.5))
        assert zone["extrapolated"] is True

    def test_non_positive_teff_raises(self):
        with pytest.raises(hab.HabitabilityError):
            hab.StellarParameters(teff_k=0.0, radius_rsun=1.0)

    def test_non_positive_radius_raises(self):
        with pytest.raises(hab.HabitabilityError):
            hab.StellarParameters(teff_k=5780.0, radius_rsun=-1.0)

    def test_unknown_boundary_raises(self):
        with pytest.raises(hab.HabitabilityError):
            hab.effective_flux(5780.0, "not_a_real_boundary")


class TestEarthSimilarityIndex:
    def test_earth_via_true_surface_temperature_is_unity(self):
        # Definitional anchor: plugging Earth's own true surface temperature
        # (288 K), not T_eq, into the raw ESI term formula must give 1.0.
        radius_term = hab._esi_term(hab.ESI_EARTH_RADIUS_REARTH, hab.ESI_EARTH_RADIUS_REARTH,
                                    hab.ESI_WEIGHT_RADIUS)
        density_term = hab._esi_term(hab.ESI_EARTH_DENSITY_GCM3, hab.ESI_EARTH_DENSITY_GCM3,
                                     hab.ESI_WEIGHT_DENSITY)
        vesc_term = hab._esi_term(hab.ESI_EARTH_ESCAPE_VELOCITY_KMS,
                                  hab.ESI_EARTH_ESCAPE_VELOCITY_KMS, hab.ESI_WEIGHT_ESCAPE_VELOCITY)
        temp_term = hab._esi_term(hab.ESI_EARTH_SURFACE_TEMP_K, hab.ESI_EARTH_SURFACE_TEMP_K,
                                  hab.ESI_WEIGHT_SURFACE_TEMP)
        assert radius_term == pytest.approx(1.0)
        assert density_term == pytest.approx(1.0)
        assert vesc_term == pytest.approx(1.0)
        assert temp_term == pytest.approx(1.0)

    def test_earth_via_equilibrium_temperature_is_below_unity(self):
        # The documented substitution: T_eq != true surface temperature, so
        # Earth scored through the module's actual code path is < 1.0.
        planet = hab.PlanetParameters(radius_rearth=1.0, mass_mearth=1.0, semi_major_axis_au=1.0)
        result = hab.earth_similarity_index(planet, _sun())
        assert result["esi_global"] is not None
        assert result["esi_global"] < 1.0
        assert any("equilibrium temperature" in w for w in result["warnings"])

    def test_missing_mass_gives_none_interior_not_a_guess(self):
        planet = hab.PlanetParameters(radius_rearth=1.0, mass_mearth=None, semi_major_axis_au=1.0)
        result = hab.earth_similarity_index(planet, _sun())
        assert result["esi_interior"] is None
        assert result["esi_global"] is None
        assert any("mass" in w for w in result["warnings"])

    def test_negative_radius_raises(self):
        planet = hab.PlanetParameters(radius_rearth=-1.0, mass_mearth=1.0)
        with pytest.raises(hab.HabitabilityError):
            hab.earth_similarity_index(planet, _sun())

    def test_negative_mass_raises(self):
        planet = hab.PlanetParameters(radius_rearth=1.0, mass_mearth=-1.0)
        with pytest.raises(hab.HabitabilityError):
            hab.earth_similarity_index(planet, _sun())


class TestEquilibriumTemperature:
    def test_earth_equilibrium_temperature_is_plausible(self):
        # Earth's well-known equilibrium temperature (A=0.3) is ~255 K.
        t_eq = hab.equilibrium_temperature(5780.0, 1.0, 1.0, bond_albedo=0.3)
        assert t_eq == pytest.approx(255.0, abs=5.0)

    def test_non_positive_semimajor_axis_raises(self):
        with pytest.raises(hab.HabitabilityError):
            hab.equilibrium_temperature(5780.0, 1.0, 0.0)

    def test_albedo_out_of_range_raises(self):
        with pytest.raises(hab.HabitabilityError):
            hab.equilibrium_temperature(5780.0, 1.0, 1.0, bond_albedo=1.0)


class TestScoreArchivePlanet:
    def test_missing_stellar_params_raises(self, monkeypatch, tmp_path):
        from astra import exoplanet_archive as ea

        record = ea.PlanetRecord(name="X b", host_name="X", period_days=1.0,
                                 period_err_days=None, duration_hours=None, depth_ppm=None,
                                 radius_earth=1.0, transit_midpoint_bjd=None)
        monkeypatch.setattr(ea, "query_confirmed_planets", lambda **_: [record])
        with pytest.raises(hab.HabitabilityError):
            hab.score_archive_planet("X b", root=tmp_path)

    def test_no_record_found_raises(self, monkeypatch, tmp_path):
        from astra import exoplanet_archive as ea

        monkeypatch.setattr(ea, "query_confirmed_planets", lambda **_: [])
        with pytest.raises(hab.HabitabilityError):
            hab.score_archive_planet("Nonexistent b", root=tmp_path)


class TestRankPlanets:
    def test_ranks_by_esi_descending_and_skips_incomplete(self):
        from astra import exoplanet_archive as ea

        good = ea.PlanetRecord(name="Good b", host_name="Good", period_days=1.0,
                               period_err_days=None, duration_hours=None, depth_ppm=None,
                               radius_earth=1.0, transit_midpoint_bjd=None,
                               mass_earth=1.0, semimajor_au=1.0, st_teff_k=5780.0,
                               st_radius_rsun=1.0, st_luminosity_lsun=1.0)
        incomplete = ea.PlanetRecord(name="NoTeff b", host_name="NoTeff", period_days=1.0,
                                     period_err_days=None, duration_hours=None, depth_ppm=None,
                                     radius_earth=1.0, transit_midpoint_bjd=None)
        ranked = hab.rank_planets([incomplete, good], limit=10)
        assert len(ranked) == 1
        assert ranked[0]["planet_name"] == "Good b"

    def test_empty_input_returns_empty_list(self):
        assert hab.rank_planets([], limit=10) == []
