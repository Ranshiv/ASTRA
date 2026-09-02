"""biosignature.py: isothermal transmission-spectrum forward model."""

from __future__ import annotations

import numpy as np
import pytest

from astra import biosignature as bio


def _system() -> bio.SystemParameters:
    return bio.SystemParameters(stellar_radius_rsun=1.0, planet_mass_mjup=1.0)


def _wave() -> np.ndarray:
    return np.linspace(0.5, 5.0, 200)


class TestScaleHeight:
    def test_doubling_temperature_doubles_scale_height_exactly(self):
        system = _system()
        atm1 = bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                        reference_radius_rjup=1.0)
        atm2 = bio.AtmosphereParameters(temperature_k=2000.0, mean_molecular_weight=2.3,
                                        reference_radius_rjup=1.0)
        assert bio.scale_height_m(atm2, system) == pytest.approx(2.0 * bio.scale_height_m(atm1, system))

    def test_doubling_mass_at_fixed_radius_halves_scale_height(self):
        # g doubles when mass doubles at fixed radius, so H (~1/g) halves.
        atm = bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                       reference_radius_rjup=1.0)
        system_light = bio.SystemParameters(stellar_radius_rsun=1.0, planet_mass_mjup=1.0)
        system_heavy = bio.SystemParameters(stellar_radius_rsun=1.0, planet_mass_mjup=2.0)
        h_light = bio.scale_height_m(atm, system_light)
        h_heavy = bio.scale_height_m(atm, system_heavy)
        assert h_light == pytest.approx(2.0 * h_heavy)

    def test_non_positive_mass_or_radius_raises(self):
        with pytest.raises(bio.BiosignatureError):
            bio.surface_gravity_ms2(0.0, 1.0)
        with pytest.raises(bio.BiosignatureError):
            bio.surface_gravity_ms2(1.0, -1.0)


class TestTransitDepth:
    def test_zero_abundance_spectrum_is_exactly_flat(self):
        system = _system()
        atmosphere = bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                              reference_radius_rjup=1.0, abundances=())
        depth = bio.transit_depth(_wave(), atmosphere, system, cross_sections={})
        assert depth.max() - depth.min() == pytest.approx(0.0, abs=1e-15)

    def test_h2o_band_produces_a_real_feature(self):
        system = _system()
        atmosphere = bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                              reference_radius_rjup=1.0, abundances=(("H2O", 0.0),))
        depth = bio.transit_depth(_wave(), atmosphere, system, cross_sections={"H2O": 1e-2})
        assert depth.max() - depth.min() > 0.0

    def test_grey_cloud_deck_monotonically_flattens_the_feature(self):
        system = _system()
        wave = np.linspace(1.3, 1.5, 100)
        spans = []
        for log10_p in (-2.0, -1.0, 0.0, 1.0, 2.0):
            atmosphere = bio.AtmosphereParameters(
                temperature_k=1000.0, mean_molecular_weight=2.3, reference_radius_rjup=1.0,
                abundances=(("H2O", 0.0),), log10_cloud_pressure_bar=log10_p)
            depth = bio.transit_depth(wave, atmosphere, system, cross_sections={"H2O": 1e-2})
            spans.append(float(depth.max() - depth.min()))
        # Non-decreasing as the cloud deck deepens (larger log10 pressure).
        assert all(spans[i] <= spans[i + 1] + 1e-15 for i in range(len(spans) - 1))

    def test_non_positive_wavelength_raises(self):
        system = _system()
        atmosphere = bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                              reference_radius_rjup=1.0)
        with pytest.raises(bio.BiosignatureError):
            bio.transit_depth(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), atmosphere, system,
                              cross_sections={})


class TestAtmosphereParametersValidation:
    def test_non_positive_temperature_raises(self):
        with pytest.raises(bio.BiosignatureError):
            bio.AtmosphereParameters(temperature_k=0.0, mean_molecular_weight=2.3,
                                     reference_radius_rjup=1.0)

    def test_non_positive_mean_molecular_weight_raises(self):
        with pytest.raises(bio.BiosignatureError):
            bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=0.0,
                                     reference_radius_rjup=1.0)

    def test_non_positive_reference_radius_raises(self):
        with pytest.raises(bio.BiosignatureError):
            bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                     reference_radius_rjup=-1.0)

    def test_unknown_molecule_raises(self):
        with pytest.raises(bio.BiosignatureError):
            bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                     reference_radius_rjup=1.0, abundances=(("XX", 0.0),))

    def test_non_positive_system_params_raise(self):
        with pytest.raises(bio.BiosignatureError):
            bio.SystemParameters(stellar_radius_rsun=0.0, planet_mass_mjup=1.0)
        with pytest.raises(bio.BiosignatureError):
            bio.SystemParameters(stellar_radius_rsun=1.0, planet_mass_mjup=-1.0)


class TestArrayRoundTrip:
    def test_to_array_from_array_round_trips_exactly(self):
        atmosphere = bio.AtmosphereParameters(temperature_k=1234.0, mean_molecular_weight=2.3,
                                              reference_radius_rjup=1.1,
                                              abundances=(("H2O", -1.5), ("CH4", -3.0)))
        molecules = ("H2O", "CH4")
        array = atmosphere.to_array(molecules)
        restored = bio.AtmosphereParameters.from_array(array, molecules=molecules,
                                                        mean_molecular_weight=2.3)
        assert restored.temperature_k == pytest.approx(atmosphere.temperature_k)
        assert restored.reference_radius_rjup == pytest.approx(atmosphere.reference_radius_rjup)
        assert dict(restored.abundances) == pytest.approx(dict(atmosphere.abundances))


class TestFiniteArraysValidation:
    def test_fewer_than_five_points_raises(self):
        with pytest.raises(bio.BiosignatureError):
            bio._finite_arrays([1.0, 2.0], [0.01, 0.01], [0.001, 0.001])

    def test_non_monotone_wavelengths_raises_after_sort_check(self):
        # Duplicate wavelengths after sorting are not strictly increasing.
        with pytest.raises(bio.BiosignatureError):
            bio._finite_arrays([1.0, 1.0, 2.0, 3.0, 4.0], [0.01] * 5, [0.001] * 5)

    def test_nans_are_masked_out(self):
        wave, depth, err = bio._finite_arrays(
            [1.0, 2.0, float("nan"), 3.0, 4.0, 5.0], [0.01, 0.02, 0.5, 0.03, 0.04, 0.05],
            [0.001] * 6)
        assert len(wave) == 5

    def test_mismatched_lengths_raises(self):
        with pytest.raises(bio.BiosignatureError):
            bio._finite_arrays([1.0, 2.0, 3.0, 4.0, 5.0], [0.01, 0.02], [0.001] * 5)


def test_forward_model_returns_json_safe_payload():
    import json
    system = _system()
    atmosphere = bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                          reference_radius_rjup=1.0, abundances=(("H2O", 0.0),))
    result = bio.forward_model(_wave(), atmosphere, system, cross_sections={"H2O": 1e-2})
    json.dumps(result)  # must not raise
    assert result["scale_height_m"] > 0


class TestDefaultBounds:
    def test_negative_depth_raises(self):
        system = _system()
        with pytest.raises(bio.BiosignatureError):
            bio.default_bounds(np.array([-0.01, 0.01, 0.02]), system, molecules=("H2O",))

    def test_bounds_are_ordered(self):
        system = _system()
        depth = np.linspace(0.008, 0.012, 50)
        bounds = bio.default_bounds(depth, system, molecules=("H2O", "CH4"))
        assert len(bounds) == 2 + 2  # temperature, radius, + 2 molecules
        for lo, hi in bounds:
            assert lo < hi
