"""Stefan-Boltzmann luminosity arithmetic, E=ED*L energy arithmetic, and the
synthetic injection-recovery study's three metrics."""

from pathlib import Path

import pytest

from astra import flare_energy as fe


def test_quiescent_bolometric_luminosity_matches_nominal_solar_value():
    # Teff=5772 K, R=1 Rsun -> the real IAU nominal solar luminosity,
    # 3.828e33 erg/s, to 4 significant figures.
    luminosity = fe.quiescent_bolometric_luminosity(1.0, 5772.0)
    assert luminosity == pytest.approx(3.828e33, rel=1e-3)


def test_quiescent_bolometric_luminosity_scales_as_r_squared_teff_fourth():
    base = fe.quiescent_bolometric_luminosity(1.0, 5000.0)
    doubled_radius = fe.quiescent_bolometric_luminosity(2.0, 5000.0)
    assert doubled_radius == pytest.approx(base * 4.0, rel=1e-9)
    doubled_teff = fe.quiescent_bolometric_luminosity(1.0, 10000.0)
    assert doubled_teff == pytest.approx(base * 16.0, rel=1e-9)


def test_quiescent_bolometric_luminosity_rejects_non_positive_input():
    with pytest.raises(fe.FlareEnergyError):
        fe.quiescent_bolometric_luminosity(0.0, 5000.0)
    with pytest.raises(fe.FlareEnergyError):
        fe.quiescent_bolometric_luminosity(1.0, 0.0)


def test_bolometric_flare_energy_is_ed_times_luminosity():
    energy = fe.bolometric_flare_energy(100.0, 3.828e33)
    assert energy == pytest.approx(100.0 * 3.828e33)


def test_bolometric_flare_energy_rejects_bad_input():
    with pytest.raises(fe.FlareEnergyError):
        fe.bolometric_flare_energy(-1.0, 1e33)
    with pytest.raises(fe.FlareEnergyError):
        fe.bolometric_flare_energy(10.0, 0.0)


# ---------------------------------------------------------------------------
# Synthetic injection-recovery study
# ---------------------------------------------------------------------------

def test_evaluate_flare_recovery_completeness_increases_with_amplitude():
    results = fe.evaluate_flare_recovery(
        amplitude_grid=[0.01, 0.3], fwhm_days=0.03, radius_solar=1.0, teff_k=5000.0,
        n_trials_per_amplitude=10, noise_sigma=0.003, seed=7)
    faint, bright = results[0], results[1]
    assert bright.completeness >= faint.completeness
    assert bright.completeness == pytest.approx(1.0)


def test_evaluate_flare_recovery_reports_small_energy_bias_at_high_amplitude():
    results = fe.evaluate_flare_recovery(
        amplitude_grid=[0.3], fwhm_days=0.03, radius_solar=1.0, teff_k=5000.0,
        n_trials_per_amplitude=10, noise_sigma=0.002, seed=11)
    summary = results[0].to_dict()["energy_fractional_bias"]
    assert summary is not None
    assert abs(summary["mean"]) < 0.3


def test_evaluate_flare_recovery_rejects_empty_grid():
    with pytest.raises(fe.FlareEnergyError):
        fe.evaluate_flare_recovery(amplitude_grid=[])


def test_evaluate_flare_recovery_rejects_bad_trial_count():
    with pytest.raises(fe.FlareEnergyError):
        fe.evaluate_flare_recovery(amplitude_grid=[0.1], n_trials_per_amplitude=0)


def test_amplitude_trial_result_to_dict_handles_zero_recoveries():
    result = fe.AmplitudeTrialResult(amplitude=0.001, n_injected=5, n_recovered=0)
    payload = result.to_dict()
    assert payload["completeness"] == 0.0
    assert payload["energy_fractional_bias"] is None


def test_flare_energy_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "flare" not in rpc_source
