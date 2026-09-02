"""technosignature.py: brute-force de-Doppler search, hit finding, ON/OFF
cadence RFI rejection, and the synthetic injection harness."""

from __future__ import annotations

import numpy as np
import pytest

from astra import technosignature as tech


def _spectrum(**kwargs):
    return tech.synthesize_waterfall(n_time=16, n_freq=1024, **kwargs)["spectrum"]


class TestDynamicSpectrumValidation:
    def test_uniform_grid_at_gigahertz_scale_is_accepted(self):
        # Regression: a naive fixed-decimal uniqueness check on np.diff
        # spuriously rejects a perfectly uniform grid at ~1.4 GHz, where
        # float64 rounding noise in the last digits exceeds a fixed
        # decimal tolerance -- this must not raise.
        spectrum = _spectrum(f0_hz=1.4e9, channel_width_hz=2.7939677)
        assert spectrum.power.shape == (16, 1024)

    def test_shape_mismatch_raises(self):
        with pytest.raises(tech.TechnosignatureError):
            tech.DynamicSpectrum(time_s=np.arange(5.0), freq_hz=np.arange(10.0),
                                 power=np.zeros((5, 5)))

    def test_single_time_sample_raises(self):
        with pytest.raises(tech.TechnosignatureError):
            tech.DynamicSpectrum(time_s=np.array([0.0]), freq_hz=np.arange(10.0),
                                 power=np.zeros((1, 10)))

    def test_non_uniform_time_raises(self):
        with pytest.raises(tech.TechnosignatureError):
            tech.DynamicSpectrum(time_s=np.array([0.0, 1.0, 3.0, 4.0]),
                                 freq_hz=np.arange(4.0), power=np.zeros((4, 4)))

    def test_non_finite_power_raises(self):
        power = np.zeros((4, 4))
        power[0, 0] = np.nan
        with pytest.raises(tech.TechnosignatureError):
            tech.DynamicSpectrum(time_s=np.arange(4.0), freq_hz=np.arange(4.0), power=power)


class TestSearchNoSignal:
    def test_pure_noise_gives_zero_hits_across_many_trials(self):
        for seed in range(20):
            spectrum = _spectrum(snr=0.0, seed=seed)
            result = tech.search(spectrum, max_drift_hz_s=4.0, snr_threshold=10.0)
            assert result["hits"] == []


class TestSearchRecoversInjectedSignal:
    def test_zero_drift_recovered_at_injected_channel(self):
        injected = _spectrum(drift_rate_hz_s=0.0, snr=50.0, start_channel=512, seed=2)
        result = tech.search(injected, max_drift_hz_s=4.0, snr_threshold=10.0)
        assert result["hits"]
        top = result["hits"][0]
        assert top["freq_channel_index"] == 512
        assert top["drift_rate_hz_s"] == pytest.approx(0.0, abs=0.1)

    def test_positive_drift_recovered_with_correct_sign(self):
        injected = _spectrum(drift_rate_hz_s=2.0, snr=50.0, start_channel=512, seed=3)
        result = tech.search(injected, max_drift_hz_s=4.0, snr_threshold=10.0)
        top = result["hits"][0]
        assert top["drift_rate_hz_s"] > 1.0

    def test_negative_drift_recovered_with_correct_sign(self):
        injected = _spectrum(drift_rate_hz_s=-2.0, snr=50.0, start_channel=512, seed=4)
        result = tech.search(injected, max_drift_hz_s=4.0, snr_threshold=10.0)
        top = result["hits"][0]
        assert top["drift_rate_hz_s"] < -1.0

    def test_drift_at_grid_edge_is_recovered(self):
        injected = _spectrum(drift_rate_hz_s=4.0, snr=50.0, start_channel=512, seed=5)
        result = tech.search(injected, max_drift_hz_s=4.0, snr_threshold=10.0)
        assert result["hits"]
        top = result["hits"][0]
        assert top["drift_rate_hz_s"] > 3.0

    def test_drift_beyond_grid_is_not_recovered_at_truth(self):
        injected = _spectrum(drift_rate_hz_s=8.0, snr=50.0, start_channel=512, seed=6)
        result = tech.search(injected, max_drift_hz_s=4.0, snr_threshold=10.0)
        for hit in result["hits"]:
            assert hit["drift_rate_hz_s"] <= 4.0


class TestBandEdgeNoWrap:
    def test_low_edge_tone_does_not_produce_a_hit_at_the_opposite_edge(self):
        injected = _spectrum(drift_rate_hz_s=-2.0, snr=80.0, start_channel=0, seed=7)
        result = tech.search(injected, max_drift_hz_s=4.0, snr_threshold=10.0)
        assert all(hit["freq_channel_index"] < 900 for hit in result["hits"])

    def test_high_edge_tone_does_not_produce_a_hit_at_the_opposite_edge(self):
        injected = _spectrum(drift_rate_hz_s=2.0, snr=80.0, start_channel=1023, seed=8)
        result = tech.search(injected, max_drift_hz_s=4.0, snr_threshold=10.0)
        assert all(hit["freq_channel_index"] > 100 for hit in result["hits"])


class TestDriftRateGrid:
    def test_non_positive_max_drift_raises(self):
        spectrum = _spectrum()
        with pytest.raises(tech.TechnosignatureError):
            tech.drift_rate_grid(spectrum, max_drift_hz_s=0.0)

    def test_grid_spans_the_requested_range(self):
        spectrum = _spectrum()
        grid = tech.drift_rate_grid(spectrum, max_drift_hz_s=4.0)
        assert grid.min() == pytest.approx(-4.0)
        assert grid.max() == pytest.approx(4.0)


class TestCadenceFilter:
    def _hit(self, freq=1000.0, drift=0.0, snr=20.0, ch=5, di=3):
        return tech.TechnosignatureHit(frequency_hz=freq, drift_rate_hz_s=drift, snr=snr,
                                       freq_channel_index=ch, drift_index=di)

    def test_hit_present_in_all_on_absent_from_off_survives(self):
        on1 = [self._hit(freq=1000.0, drift=1.0)]
        on2 = [self._hit(freq=1000.03, drift=1.02)]
        off1 = [self._hit(freq=2000.0, drift=0.0)]
        survivors = tech.cadence_filter([on1, on2], [off1], frequency_tolerance_hz=1.0,
                                        drift_tolerance_hz_s=0.1)
        assert len(survivors) == 1

    def test_rfi_present_in_on_and_off_is_rejected(self):
        rfi = self._hit(freq=1000.0, drift=0.0)
        survivors = tech.cadence_filter([[rfi], [rfi]], [[rfi]], frequency_tolerance_hz=1.0,
                                        drift_tolerance_hz_s=0.1)
        assert survivors == []

    def test_hit_missing_from_one_on_scan_is_rejected(self):
        hit = self._hit(freq=1000.0)
        other_hit = self._hit(freq=5000.0)
        survivors = tech.cadence_filter([[hit], [other_hit]], [[]], frequency_tolerance_hz=1.0,
                                        drift_tolerance_hz_s=0.1)
        assert survivors == []

    def test_empty_on_list_raises(self):
        with pytest.raises(tech.TechnosignatureError):
            tech.cadence_filter([], [[]], frequency_tolerance_hz=1.0, drift_tolerance_hz_s=0.1)


class TestFindHits:
    def test_flat_zero_power_returns_no_hits(self):
        spectrum = tech.DynamicSpectrum(time_s=np.arange(4.0), freq_hz=np.arange(4.0) * 1.0,
                                        power=np.zeros((4, 4)))
        plane = tech.dedrift_bruteforce(spectrum, np.array([0.0]))
        hits = tech.find_hits(plane, spectrum, np.array([0.0]))
        assert hits == []
