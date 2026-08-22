"""Cross-messenger event-to-event correlation: the Bayes-factor statistic
and its scrambled-time-shift calibration. No network -- all events are
synthetic dicts shaped like `events.EventPacket.to_dict()`, following the
house convention already established for gw.py/frb.py/association.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astra import association


def _point_event(event_id, provider, ra_deg, dec_deg, event_time,
                 error_radius_arcsec=5.0):
    return {
        "event_id": event_id, "provider": provider, "event_time": event_time,
        "localization": {"type": "point", "ra_deg": ra_deg, "dec_deg": dec_deg,
                         "error_radius_arcsec": error_radius_arcsec},
    }


def _healpix_event(event_id, provider, event_time, pixels, nside):
    return {
        "event_id": event_id, "provider": provider, "event_time": event_time,
        "localization": {"type": "healpix", "pixels": pixels, "healpix_nside": nside},
    }


class TestEventToEventCorrelation:
    def test_same_provider_pairs_are_skipped(self):
        events = [
            _point_event("E1", "icecube", 180.0, 10.0, "2026-01-01T00:00:00Z"),
            _point_event("E2", "icecube", 180.0, 10.0, "2026-01-01T00:10:00Z"),
        ]
        assert association.event_to_event_correlation(events) == []

    def test_close_pair_scores_a_large_bayes_factor(self):
        events = [
            _point_event("GW1", "gw", 180.0, 10.0, "2026-01-01T00:00:00Z", error_radius_arcsec=3600.0),
            _point_event("FRB1", "frb", 180.001, 10.001, "2026-01-01T00:05:00Z", error_radius_arcsec=180.0),
        ]
        result = association.event_to_event_correlation(events, window_days=1.0,
                                                         background_window_days=365.0)
        assert len(result) == 1
        pair = result[0]
        assert pair["provider_a"] == "gw"
        assert pair["provider_b"] == "frb"
        assert pair["bayes_factor"] > 1.0
        assert math.isfinite(pair["log_bayes_factor"])

    def test_temporally_distant_pair_gets_a_zero_bayes_factor(self):
        events = [
            _point_event("GW1", "gw", 180.0, 10.0, "2026-01-01T00:00:00Z"),
            _point_event("FRB1", "frb", 180.0, 10.0, "2027-06-01T00:00:00Z"),
        ]
        result = association.event_to_event_correlation(events, window_days=1.0)
        assert len(result) == 1
        assert result[0]["bayes_factor"] == 0.0
        assert result[0]["log_bayes_factor"] == float("-inf")

    def test_farther_pair_scores_lower_than_a_closer_pair(self):
        base_time = "2026-01-01T00:00:00Z"
        close = [
            _point_event("GW1", "gw", 180.0, 10.0, base_time, error_radius_arcsec=60.0),
            _point_event("FRB1", "frb", 180.001, 10.0, base_time, error_radius_arcsec=60.0),
        ]
        far = [
            _point_event("GW1", "gw", 180.0, 10.0, base_time, error_radius_arcsec=60.0),
            _point_event("FRB1", "frb", 180.5, 10.0, base_time, error_radius_arcsec=60.0),
        ]
        close_result = association.event_to_event_correlation(close, window_days=1.0)
        far_result = association.event_to_event_correlation(far, window_days=1.0)
        assert close_result[0]["bayes_factor"] > far_result[0]["bayes_factor"]

    def test_events_without_a_position_are_skipped(self):
        events = [
            {"event_id": "E1", "provider": "icecube", "event_time": "2026-01-01T00:00:00Z",
             "localization": {}},
            _point_event("E2", "fermi", 180.0, 10.0, "2026-01-01T00:00:00Z"),
        ]
        assert association.event_to_event_correlation(events) == []

    def test_healpix_localized_events_are_supported(self):
        nside = 32
        import astropy.units as u
        from astropy_healpix import HEALPix
        healpix = HEALPix(nside=nside, order="nested")
        rng = np.random.default_rng(5)
        indices = rng.choice(healpix.npix, size=30, replace=False)
        raw = rng.random(30)
        pixels = [{"index": int(i), "probability": float(p)}
                 for i, p in zip(indices, raw / raw.sum())]
        peak_index = indices[np.argmax(raw)]
        lon, lat = healpix.healpix_to_lonlat(int(peak_index))

        events = [
            _healpix_event("GW1", "gw", "2026-01-01T00:00:00Z", pixels, nside),
            _point_event("FRB1", "frb", float(lon.to(u.deg).value),
                        float(lat.to(u.deg).value), "2026-01-01T00:05:00Z"),
        ]
        result = association.event_to_event_correlation(events, window_days=1.0)
        assert len(result) == 1
        assert result[0]["delta_theta_arcsec"] < 60.0  # near the map's own peak pixel


class TestSpatialAndTemporalLikelihoodRatios:
    def test_spatial_ratio_peaks_near_zero_offset_and_decays(self):
        near = association._spatial_likelihood_ratio(1.0, 60.0)
        far = association._spatial_likelihood_ratio(600.0, 60.0)
        assert near > far > 0

    def test_temporal_ratio_is_zero_outside_the_window(self):
        assert association._temporal_likelihood_ratio(2.0, window_days=1.0,
                                                       background_window_days=365.0) == 0.0

    def test_temporal_ratio_is_constant_inside_the_window(self):
        inside_a = association._temporal_likelihood_ratio(0.1, window_days=1.0,
                                                           background_window_days=365.0)
        inside_b = association._temporal_likelihood_ratio(-0.9, window_days=1.0,
                                                           background_window_days=365.0)
        assert inside_a == pytest.approx(inside_b)
        assert inside_a == pytest.approx(365.0)


def _background_population(rng, n=40, providers=("icecube", "fermi", "swift", "alerce")):
    """Many uncorrelated events in the same sky patch, spread over a year.

    A single true pair (as the other test in this class uses) gives the
    time-scramble procedure almost no chance of re-creating ANY coincidence
    at all -- with only two events, landing back within a day-scale window
    after a random shift is a vanishingly rare accident, so
    `significance.calibrate`'s reference population is correctly empty for
    that degenerate input, not buggy. A real calibration needs a real
    background RATE, which only shows up with many events.

    Positions are drawn from a modest patch around the true pair (a ~10 deg
    box), not the whole sky: `_spatial_likelihood_ratio`'s Rayleigh model
    uses arcsecond/arcminute-scale sigma, so a position drawn uniformly
    across the FULL sky is, correctly, astronomically unlikely to score a
    representable (non-underflowing) spatial likelihood at all -- that is
    the statistic doing its job, not a test bug, but it means a synthetic
    background needs to represent "the same monitored patch of sky a real
    multi-messenger follow-up campaign would restrict to" for the
    scramble's incidental-coincidence rate to be measurable at a workable
    sample size.
    """
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = []
    for i in range(n):
        when = base + timedelta(days=float(rng.uniform(0, 365)))
        events.append(_point_event(
            f"BG{i}", providers[i % len(providers)],
            float(rng.uniform(175.0, 185.0)), float(rng.uniform(5.0, 15.0)),
            when.isoformat(),
            error_radius_arcsec=float(rng.uniform(60.0, 600.0)),
        ))
    return events


class TestCalibrateEventGraph:
    def test_true_pair_separates_from_the_scrambled_background(self):
        rng = np.random.default_rng(9)
        events = _background_population(rng) + [
            _point_event("GW1", "gw", 180.0, 10.0, "2026-01-01T00:00:00Z", error_radius_arcsec=60.0),
            _point_event("FRB1", "frb", 180.001, 10.001, "2026-01-01T00:05:00Z",
                        error_radius_arcsec=60.0),
        ]
        result = association.calibrate_event_graph(
            events, window_days=1.0, background_window_days=365.0,
            n_trials=50, seed=3)

        assert result["observed_pairs"] > 1
        assert result["observed_finite_scores"] >= 1
        calibration = result["calibration"]
        assert calibration["ready"] is True
        assert 0.0 <= calibration["estimated_fdr"] <= 1.0
        # The true pair's own log-Bayes-factor should rank at or near the
        # top of everything observed -- it is by far the tightest temporal
        # and spatial coincidence in the population.
        observed = association.event_to_event_correlation(
            events, window_days=1.0, background_window_days=365.0)
        true_pair = next(p for p in observed if {p["provider_a"], p["provider_b"]} == {"gw", "frb"})
        others = [p["log_bayes_factor"] for p in observed if p is not true_pair
                 and math.isfinite(p["log_bayes_factor"])]
        assert not others or true_pair["log_bayes_factor"] >= max(others)

    def test_no_finite_pairs_still_returns_a_well_formed_report(self):
        events = [
            _point_event("GW1", "gw", 180.0, 10.0, "2026-01-01T00:00:00Z"),
            _point_event("FRB1", "frb", 180.0, 10.0, "2030-01-01T00:00:00Z"),
        ]
        result = association.calibrate_event_graph(
            events, window_days=1.0, n_trials=10, seed=1)
        assert result["observed_pairs"] == 1
        assert result["observed_finite_scores"] == 0
        assert result["calibration"]["ready"] is False
