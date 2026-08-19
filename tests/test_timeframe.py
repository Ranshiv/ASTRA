"""Time-frame conversion to BJD_TDB."""

from __future__ import annotations

import numpy as np
import pytest

from astra import timeframe
from astra.surveys.base import LightCurve, SourceRef

RA, DEC = 291.3663, 42.7844
EPOCH = 2458600.5


def make_curve(time, system, survey="ZTF"):
    time = np.asarray(time, dtype=float)
    return LightCurve(
        source=SourceRef(survey=survey, object_id="x", ra_deg=RA, dec_deg=DEC),
        release="dr24", band="g", value_kind="mag",
        time=time, value=np.full(len(time), 18.0),
        value_err=np.full(len(time), 0.01), time_system=system,
    )


class TestOffsetMagnitude:
    def test_total_offset_is_about_a_minute_not_eight(self):
        """The +-8.3 min figure is geocentric->barycentric; HJD absorbs it."""
        offset = timeframe.measure_frame_offset("HJD_UTC", RA, DEC, EPOCH)
        assert 60.0 < offset.total_seconds < 80.0

    def test_scale_conversion_dominates(self):
        """TDB - UTC is ~69 s; the helio->bary residual is only a few seconds."""
        offset = timeframe.measure_frame_offset("HJD_UTC", RA, DEC, EPOCH)
        assert abs(offset.scale_seconds) > 60.0
        assert abs(offset.reference_seconds) < 10.0

    def test_offset_is_comparable_to_a_tess_cadence(self):
        """Which is why it cannot simply be ignored."""
        offset = timeframe.measure_frame_offset("HJD_UTC", RA, DEC, EPOCH)
        assert offset.total_seconds > 0.25 * 120.0

    def test_offset_depends_on_sky_position(self):
        east = timeframe.measure_frame_offset("HJD_UTC", 90.0, 0.0, EPOCH)
        west = timeframe.measure_frame_offset("HJD_UTC", 270.0, 0.0, EPOCH)
        assert abs(east.total_seconds - west.total_seconds) > 1.0

    def test_target_frame_needs_no_correction(self):
        offset = timeframe.measure_frame_offset("BJD_TDB", RA, DEC, EPOCH)
        assert offset.total_seconds == 0.0


class TestConversion:
    def test_barycentric_input_is_untouched(self):
        time = np.array([2458600.5, 2458601.5])
        out = timeframe.to_bjd_tdb(time, "BJD_TDB", RA, DEC, "TESS")
        np.testing.assert_array_equal(out, time)

    def test_conversion_preserves_intervals(self):
        """Spacing must survive: a shifted clock must not stretch the data."""
        time = 2458600.5 + np.arange(50) * 0.1
        out = timeframe.to_bjd_tdb(time, "HJD_UTC", RA, DEC)

        np.testing.assert_allclose(np.diff(out), np.diff(time), atol=1e-6)

    def test_conversion_moves_times_by_the_measured_offset(self):
        time = np.array([EPOCH])
        out = timeframe.to_bjd_tdb(time, "HJD_UTC", RA, DEC)
        shift_seconds = (out[0] - EPOCH) * 86400.0

        expected = timeframe.measure_frame_offset("HJD_UTC", RA, DEC, EPOCH)
        assert shift_seconds == pytest.approx(expected.total_seconds, abs=0.5)

    def test_mjd_input_is_converted_to_jd(self):
        mjd = np.array([58600.0])
        out = timeframe.to_bjd_tdb(mjd, "MJD_UTC", RA, DEC)
        assert out[0] == pytest.approx(58600.0 + 2400000.5, abs=0.01)

    def test_empty_array_is_handled(self):
        out = timeframe.to_bjd_tdb(np.array([]), "HJD_UTC", RA, DEC)
        assert len(out) == 0

    def test_unsupported_system_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported"):
            timeframe.to_bjd_tdb(np.array([1.0]), "GALACTIC", RA, DEC)

    def test_precision_survives_conversion(self):
        """float64 throughout: a 68 s shift must not lose the 2-min cadence."""
        time = 2458600.5 + np.arange(100) * (2.0 / 1440.0)
        out = timeframe.to_bjd_tdb(time, "HJD_UTC", RA, DEC)

        cadence_seconds = float(np.median(np.diff(out))) * 86400.0
        assert cadence_seconds == pytest.approx(120.0, abs=0.01)


class TestAlign:
    def test_align_sets_the_target_system(self):
        aligned = timeframe.align(make_curve([EPOCH, EPOCH + 1], "HJD_UTC"))
        assert aligned.time_system == "BJD_TDB"

    def test_already_aligned_curve_is_returned_unchanged(self):
        curve = make_curve([EPOCH], "BJD_TDB", survey="TESS")
        assert timeframe.align(curve) is curve

    def test_align_preserves_values(self):
        curve = make_curve([EPOCH, EPOCH + 1], "HJD_UTC")
        aligned = timeframe.align(curve)
        np.testing.assert_array_equal(aligned.value, curve.value)

    def test_empty_curve_aligns_without_error(self):
        assert len(timeframe.align(make_curve([], "HJD_UTC"))) == 0


class TestOverlap:
    def test_overlapping_curves_report_their_intersection(self):
        first = make_curve([EPOCH, EPOCH + 10], "BJD_TDB", "TESS")
        second = make_curve([EPOCH + 5, EPOCH + 20], "BJD_TDB", "TESS")
        assert timeframe.overlap_days(first, second) == pytest.approx(5.0)

    def test_disjoint_curves_report_zero(self):
        first = make_curve([EPOCH, EPOCH + 1], "BJD_TDB", "TESS")
        second = make_curve([EPOCH + 100, EPOCH + 101], "BJD_TDB", "TESS")
        assert timeframe.overlap_days(first, second) == 0.0

    def test_overlap_uses_aligned_times(self):
        """Comparing raw HJD against BJD would be off by the frame offset."""
        ztf = make_curve([EPOCH, EPOCH + 10], "HJD_UTC")
        tess = make_curve([EPOCH + 5, EPOCH + 20], "BJD_TDB", "TESS")
        assert timeframe.overlap_days(ztf, tess) > 4.9
