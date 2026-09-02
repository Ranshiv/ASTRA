"""discard_corroboration.py: cross-survey support for discard-pile events
(Direction 2, step 2)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import discard_corroboration as dc
from astra.discard_pile import DiscardRecord
from astra.surveys.base import LightCurve, SourceRef


def _record(time_start=20.0, time_end=24.0, survey="ZTF"):
    return DiscardRecord(
        object_id="1", survey=survey, band="g", flag_category="flagged",
        epoch_count=5, time_start=time_start, time_end=time_end,
        magnitude_offset=0.4, max_step=0.05, coherent=True,
    )


def _curve(survey, time, value, err=0.02, band="g"):
    return LightCurve(
        source=SourceRef(survey=survey, object_id="1", ra_deg=0.0, dec_deg=0.0),
        release="dr1", band=band, value_kind="mag",
        time=np.asarray(time, dtype=float), value=np.asarray(value, dtype=float),
        value_err=np.full(len(time), err), time_system="JD_UTC",
    )


class TestCorroborate:
    def test_an_independent_survey_with_a_coincident_deviation_supports(self):
        record = _record(time_start=20.0, time_end=24.0)
        time = np.arange(60, dtype=float)
        value = np.full(60, 15.0)
        value[20:25] = 15.5  # same window, a real coincident deviation
        other = _curve("Gaia", time, value)

        result = dc.corroborate(record, [other])

        assert result.corroborated is True
        assert "Gaia" in result.to_dict()["supporting_surveys"]

    def test_a_flat_independent_curve_does_not_support(self):
        record = _record()
        time = np.arange(60, dtype=float)
        value = np.full(60, 15.0) + np.random.default_rng(0).normal(0, 0.001, 60)
        other = _curve("Gaia", time, value)

        result = dc.corroborate(record, [other])

        assert result.corroborated is False
        assert result.components[0].supports is False

    def test_no_counterpart_coverage_yields_no_components_and_no_corroboration(self):
        record = _record(time_start=20.0, time_end=24.0)
        # Counterpart curve only observed long before the discard window.
        other = _curve("Gaia", np.arange(0, 5, dtype=float), np.full(5, 15.0))

        result = dc.corroborate(record, [other])

        assert result.corroborated is False
        assert result.components[0].in_window_points == 0
        assert result.components[0].z_score is None

    def test_no_other_curves_yields_no_coverage_reason(self):
        result = dc.corroborate(_record(), [])

        assert result.corroborated is False
        assert result.components == ()
        assert "no independent-survey coverage" in result.reasons[0]

    def test_multiple_surveys_all_supporting_are_all_listed(self):
        record = _record()
        time = np.arange(60, dtype=float)
        value = np.full(60, 15.0)
        value[20:25] = 15.5
        gaia = _curve("Gaia", time, value)
        tess = _curve("TESS", time, value, band="TESS")

        result = dc.corroborate(record, [gaia, tess])

        assert result.corroborated is True
        assert set(result.to_dict()["supporting_surveys"]) == {"Gaia", "TESS"}

    def test_min_supporting_surveys_requires_more_than_one(self):
        record = _record()
        time = np.arange(60, dtype=float)
        value = np.full(60, 15.0)
        value[20:25] = 15.5
        gaia = _curve("Gaia", time, value)

        result = dc.corroborate(record, [gaia], min_supporting_surveys=2)

        assert result.corroborated is False

    def test_to_dict_includes_the_underlying_record(self):
        result = dc.corroborate(_record(), [])
        payload = result.to_dict()
        assert payload["record"]["object_id"] == "1"
