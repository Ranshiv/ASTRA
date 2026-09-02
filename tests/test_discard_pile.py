"""discard_pile.py: coherent discarded-epoch runs from real ZTF catflags
(Direction 2, "anomalies in the discard pile")."""

from __future__ import annotations

import numpy as np
import pytest

from astra import discard_pile as dp
from astra.surveys.base import LightCurve, SourceRef

SOURCE = SourceRef(survey="ZTF", object_id="1", ra_deg=0.0, dec_deg=0.0)


def _curve(n=60, value=None, time=None, band="g"):
    n = len(value) if value is not None else n
    time = np.arange(n, dtype=float) if time is None else time
    value = np.full(n, 18.0, dtype=float) if value is None else value
    return LightCurve(
        source=SOURCE, release="dr24", band=band, value_kind="mag",
        time=time, value=value, value_err=np.full(n, 0.02), time_system="HJD_UTC",
    )


class TestExtractDiscardRecords:
    def test_a_coherent_fade_in_a_flagged_run_is_recorded(self):
        n = 60
        value = np.full(n, 18.0)
        value[20:25] = np.array([18.4, 18.5, 18.55, 18.5, 18.4])  # smooth excursion
        catflags = np.zeros(n, dtype=np.uint32)
        catflags[20:25] = 32768

        records = dp.extract_discard_records(_curve(value=value), catflags, min_run_length=3)

        assert len(records) == 1
        record = records[0]
        assert record.flag_category == "flagged"
        assert record.epoch_count == 5
        assert record.coherent is True
        assert record.magnitude_offset > 0  # fainter than baseline

    def test_noisy_scatter_in_a_flagged_run_is_not_coherent(self):
        n = 60
        value = np.full(n, 18.0)
        rng = np.random.default_rng(0)
        value[20:25] = 18.0 + rng.uniform(-2.0, 2.0, size=5)  # large, jittery steps
        catflags = np.zeros(n, dtype=np.uint32)
        catflags[20:25] = 32768

        records = dp.extract_discard_records(_curve(value=value), catflags, min_run_length=3)

        assert len(records) == 1
        assert records[0].coherent is False

    def test_a_run_shorter_than_min_run_length_is_ignored(self):
        n = 60
        catflags = np.zeros(n, dtype=np.uint32)
        catflags[20:21] = 32768

        records = dp.extract_discard_records(_curve(n=n), catflags, min_run_length=3)

        assert records == []

    def test_a_fully_clean_curve_yields_no_records(self):
        n = 60
        catflags = np.zeros(n, dtype=np.uint32)

        records = dp.extract_discard_records(_curve(n=n), catflags, min_run_length=3)

        assert records == []

    def test_a_curve_with_no_accepted_epochs_yields_no_records(self):
        n = 20
        catflags = np.full(n, 32768, dtype=np.uint32)

        records = dp.extract_discard_records(_curve(n=n), catflags, min_run_length=3)

        assert records == []

    def test_mismatched_catflags_length_yields_no_records(self):
        catflags = np.zeros(30, dtype=np.uint32)

        records = dp.extract_discard_records(_curve(n=60), catflags, min_run_length=3)

        assert records == []

    def test_empty_curve_yields_no_records(self):
        catflags = np.zeros(0, dtype=np.uint32)

        records = dp.extract_discard_records(_curve(n=0), catflags, min_run_length=3)

        assert records == []

    def test_two_separate_flagged_runs_yield_two_records(self):
        n = 80
        catflags = np.zeros(n, dtype=np.uint32)
        catflags[10:14] = 32768
        catflags[50:55] = 32768
        value = np.full(n, 18.0)
        value[10:14] = [18.3, 18.35, 18.3, 18.25]
        value[50:55] = [17.7, 17.65, 17.6, 17.65, 17.7]

        records = dp.extract_discard_records(_curve(value=value), catflags,
                                              min_run_length=3)

        assert len(records) == 2
        assert {r.epoch_count for r in records} == {4, 5}

    def test_to_dict_round_trips_the_key_fields(self):
        n = 60
        catflags = np.zeros(n, dtype=np.uint32)
        catflags[20:25] = 32768
        value = np.full(n, 18.0)
        value[20:25] = [18.4, 18.5, 18.55, 18.5, 18.4]

        record = dp.extract_discard_records(_curve(value=value), catflags,
                                            min_run_length=3)[0]
        payload = record.to_dict()

        assert payload["object_id"] == "1"
        assert payload["survey"] == "ZTF"
        assert payload["band"] == "g"
        assert payload["flag_category"] == "flagged"
        assert payload["coherent"] is True


class _FakeConnector:
    """Duck-types `fetch_light_curves_with_quality` -- no network, mirrors
    `test_ztf_artifact_patches.py`'s fake-connector convention."""

    def __init__(self, flagged_slice=None, fail_object_id=None, n_points=60):
        self.flagged_slice = flagged_slice
        self.fail_object_id = fail_object_id
        self.n_points = n_points

    def fetch_light_curves_with_quality(self, source: SourceRef):
        if source.object_id == self.fail_object_id:
            raise RuntimeError("simulated fetch failure")
        n = self.n_points
        time = np.arange(n, dtype=float)
        value = np.full(n, 18.0)
        catflags = np.zeros(n, dtype=np.uint32)
        if self.flagged_slice is not None:
            value[self.flagged_slice] = 18.5
            catflags[self.flagged_slice] = 32768
        curve = LightCurve(
            source=source, release="dr24", band="g", value_kind="mag",
            time=time, value=value, value_err=np.full(n, 0.02), time_system="HJD_UTC",
        )
        return [(curve, catflags)]


class TestScanSource:
    def test_recovers_a_discard_record_via_the_connector(self):
        connector = _FakeConnector(flagged_slice=slice(20, 25))

        records = dp.scan_source(connector, SOURCE, min_run_length=3)

        assert len(records) == 1
        assert records[0].object_id == SOURCE.object_id

    def test_a_failing_source_does_not_abort_the_batch(self):
        sources = [
            SourceRef(survey="ZTF", object_id="bad", ra_deg=0.0, dec_deg=0.0),
            SourceRef(survey="ZTF", object_id="good", ra_deg=0.0, dec_deg=0.0),
        ]
        connector = _FakeConnector(flagged_slice=slice(20, 25), fail_object_id="bad")

        records = dp.scan_sources(connector, sources, min_run_length=3)

        assert len(records) == 1
        assert records[0].object_id == "good"

    def test_no_sources_returns_an_empty_list(self):
        assert dp.scan_sources(_FakeConnector(), []) == []
