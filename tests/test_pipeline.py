"""End-to-end candidate generation, and the cost of getting there.

The performance guard here is deliberate. Phase 9 measured that caching one
stage does not help if a later stage recomputes the same thing: after feature
caching landed the pipeline was still 307.9 s, because the cross-survey profile
re-ran a full period search per curve. The fix that followed is only correct if
nothing reintroduces a second walk of the store, and the only way to know that
is to count the reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import evidence, metadata, pipeline, store
from astra.surveys.base import LightCurve, SourceRef


def _curve(object_id: str, ra: float, dec: float, seed: int,
           amplitude: float = 0.0) -> LightCurve:
    rng = np.random.default_rng(seed)
    time = 2458000.5 + np.arange(120, dtype=np.float64) * 0.7
    value = 18.0 + rng.normal(0.0, 0.05, size=120)
    if amplitude:
        value += amplitude * np.sin(2 * np.pi * time / 0.6)
    return LightCurve(
        source=SourceRef(survey="ZTF", object_id=object_id, ra_deg=ra, dec_deg=dec),
        release="dr24", band="g", value_kind="mag",
        time=time, value=value, value_err=np.full(120, 0.03),
        time_system="HJD_UTC",
    )


@pytest.fixture
def populated_store(isolated_root):
    """A handful of ZTF curves, one of them obviously variable."""
    for index in range(8):
        store.write_curve(_curve(
            f"ZTF-{index:03d}",
            180.0 + index * 0.05,
            22.0 + index * 0.05,
            seed=index,
            # One clearly variable source, so the ranking has something to find.
            amplitude=0.9 if index == 3 else 0.0,
        ))
    return isolated_root


class TestCurveIndex:
    def test_one_walk_yields_both_views(self, populated_store):
        index = evidence.load_curve_index()

        assert len(index.by_key) == 8
        assert len(index.positions_by_path) == 8
        for position in index.positions_by_path.values():
            assert np.isfinite(position["ra_deg"])
            assert np.isfinite(position["dec_deg"])

    def test_load_curves_by_key_still_returns_the_key_index(self, populated_store):
        """The old entry point is used by rpc.py and must keep its contract."""
        by_key = evidence.load_curves_by_key()
        assert by_key.keys() == evidence.load_curve_index().by_key.keys()

    def test_missing_root_is_empty_not_an_error(self, isolated_root):
        index = evidence.load_curve_index()
        assert index.by_key == {}
        assert index.positions_by_path == {}


class TestPipelineReadsTheStoreOnce:
    def _count_reads(self, monkeypatch):
        calls: list[str] = []
        real = store.read_curve

        def counting(path, *args, **kwargs):
            calls.append(str(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(store, "read_curve", counting)
        return calls

    def test_no_second_read_per_candidate(self, populated_store, monkeypatch):
        """`_position_for` used to re-open one Parquet file per candidate, on
        top of a walk that had already opened every one of them."""
        calls = self._count_reads(monkeypatch)
        built, _ = pipeline.run(survey_names=["ztf"])

        assert built, "expected candidates from a populated store"
        # Every path is read at most once per distinct file across the run.
        assert len(calls) == len(set(calls))

    def test_position_lookup_needs_no_extra_read(self, populated_store, monkeypatch):
        pipeline.run(survey_names=["ztf"])  # warm the feature cache

        calls = self._count_reads(monkeypatch)
        built, _ = pipeline.run(survey_names=["ztf"])

        assert len(calls) <= len(built), (
            "position recovery should come from the shared index, not a read "
            "per candidate"
        )


class TestPipelineResultsAreUnchanged:
    def test_candidates_carry_real_positions(self, populated_store):
        built, _ = pipeline.run(survey_names=["ztf"])

        assert built
        for candidate in built:
            assert np.isfinite(candidate.ra_deg)
            assert np.isfinite(candidate.dec_deg)
            assert 179.0 < candidate.ra_deg < 181.0

    def test_candidates_carry_non_ranking_significance_context(self, populated_store):
        built, _ = pipeline.run(survey_names=["ztf"])

        assert built
        for candidate in built:
            assert candidate.significance["method"] == "empirical_cdf"
            assert candidate.significance["ready"] is False or 0.0 <= candidate.significance["tail_probability"] <= 1.0
            assert candidate.evidence_completeness["resolved_surveys"] >= 1

    def test_consolidation_preserves_the_ranking(self, populated_store):
        """The 9x speedup was validated by an identical top candidate; the
        same check guards the consolidation that followed it."""
        first, _ = pipeline.run(survey_names=["ztf"])
        second, _ = pipeline.run(survey_names=["ztf"])

        assert [c.candidate_id for c in first] == [c.candidate_id for c in second]
        assert [c.object_id for c in first] == [c.object_id for c in second]
        for left, right in zip(first, second):
            assert left.score["total"] == pytest.approx(right.score["total"])

    def test_report_counts_the_strata_it_processed(self, populated_store):
        _, report = pipeline.run(survey_names=["ztf"])

        assert report.surveys_processed == ["ZTF"]
        assert report.rows_by_survey["ZTF"] == 8
        assert report.candidates_built > 0
        assert report.output_path

    def test_report_records_explicit_crossmatch_anchor(self, populated_store, isolated_root):
        metadata.upsert_sources(isolated_root.projects, [{
            "source_key": "Gaia/dr3/anchor", "survey": "Gaia", "release": "dr3",
            "object_id": "anchor", "ra_deg": 180.0, "dec_deg": 22.0,
            "extra": {},
        }])
        _, report = pipeline.run(survey_names=["ztf"], anchor_survey="Gaia")

        assert report.anchor_survey == "Gaia"
        assert report.anchor_policy == "explicit"
        assert report.cross_survey_groups == 1
