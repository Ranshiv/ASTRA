"""`research.acquire`'s per-object time-split inputs and degeneracy guard.

Before this test existed, `acquire_core_corpus` fed `splits.sky_time_split`
a constant cone-centre position and a constant placeholder MJD for every
object -- every object landed in one sky/time cell, so `detect_leakage`
reported `clean: true` because there was nothing to leak between, not
because the split actually separated anything. See docs/DEFERRED.txt and
the P0 research plan.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from astra import manifest as manifest_mod
from astra import metadata as metadata_mod
from astra import store as store_mod
from astra.research import acquire as research_acquire
from astra.surveys.base import ConeQuery, LightCurve, SourceRef


def _write_object_curve(root, *, survey: str, release: str, object_id: str,
                        ra_deg: float, dec_deg: float, mjd_start: float) -> None:
    source = SourceRef(survey=survey, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg)
    time = mjd_start + 2_400_000.5 + np.arange(20, dtype=np.float64) * 0.1
    curve = LightCurve(source=source, release=release, band="g", value_kind="mag",
                       time=time, value=np.full(20, 18.0), value_err=np.full(20, 0.03),
                       time_system="JD_UTC")
    store_mod.write_curve(curve, root)


def _manifest_with(object_ids_and_positions, *, survey: str = "ZTF", release: str = "dr24"):
    m = manifest_mod.Manifest(dataset_id="test-corpus")
    query = manifest_mod.SurveyQuery(
        survey=survey, release=release, ra_deg=180.0, dec_deg=20.0,
        radius_arcsec=90.0, limit=200,
        object_ids=sorted(oid for oid, _, _ in object_ids_and_positions),
    )
    m.add(query)
    return m


def test_object_time_records_uses_real_curve_positions_and_epochs(isolated_root):
    root = isolated_root.datasets
    _write_object_curve(root, survey="ZTF", release="dr24", object_id="a",
                        ra_deg=10.0, dec_deg=5.0, mjd_start=59000.0)
    _write_object_curve(root, survey="ZTF", release="dr24", object_id="b",
                        ra_deg=210.0, dec_deg=-40.0, mjd_start=59400.0)
    m = _manifest_with([("a", 10.0, 5.0), ("b", 210.0, -40.0)])

    records, dropped = research_acquire._object_time_records(m)

    assert dropped == 0
    by_id = {r["object_id"]: r for r in records}
    assert by_id["a"]["ra_deg"] == pytest.approx(10.0)
    assert by_id["a"]["dec_deg"] == pytest.approx(5.0)
    assert by_id["a"]["mjd"] == pytest.approx(59000.0 + 19 * 0.1 / 2, abs=1.0)
    # The two objects must not collapse onto the same position/epoch.
    assert by_id["a"]["ra_deg"] != by_id["b"]["ra_deg"]
    assert by_id["a"]["mjd"] != by_id["b"]["mjd"]


def test_object_time_records_drops_objects_with_no_stored_curve(isolated_root):
    root = isolated_root.datasets
    _write_object_curve(root, survey="ZTF", release="dr24", object_id="has-curve",
                        ra_deg=10.0, dec_deg=5.0, mjd_start=59000.0)
    m = _manifest_with([("has-curve", 10.0, 5.0), ("no-curve", 10.0, 5.0)])

    records, dropped = research_acquire._object_time_records(m)

    assert dropped == 1
    assert {r["object_id"] for r in records} == {"has-curve"}


def test_assert_not_degenerate_passes_distinct_records():
    records = [
        {"object_id": "a", "ra_deg": 10.0, "dec_deg": 5.0, "mjd": 59000.0},
        {"object_id": "b", "ra_deg": 210.0, "dec_deg": -40.0, "mjd": 59400.0},
    ]
    research_acquire._assert_not_degenerate(records, dataset_id="ok")  # must not raise


def test_assert_not_degenerate_rejects_constant_input():
    """The exact failure mode the old acquire_core_corpus code produced:
    every object at the same position and epoch."""
    records = [
        {"object_id": f"obj{i}", "ra_deg": 180.122, "dec_deg": 22.411, "mjd": 59000.0}
        for i in range(5)
    ]
    with pytest.raises(ValueError, match="degenerate"):
        research_acquire._assert_not_degenerate(records, dataset_id="demo")


def test_assert_not_degenerate_allows_trivial_single_object_corpus():
    """A one-object (or empty) corpus has nothing to separate; it is not the
    same failure as N objects collapsing onto one cell."""
    research_acquire._assert_not_degenerate([], dataset_id="empty")
    research_acquire._assert_not_degenerate(
        [{"object_id": "a", "ra_deg": 1.0, "dec_deg": 1.0, "mjd": 59000.0}],
        dataset_id="one")


def test_acquired_sources_reads_real_positions_from_metadata_store(isolated_root):
    """`_acquired_sources` must use the discovery-time metadata rows (present
    even for objects whose light curve later failed to fetch), not the
    old cone-centre placeholder."""
    metadata_mod.upsert_sources(isolated_root.projects, [
        {"source_key": "k1", "survey": "ZTF", "release": "dr24",
         "object_id": "a", "ra_deg": 10.0, "dec_deg": 5.0, "extra": {}},
        {"source_key": "k2", "survey": "ZTF", "release": "dr24",
         "object_id": "not-in-manifest", "ra_deg": 99.0, "dec_deg": 1.0, "extra": {}},
    ])
    manifest = manifest_mod.Manifest(dataset_id="test")
    manifest.add(manifest_mod.SurveyQuery(survey="ZTF", release="dr24", ra_deg=10.0,
                                          dec_deg=5.0, radius_arcsec=90.0, limit=10,
                                          object_ids=["a"]))

    sources = research_acquire._acquired_sources(manifest, ["a"])

    assert len(sources) == 1
    assert sources[0].object_id == "a"
    assert sources[0].ra_deg == pytest.approx(10.0)
    assert sources[0].dec_deg == pytest.approx(5.0)


class _FakeSimbad:
    """Stands in for astroquery.simbad.Simbad without touching the network."""

    def __init__(self, table: Table):
        self._table = table

    def add_votable_fields(self, *_args, **_kwargs) -> None:
        return None

    def query_region(self, *_args, **_kwargs) -> Table:
        return self._table


def _install_fake_simbad(monkeypatch, table: Table) -> None:
    import types
    fake_module = types.SimpleNamespace(Simbad=lambda: _FakeSimbad(table))
    monkeypatch.setitem(__import__("sys").modules, "astroquery.simbad", fake_module)


def test_pull_simbad_labels_matches_by_position_not_by_field_membership(monkeypatch):
    """The old version returned every SIMBAD row in the field keyed by
    SIMBAD's own main_id -- 'known objects present in this field', never
    'this ASTRA object has this label' (docs/DATA_CARD.md). A source with no
    real nearby SIMBAD counterpart must get no label."""
    sources = [
        SourceRef(survey="ZTF", object_id="near-match", ra_deg=10.0, dec_deg=5.0),
        SourceRef(survey="ZTF", object_id="far-from-anything", ra_deg=300.0, dec_deg=-60.0),
    ]
    table = Table({
        "main_id": ["Star A"],
        "otype": ["RRLyr"],
        "ra": [10.0002],   # ~0.7 arcsec from near-match; far outside far-from-anything's radius
        "dec": [5.0001],
    })
    _install_fake_simbad(monkeypatch, table)

    records = research_acquire._pull_simbad_labels(
        sources, ConeQuery(ra_deg=10.0, dec_deg=5.0, radius_arcsec=90.0), radius_arcsec=2.0)

    assert len(records) == 1
    assert records[0].object_id == "near-match"
    assert records[0].label == "RRLyr"
    assert records[0].label_source == "SIMBAD"
    assert 0.0 < records[0].confidence <= 1.0


def test_pull_simbad_labels_confidence_drops_with_separation_and_crowding(monkeypatch):
    source = [SourceRef(survey="ZTF", object_id="obj", ra_deg=10.0, dec_deg=5.0)]

    tight_table = Table({"main_id": ["A"], "otype": ["*"], "ra": [10.0001], "dec": [5.0]})
    _install_fake_simbad(monkeypatch, tight_table)
    tight = research_acquire._pull_simbad_labels(
        source, ConeQuery(ra_deg=10.0, dec_deg=5.0, radius_arcsec=90.0), radius_arcsec=2.0)

    crowded_table = Table({"main_id": ["A", "B"], "otype": ["*", "*"],
                           "ra": [10.0001, 10.0003], "dec": [5.0, 5.0]})
    _install_fake_simbad(monkeypatch, crowded_table)
    crowded = research_acquire._pull_simbad_labels(
        source, ConeQuery(ra_deg=10.0, dec_deg=5.0, radius_arcsec=90.0), radius_arcsec=2.0)

    assert tight[0].confidence > crowded[0].confidence


def test_pull_simbad_labels_empty_sources_makes_no_query(monkeypatch):
    def _boom(*_a, **_kw):
        raise AssertionError("should not query SIMBAD with zero sources")
    import types
    monkeypatch.setitem(__import__("sys").modules, "astroquery.simbad",
                        types.SimpleNamespace(Simbad=_boom))

    records = research_acquire._pull_simbad_labels(
        [], ConeQuery(ra_deg=10.0, dec_deg=5.0, radius_arcsec=90.0))
    assert records == []
