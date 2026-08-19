"""Ablation studies and the section 20 experiment groups."""

from __future__ import annotations

import numpy as np
import pytest

from astra import ablation
from astra.ablation import AblationRow, AblationStudy


class TestGroupDefinitions:
    def test_all_seven_plan_section_20_groups_exist(self):
        assert set(ablation.SURVEY_GROUPS) == {
            "ztf_only", "gaia_only", "tess_only",
            "ztf_gaia", "ztf_tess", "gaia_tess", "all_three",
        }

    def test_single_survey_groups_have_one_member(self):
        for name in ("ztf_only", "gaia_only", "tess_only"):
            assert len(ablation.SURVEY_GROUPS[name]) == 1

    def test_all_three_has_three_members(self):
        assert len(ablation.SURVEY_GROUPS["all_three"]) == 3

    def test_feature_groups_cover_the_real_feature_names(self):
        from astra.features import FEATURE_NAMES

        grouped = {name for group in ablation.FEATURE_GROUPS.values()
                   for name in group}
        assert grouped == set(FEATURE_NAMES)


class TestStudyArithmetic:
    def test_best_picks_the_highest_auc(self):
        study = AblationStudy("x", [AblationRow("a", 0.6), AblationRow("b", 0.9)])
        assert study.best().name == "b"

    def test_best_ignores_incomparable_rows(self):
        study = AblationStudy("x", [
            AblationRow("a", 0.6),
            AblationRow("b", 0.99, comparable=False),
        ])
        assert study.best().name == "a"

    def test_best_ignores_unscored_rows(self):
        study = AblationStudy("x", [AblationRow("a", 0.6), AblationRow("b", None)])
        assert study.best().name == "a"

    def test_deltas_are_relative_to_the_baseline(self):
        study = AblationStudy("x", [
            AblationRow("base", 0.70),
            AblationRow("variant", 0.62),
        ], baseline="base")

        assert study.deltas()["variant"] == pytest.approx(-0.08)

    def test_deltas_are_empty_without_a_baseline(self):
        study = AblationStudy("x", [AblationRow("a", 0.6)])
        assert study.deltas() == {}

    def test_no_scored_rows_yields_no_best(self):
        study = AblationStudy("x", [AblationRow("a", None)])
        assert study.best() is None

    def test_study_serialises(self):
        study = AblationStudy("x", [AblationRow("base", 0.7),
                                    AblationRow("v", 0.6)], baseline="base")
        payload = study.to_dict()

        assert payload["best"] == "base"
        assert payload["deltas_vs_baseline"]["v"] == pytest.approx(-0.1)


class TestRepeatedAblation:
    def test_aggregate_reports_seed_spread_without_treating_missing_as_zero(self):
        studies = [
            AblationStudy("x", [
                AblationRow("baseline", 0.70, 0.55),
                AblationRow("variant", None, note="no usable rows"),
            ]),
            AblationStudy("x", [
                AblationRow("baseline", 0.90, 0.75),
                AblationRow("variant", 0.80, 0.60),
            ]),
        ]

        payload = ablation.aggregate_repeated(studies)
        rows = {row["name"]: row for row in payload["rows"]}

        assert rows["baseline"]["scored_runs"] == 2
        assert rows["baseline"]["roc_auc"]["mean"] == pytest.approx(0.8)
        assert rows["variant"]["runs"] == 2
        assert rows["variant"]["scored_runs"] == 1
        assert rows["variant"]["roc_auc"]["mean"] == pytest.approx(0.8)
        assert rows["variant"]["unscored_notes"] == ["no usable rows"]
        assert "independent seeds" in payload["interval"]

    def test_run_repeated_requires_two_distinct_seeds(self):
        with pytest.raises(ValueError, match="two distinct seeds"):
            ablation.run_repeated(seeds=(42,))
        with pytest.raises(ValueError, match="two distinct seeds"):
            ablation.run_repeated(seeds=(42, 42))

    def test_run_repeated_persists_aggregate_results(self, isolated_root, monkeypatch):
        def fake_study(kind):
            def build(_fraction, seed, survey=None):
                return AblationStudy(kind, [
                    AblationRow("baseline", 0.60 + seed / 1000, 0.50),
                    AblationRow("variant", 0.55 + seed / 1000, 0.45),
                ], baseline="baseline")
            return build

        monkeypatch.setattr(ablation, "survey_ablation", fake_study("survey"))
        monkeypatch.setattr(ablation, "feature_ablation", fake_study("feature"))
        monkeypatch.setattr(ablation, "detector_ablation", fake_study("detector"))

        result = ablation.run_repeated(fraction=0.2, seeds=(3, 7))

        assert result["seeds"] == [3, 7]
        assert result["survey_groups"]["rows"][0]["scored_runs"] == 2
        assert result["feature_groups"]["rows"][0]["roc_auc"]["mean"] == pytest.approx(0.605)
        assert (isolated_root.projects / "experiments" / f"{result['experiment_id']}.json").exists()


class TestSurveyAblationGuards:
    def test_missing_survey_data_is_reported_not_scored(self, isolated_root):
        """An empty store must produce honest 'no data' rows, not zeros.

        Gaia-containing groups get a distinct, more precise note than
        "no stored data": Gaia joins as columns rather than contributing
        rows (see ablation._injected_matrix / featurematrix.join_gaia_columns),
        so an empty store reports "no stored Gaia catalogue data to join"
        for it specifically, not a generic missing-survey message.
        """
        study = ablation.survey_ablation()

        assert len(study.rows) == len(ablation.SURVEY_GROUPS)
        assert all(row.comparable is False for row in study.rows)
        assert all(row.note for row in study.rows)
        for row in study.rows:
            surveys = ablation.SURVEY_GROUPS[row.name]
            if "gaia" in surveys:
                assert "gaia" in row.note.lower()
            else:
                assert "no stored data" in row.note

    def test_mismatched_survey_sizes_are_refused(self, isolated_root, monkeypatch):
        """ZTF-only vs ZTF+TESS is meaningless if TESS has 3 of 400 objects:
        the 'improvement' would just be a different population."""
        from astra import featurematrix
        from astra.featurematrix import FeatureMatrix
        from astra.features import FEATURE_NAMES

        def fake_build(survey=None, **_kwargs):
            counts = {"ztf": 400, "tess": 3, "gaia": 0}
            rows = counts.get((survey or "").lower(), 0)
            rng = np.random.default_rng(0)
            return FeatureMatrix(
                values=rng.normal(size=(rows, len(FEATURE_NAMES))),
                identities=[{"object_id": str(i), "survey": survey or "?",
                             "band": "g", "path": f"p{i}"} for i in range(rows)],
            )

        monkeypatch.setattr(featurematrix, "build", fake_build)

        study = ablation.survey_ablation()
        row = next(r for r in study.rows if r.name == "ztf_tess")

        assert row.comparable is False
        assert "differ too much" in row.note

    def test_comparable_sizes_are_scored(self, isolated_root, monkeypatch):
        from astra import featurematrix, tensors
        from astra.featurematrix import FeatureMatrix
        from astra.features import FEATURE_NAMES
        from astra.tensors import SequenceBatch

        counts = {"ztf": 100, "tess": 90, "gaia": 0}

        def fake_feature_build(survey=None, **_kwargs):
            rows = counts.get((survey or "").lower(), 0)
            rng = np.random.default_rng(0)
            return FeatureMatrix(
                values=rng.normal(size=(rows, len(FEATURE_NAMES))),
                identities=[{"object_id": str(i), "survey": survey or "?",
                             "band": "g", "path": f"{survey}{i}"}
                            for i in range(rows)],
            )

        def fake_sequence_build(survey=None, **_kwargs):
            rows = counts.get((survey or "").lower(), 0)
            rng = np.random.default_rng(1)
            grid = np.linspace(0, 4 * np.pi, 64)
            values = np.stack([
                np.sin(grid * rng.uniform(0.5, 2.0)) + rng.normal(0, 0.05, 64)
                for _ in range(rows)
            ]).astype(np.float32) if rows else np.empty((0, 64), dtype=np.float32)
            return SequenceBatch(
                values=np.stack([values, np.ones_like(values)], axis=1),
                identities=[{"object_id": str(i), "survey": survey or "?",
                             "band": "g", "path": f"{survey}{i}"}
                            for i in range(rows)],
                length=64,
            )

        monkeypatch.setattr(featurematrix, "build", fake_feature_build)
        monkeypatch.setattr(tensors, "build", fake_sequence_build)

        study = ablation.survey_ablation(fraction=0.1)
        row = next(r for r in study.rows if r.name == "ztf_tess")

        assert row.comparable is True
        assert row.rows_scored > 0
        # Injected anomalies must actually be recoverable, unlike the random
        # labels an earlier version of this study used.
        assert row.roc_auc is not None and row.roc_auc > 0.7


class TestDetectorAblation:
    def test_ensemble_is_compared_against_its_members(self, isolated_root,
                                                      monkeypatch):
        from astra import tensors
        from astra.tensors import SequenceBatch

        rng = np.random.default_rng(3)
        grid = np.linspace(0, 4 * np.pi, 64)
        values = np.stack([
            np.sin(grid * rng.uniform(0.5, 2.0)) + rng.normal(0, 0.05, 64)
            for _ in range(120)
        ]).astype(np.float32)
        batch = SequenceBatch(
            values=np.stack([values, np.ones_like(values)], axis=1),
            identities=[{"object_id": str(i), "survey": "ZTF", "band": "g",
                         "path": f"p{i}"} for i in range(120)],
            length=64,
        )
        monkeypatch.setattr(tensors, "build", lambda **_kwargs: batch)

        study = ablation.detector_ablation(fraction=0.1)
        names = {row.name for row in study.rows}

        assert "ensemble" in names
        assert study.baseline == "ensemble"
        assert len(names) >= 2

    def test_insufficient_data_is_reported(self, isolated_root):
        study = ablation.detector_ablation()
        assert study.rows[0].note


class TestFeatureAblation:
    def test_insufficient_data_is_reported(self, isolated_root):
        study = ablation.feature_ablation()
        assert study.rows[0].note

    def test_families_are_dropped_one_at_a_time(self, isolated_root, monkeypatch):
        from astra import tensors
        from astra.tensors import SequenceBatch

        rng = np.random.default_rng(4)
        grid = np.linspace(0, 4 * np.pi, 64)
        values = np.stack([
            np.sin(grid * rng.uniform(0.5, 2.0)) + rng.normal(0, 0.05, 64)
            for _ in range(120)
        ]).astype(np.float32)
        batch = SequenceBatch(
            values=np.stack([values, np.ones_like(values)], axis=1),
            identities=[{"object_id": str(i), "survey": "ZTF", "band": "g",
                         "path": f"p{i}"} for i in range(120)],
            length=64,
        )
        monkeypatch.setattr(tensors, "build", lambda **_kwargs: batch)

        study = ablation.feature_ablation(fraction=0.1)
        names = {row.name for row in study.rows}

        assert "all_features" in names
        assert any(name.startswith("without_") for name in names)


class TestGaiaColumnJoinInAblation:
    """End-to-end: Gaia catalogue rows join onto ZTF sequences as columns,
    which is what makes ztf_gaia scorable at all (see docs/DEFERRED.txt's
    'Section 20's survey groups cannot be run yet' entry -- this closes the
    ztf_gaia half of that gap; gaia_tess/all_three still need more real TESS
    coverage, which is a separate, later piece of work)."""

    N_OBJECTS = 25

    def _seed_matched_ztf_and_gaia(self, isolated_root):
        from astra import metadata, store
        from astra.surveys.base import LightCurve, SourceRef

        rng = np.random.default_rng(11)
        gaia_rows = []
        for i in range(self.N_OBJECTS):
            ra = 180.0 + i * 0.01  # 36 arcsec apart: no cross-object bleed
            source = SourceRef(survey="ZTF", object_id=f"obj{i}",
                               ra_deg=ra, dec_deg=22.0)
            curve = LightCurve(
                source=source, release="dr24", band="g", value_kind="mag",
                time=2458000.0 + np.arange(20, dtype=np.float64) * 0.5,
                value=18.0 + rng.normal(0, 0.05, 20),
                value_err=np.full(20, 0.02))
            store.write_curve(curve)
            gaia_rows.append({
                "source_key": f"Gaia/dr3/{i}", "survey": "Gaia", "release": "dr3",
                "object_id": str(i), "ra_deg": ra, "dec_deg": 22.0,
                "extra": {"parallax": 5.0, "parallax_error": 0.1,
                         "pmra": 1.0, "pmdec": -1.0,
                         "phot_g_mean_mag": 15.0, "phot_bp_mean_mag": 15.3,
                         "phot_rp_mean_mag": 14.6},
            })
        metadata.upsert_sources(isolated_root.projects, gaia_rows)

    def test_ztf_gaia_becomes_scorable_with_matched_data(self, isolated_root):
        self._seed_matched_ztf_and_gaia(isolated_root)

        study = ablation.survey_ablation(fraction=0.2, seed=7)
        ztf_only = next(r for r in study.rows if r.name == "ztf_only")
        ztf_gaia = next(r for r in study.rows if r.name == "ztf_gaia")

        assert ztf_only.comparable is True
        # Before the Gaia join this group was always refused ("no stored
        # data for gaia"), regardless of how much Gaia catalogue data
        # existed, because Gaia contributes no rows to a sequence study.
        assert ztf_gaia.comparable is True
        assert ztf_gaia.roc_auc is not None
        # Population control: joining columns must not change which or how
        # many objects are being scored relative to the ungaia'd baseline.
        assert ztf_gaia.rows_scored == ztf_only.rows_scored
        assert "gaia join matched" in ztf_gaia.note
        assert f"{self.N_OBJECTS}/{self.N_OBJECTS}" in ztf_gaia.note

    def test_gaia_only_stays_structurally_unscorable(self, isolated_root):
        """Gaia has no light curves at all, so column-joining fixes ztf_gaia
        but cannot manufacture a sequence for Gaia to stand alone on."""
        self._seed_matched_ztf_and_gaia(isolated_root)

        study = ablation.survey_ablation(fraction=0.2, seed=7)
        gaia_only = next(r for r in study.rows if r.name == "gaia_only")

        assert gaia_only.comparable is False
        assert "catalogue connector" in gaia_only.note


class TestSurveyStratification:
    """Mixing ZTF and TESS inside a detector ablation measures the mixture.

    ZTF curves are hundreds of magnitudes over years; TESS curves are tens of
    thousands of flux points at two-minute cadence. Pooled, the detectors
    partly separate by survey rather than by behaviour -- the survey bias plan
    section 36 warns about, reproduced inside the study meant to measure
    something else.
    """

    def _batch(self, monkeypatch, survey_of):
        from astra import tensors
        from astra.tensors import SequenceBatch

        rng = np.random.default_rng(11)
        grid = np.linspace(0, 4 * np.pi, 64)
        values = np.stack([
            np.sin(grid * rng.uniform(0.5, 2.0)) + rng.normal(0, 0.05, 64)
            for _ in range(120)
        ]).astype(np.float32)
        identities = [{"object_id": str(i), "survey": survey_of(i),
                       "band": "g", "path": f"p{i}"} for i in range(120)]

        seen: dict = {}

        def build(survey=None, **_kwargs):
            seen["survey"] = survey
            keep = [i for i, ident in enumerate(identities)
                    if survey is None
                    or ident["survey"].upper() == survey.upper()]
            return SequenceBatch(
                values=np.stack([values[keep],
                                 np.ones_like(values[keep])], axis=1),
                identities=[identities[i] for i in keep], length=64)

        monkeypatch.setattr(tensors, "build", build)
        return seen

    def test_detector_ablation_passes_the_survey_filter_through(
            self, isolated_root, monkeypatch):
        seen = self._batch(monkeypatch, lambda i: "ZTF")
        ablation.detector_ablation(fraction=0.1, survey="ztf")
        assert seen["survey"] == "ztf"

    def test_feature_ablation_passes_the_survey_filter_through(
            self, isolated_root, monkeypatch):
        seen = self._batch(monkeypatch, lambda i: "ZTF")
        ablation.feature_ablation(fraction=0.1, survey="ztf")
        assert seen["survey"] == "ztf"

    def test_default_is_unstratified_so_existing_behaviour_is_unchanged(
            self, isolated_root, monkeypatch):
        seen = self._batch(monkeypatch, lambda i: "ZTF")
        ablation.detector_ablation(fraction=0.1)
        assert seen["survey"] is None

    def test_stratifying_actually_narrows_the_population(
            self, isolated_root, monkeypatch):
        """The guard that matters: filtering must reach tensors.build, not be
        accepted and quietly dropped."""
        self._batch(monkeypatch, lambda i: "ZTF" if i < 60 else "TESS")

        mixed = ablation.detector_ablation(fraction=0.1)
        ztf_only = ablation.detector_ablation(fraction=0.1, survey="ztf")

        mixed_rows = {row.name: row.rows_scored for row in mixed.rows}
        ztf_rows = {row.name: row.rows_scored for row in ztf_only.rows}
        assert mixed_rows["ensemble"] > ztf_rows["ensemble"]

    def test_run_all_records_the_survey_in_the_experiment(
            self, isolated_root, monkeypatch):
        """Provenance, not decoration: a stratified run and an older
        unstratified one must stay distinguishable."""
        from astra import experiment

        def fake(_fraction, _seed, survey=None, **_kwargs):
            return AblationStudy("x", [AblationRow("baseline", 0.7, 0.5)],
                                 baseline="baseline")

        monkeypatch.setattr(ablation, "survey_ablation",
                            lambda _f, _s: AblationStudy(
                                "survey", [AblationRow("g", 0.7, 0.5)]))
        monkeypatch.setattr(ablation, "feature_ablation", fake)
        monkeypatch.setattr(ablation, "detector_ablation", fake)

        result = ablation.run_all(fraction=0.2, seed=5, survey="ztf")
        record = experiment.load(result["experiment_id"])

        assert record.configuration["survey"] == "ztf"
        assert "ztf" in record.notes
