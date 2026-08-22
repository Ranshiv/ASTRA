"""End-to-end acquisition against fake connectors — no network involved."""

from __future__ import annotations

import time as time_mod

import numpy as np
import pytest

from astra import acquire, surveys
from astra.surveys.base import LightCurve, SourceRef, SurveyConnector


class FakeConnector(SurveyConnector):
    name = "FAKE"
    release = "v1"

    def cone_search(self, query, limit=100):
        return [
            SourceRef(survey=self.name, object_id=f"obj{i}",
                      ra_deg=query.ra_deg, dec_deg=query.dec_deg)
            for i in range(min(3, limit))
        ]

    def fetch_light_curves(self, source):
        time = 2458000.5 + np.arange(50, dtype=np.float64)
        return [LightCurve(
            source=source, release=self.release, band="g", value_kind="mag",
            time=time, value=np.full(50, 18.0), value_err=np.full(50, 0.02),
            time_system="HJD_UTC",
        )]


class BrokenConnector(SurveyConnector):
    name = "BROKEN"
    release = "v1"

    def cone_search(self, query, limit=100):
        raise ConnectionError("archive unreachable")

    def fetch_light_curves(self, source):
        return []


@pytest.fixture
def registered(isolated_root):
    surveys.register("fake", FakeConnector)
    surveys.register("broken", BrokenConnector)
    yield isolated_root
    surveys._REGISTRY.pop("fake", None)
    surveys._REGISTRY.pop("broken", None)


def test_acquisition_stores_curves(registered, cone):
    result = acquire.acquire(cone, survey_names=["fake"], limit=3)

    outcome = result.outcomes[0]
    assert outcome.sources_found == 3
    assert outcome.curves_stored == 3
    assert outcome.points_stored == 150


def test_acquisition_writes_a_sealed_manifest(registered, cone):
    result = acquire.acquire(cone, survey_names=["fake"], limit=3)

    assert result.content_hash is not None
    assert result.manifest_path is not None


def test_a_failing_archive_does_not_abort_the_run(registered, cone):
    """An archive being down should degrade the dataset, not kill the job."""
    result = acquire.acquire(cone, survey_names=["broken", "fake"], limit=3)

    broken = next(o for o in result.outcomes if o.survey == "BROKEN")
    working = next(o for o in result.outcomes if o.survey == "FAKE")

    assert "unreachable" in broken.error
    assert working.curves_stored == 3


def test_unknown_survey_is_reported_not_raised(registered, cone):
    result = acquire.acquire(cone, survey_names=["nope"], limit=1)
    assert "unknown survey" in result.outcomes[0].error


def test_second_run_skips_already_fetched_objects_without_refetching(
        registered, cone, monkeypatch):
    """A resume must not pay the network cost again.

    Previously `skip_existing` was checked only after `fetch_light_curves`
    had already downloaded the object, so resuming a 5000-object campaign
    re-fetched everything and saved only the Parquet write. The cursor now
    filters before the fetch.
    """
    acquire.acquire(cone, survey_names=["fake"], limit=3)

    calls: list[str] = []
    original = FakeConnector.fetch_light_curves

    def counting(self, source):
        calls.append(source.object_id)
        return original(self, source)

    monkeypatch.setattr(FakeConnector, "fetch_light_curves", counting)
    second = acquire.acquire(cone, survey_names=["fake"], limit=3)

    outcome = second.outcomes[0]
    assert outcome.already_fetched == 3
    assert outcome.curves_stored == 0
    assert calls == [], "resumed run re-downloaded already-fetched objects"


def test_reacquisition_reproduces_the_same_content_hash(registered, cone):
    """The reproducibility guarantee: same query, same hash, no second copy."""
    first = acquire.acquire(cone, survey_names=["fake"], limit=3)
    second = acquire.acquire(cone, survey_names=["fake"], limit=3)

    assert first.content_hash == second.content_hash


def test_result_serialises_for_the_ui(registered, cone):
    payload = acquire.acquire(cone, survey_names=["fake"], limit=3).to_dict()

    assert payload["totals"]["curves"] == 3
    assert payload["query"]["radius_arcsec"] == 10.0
    assert payload["surveys"][0]["survey"] == "FAKE"


def test_dataset_id_is_derived_from_the_cone(cone):
    generated = acquire.default_dataset_id(cone)
    assert generated.startswith("cone_180.122000_22.411000_10.000")


class _RegionScopedFakeConnector(FakeConnector):
    """Like FakeConnector, but object ids are unique per sky position.

    FakeConnector's object ids ("obj0".."obj2") repeat for any query, which
    is fine for single-cone tests but would make two genuinely distinct sky
    regions collide on the same `source_key` -- a coincidence that would
    never happen against a real catalogue, where object ids are unique across
    the whole sky. Folding the query position into the id keeps this test's
    two regions realistically distinct.
    """

    name = "FAKE_REGION"

    def cone_search(self, query, limit=100):
        return [
            SourceRef(survey=self.name,
                     object_id=f"obj_{query.ra_deg:.3f}_{i}",
                     ra_deg=query.ra_deg, dec_deg=query.dec_deg)
            for i in range(min(3, limit))
        ]


def test_acquire_project_iterates_all_regions(registered):
    from astra import project

    surveys.register("fake-region", _RegionScopedFakeConnector)
    try:
        created = project.create(
            name="Mosaic",
            query_regions=[
                {"ra_deg": 180.122, "dec_deg": 22.411, "radius_arcsec": 10},
                {"ra_deg": 30.0, "dec_deg": -5.0, "radius_arcsec": 10},
            ],
        )

        result = acquire.acquire_project(created["project_id"],
                                         survey_names=["fake-region"], limit=3)
    finally:
        surveys._REGISTRY.pop("fake-region", None)

    assert len(result.regions) == 2
    assert {round(r.query.ra_deg, 3) for r in result.regions} == {180.122, 30.0}
    assert all(r.outcomes[0].curves_stored == 3 for r in result.regions)
    payload = result.to_dict()
    assert payload["totals"]["curves"] == 6


def test_acquire_project_rejects_empty_regions(registered):
    from astra import project

    created = project.create(name="No regions")
    with pytest.raises(ValueError, match="query_regions"):
        acquire.acquire_project(created["project_id"], survey_names=["fake"], limit=3)


def test_acquire_project_dedupes_overlapping_regions(registered):
    from astra import metadata, project

    created = project.create(
        name="Overlap",
        query_regions=[
            {"ra_deg": 180.122, "dec_deg": 22.411, "radius_arcsec": 10},
            {"ra_deg": 180.122, "dec_deg": 22.411, "radius_arcsec": 20},
        ],
    )

    acquire.acquire_project(created["project_id"], survey_names=["fake"], limit=3)

    from astra import config
    stored = metadata.list_sources(config.PATHS.projects)
    fake_sources = [row for row in stored if row["survey"] == "FAKE"]
    assert len(fake_sources) == 3


class FlakyConnector(SurveyConnector):
    """Fails on a fixed subset of objects, succeeds on the rest."""

    name = "FLAKY"
    release = "v1"
    failing = {"obj1", "obj3"}

    def cone_search(self, query, limit=100):
        return [
            SourceRef(survey=self.name, object_id=f"obj{i}",
                      ra_deg=query.ra_deg, dec_deg=query.dec_deg)
            for i in range(min(5, limit))
        ]

    def fetch_light_curves(self, source):
        if source.object_id in self.failing:
            raise ConnectionError(f"archive refused {source.object_id}")
        time = 2458000.5 + np.arange(50, dtype=np.float64)
        return [LightCurve(
            source=source, release=self.release, band="g", value_kind="mag",
            time=time, value=np.full(50, 18.0), value_err=np.full(50, 0.02),
            time_system="HJD_UTC",
        )]


class EmptyConnector(SurveyConnector):
    """A catalogue connector: real sources, no light curves. Like Gaia."""

    name = "CATALOGUE"
    release = "v1"

    def cone_search(self, query, limit=100):
        return [SourceRef(survey=self.name, object_id=f"cat{i}",
                          ra_deg=query.ra_deg, dec_deg=query.dec_deg)
                for i in range(min(4, limit))]

    def fetch_light_curves(self, source):
        return []


@pytest.fixture
def flaky(isolated_root):
    surveys.register("flaky", FlakyConnector)
    surveys.register("catalogue", EmptyConnector)
    yield isolated_root
    surveys._REGISTRY.pop("flaky", None)
    surveys._REGISTRY.pop("catalogue", None)


class TestFailureAccounting:
    """Silent large-scale loss was the worst campaign hazard.

    `outcome.error` is a single string overwritten on every failure, so a run
    that lost 500 objects reported one message and no count.
    """

    def test_failures_are_counted_not_just_overwritten(self, flaky, cone):
        result = acquire.acquire(cone, survey_names=["flaky"], limit=5)
        outcome = result.outcomes[0]

        assert outcome.failed_objects == 2
        assert outcome.curves_stored == 3

    def test_failures_persist_per_object_for_inspection(self, flaky, cone):
        from astra import config, metadata

        acquire.acquire(cone, survey_names=["flaky"], limit=5)
        progress = metadata.acquisition_progress(config.PATHS.projects,
                                                 survey="FLAKY")

        assert progress["failed"] == 2
        assert progress["done"] == 3
        failed_ids = {f["object_id"] for f in progress["recent_failures"]}
        assert failed_ids == {"obj1", "obj3"}

    def test_a_resumed_run_retries_only_the_failures(self, flaky, cone,
                                                     monkeypatch):
        acquire.acquire(cone, survey_names=["flaky"], limit=5)

        attempted: list[str] = []
        original = FlakyConnector.fetch_light_curves

        def counting(self, source):
            attempted.append(source.object_id)
            return original(self, source)

        monkeypatch.setattr(FlakyConnector, "fetch_light_curves", counting)
        acquire.acquire(cone, survey_names=["flaky"], limit=5)

        assert sorted(attempted) == ["obj1", "obj3"]

    def test_retry_budget_is_finite(self, flaky, cone):
        from astra import config, metadata

        for _ in range(metadata.MAX_FETCH_ATTEMPTS + 2):
            acquire.acquire(cone, survey_names=["flaky"], limit=5)

        pending = metadata.pending_sources(config.PATHS.projects,
                                           survey="FLAKY")
        assert pending == [], "exhausted failures must stop being retried"
        progress = metadata.acquisition_progress(config.PATHS.projects,
                                                 survey="FLAKY")
        assert progress["failed"] == 2

    def test_catalogue_connector_completes_rather_than_retrying(self, flaky,
                                                               cone):
        """Gaia returns no curves by design; that is done, not failed."""
        from astra import config, metadata

        acquire.acquire(cone, survey_names=["catalogue"], limit=4)
        progress = metadata.acquisition_progress(config.PATHS.projects,
                                                 survey="CATALOGUE")

        assert progress["empty"] == 4
        assert progress["failed"] == 0
        assert metadata.pending_sources(config.PATHS.projects,
                                        survey="CATALOGUE") == []

    def test_progress_reports_a_completion_fraction(self, flaky, cone):
        from astra import config, metadata

        acquire.acquire(cone, survey_names=["flaky"], limit=5)
        progress = metadata.acquisition_progress(config.PATHS.projects,
                                                 survey="FLAKY")

        assert progress["total"] == 5
        assert progress["complete_fraction"] == pytest.approx(0.6)


class TestSurveyOptions:
    """B2: acquire() must thread per-survey connector kwargs (e.g. an HLSP
    author choice for TESS) down to surveys.get(), and do nothing different
    when survey_options is omitted. See docs/DEFERRED.txt Phase 8 for why
    this plumbing was missing entirely before."""

    def test_survey_options_reach_surveys_get(self, registered, cone, monkeypatch):
        calls: list[tuple[str, dict]] = []
        real_get = surveys.get

        def recording_get(name, **kwargs):
            calls.append((name, kwargs))
            return real_get(name, **kwargs)

        monkeypatch.setattr(surveys, "get", recording_get)

        acquire.acquire(cone, survey_names=["fake"], limit=3,
                        survey_options={"fake": {}})

        assert calls == [("fake", {})]

    def test_survey_options_kwargs_are_forwarded_verbatim(self, isolated_root, cone,
                                                           monkeypatch):
        received: dict = {}

        class KwargConnector(SurveyConnector):
            name = "KWARG"
            release = "v1"

            def __init__(self, choice: str = "default"):
                received["choice"] = choice

            def cone_search(self, query, limit=100):
                return []

            def fetch_light_curves(self, source):
                return []

        surveys.register("kwarg", KwargConnector)
        try:
            acquire.acquire(cone, survey_names=["kwarg"], limit=3,
                            survey_options={"kwarg": {"choice": "QLP"}})
            assert received["choice"] == "QLP"
        finally:
            surveys._REGISTRY.pop("kwarg", None)

    def test_omitting_survey_options_is_unchanged_behaviour(self, registered, cone):
        """No survey_options at all must reproduce today's zero-kwarg call.

        skip_existing=False on both calls: with the default True, the second
        acquire() would find the first call's objects already fetched and
        report zero curves stored regardless of survey_options -- that would
        be the resumable-fetch cursor working correctly, not evidence about
        this parameter either way.
        """
        result_without = acquire.acquire(cone, survey_names=["fake"], limit=3,
                                         dataset_id="a", skip_existing=False)
        result_with_empty = acquire.acquire(cone, survey_names=["fake"], limit=3,
                                            dataset_id="b", skip_existing=False,
                                            survey_options={})

        assert [o.to_dict() for o in result_without.outcomes] == \
            [o.to_dict() for o in result_with_empty.outcomes]

    def test_survey_options_are_scoped_per_survey_name(self, isolated_root, cone):
        """An option meant for "tess" must not leak into another survey's
        connector construction."""
        received: dict[str, dict] = {}

        class RecordingConnector(SurveyConnector):
            release = "v1"

            def __init__(self, **kwargs):
                received[self.name] = kwargs

            def cone_search(self, query, limit=100):
                return []

            def fetch_light_curves(self, source):
                return []

        class ConnectorA(RecordingConnector):
            name = "A"

        class ConnectorB(RecordingConnector):
            name = "B"

        surveys.register("a", ConnectorA)
        surveys.register("b", ConnectorB)
        try:
            acquire.acquire(cone, survey_names=["a", "b"], limit=3,
                            survey_options={"a": {"author": "QLP"}})
            assert received["A"] == {"author": "QLP"}
            assert received["B"] == {}
        finally:
            surveys._REGISTRY.pop("a", None)
            surveys._REGISTRY.pop("b", None)


class SlowConnector(SurveyConnector):
    """cone_search() that blocks -- the astroquery/lightkurve shape this
    module's timeout wrapper exists to bound."""

    name = "SLOW"
    release = "v1"

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.was_called_from_a_worker_thread = False

    def cone_search(self, query, limit=100):
        import threading

        self.was_called_from_a_worker_thread = (
            threading.current_thread() is not threading.main_thread())
        time_mod.sleep(self.delay)
        return [SourceRef(survey=self.name, object_id="obj0",
                          ra_deg=query.ra_deg, dec_deg=query.dec_deg)]

    def fetch_light_curves(self, source):
        return []


class FakeProgress:
    """Minimal stand-in for jobs.JobContext: cancel after N checks, and
    record every progress message published."""

    def __init__(self, cancel_after: int | None = None):
        self.checks = 0
        self.cancel_after = cancel_after
        self.messages: list[str] = []

    def raise_if_cancelled(self) -> None:
        from astra.jobs import JobCancelled

        self.checks += 1
        if self.cancel_after is not None and self.checks > self.cancel_after:
            raise JobCancelled("cancelled by test")

    def update(self, *, message=None, **_kwargs) -> None:
        if message is not None:
            self.messages.append(message)

    def checkpoint(self, _value) -> None:
        pass


@pytest.fixture
def registered_slow():
    surveys.register("slow", SlowConnector)
    yield
    surveys._REGISTRY.pop("slow", None)


class TestConeSearchTimeout:
    """`_acquire_one` used to call `connector.cone_search()` directly and
    unconditionally: no ASTRA-enforced bound, no cancellation check, and no
    progress update between "Searching {survey}" and the call returning.
    A slow or unresponsive catalog query was indistinguishable, from the UI,
    from the whole application having hung.
    """

    def test_a_call_that_finishes_in_time_is_unaffected(
            self, isolated_root, registered_slow, cone):
        result = acquire.acquire(cone, survey_names=["slow"], limit=5)

        outcome = result.outcomes[0]
        assert outcome.error is None
        assert outcome.sources_found == 1

    def test_the_call_runs_on_a_worker_thread_not_inline(
            self, isolated_root, cone):
        """This is what makes a bound and cancellation possible at all --
        astroquery/lightkurve calls cannot be interrupted once entered."""
        connector = SlowConnector(delay=0.01)
        acquire._cone_search_with_timeout(connector, cone, 5, "SLOW")
        assert connector.was_called_from_a_worker_thread is True

    def test_exceeding_the_bound_is_reported_as_that_surveys_error(
            self, isolated_root, cone, monkeypatch):
        """The survey fails; the run does not hang. Matches how a genuine
        archive failure is already handled -- degrade, don't abort."""
        monkeypatch.setattr(acquire, "CONE_SEARCH_TIMEOUT_S", 0.05)
        monkeypatch.setattr(acquire, "CONE_SEARCH_POLL_S", 0.02)
        surveys._REGISTRY["slow"] = lambda **_kwargs: SlowConnector(delay=5.0)
        try:
            result = acquire.acquire(cone, survey_names=["slow"], limit=5)
        finally:
            surveys._REGISTRY.pop("slow", None)

        outcome = result.outcomes[0]
        assert outcome.error is not None
        assert "did not respond within" in outcome.error

    def test_a_slow_but_finishing_call_still_reports_progress_while_waiting(
            self, isolated_root, cone, monkeypatch):
        monkeypatch.setattr(acquire, "CONE_SEARCH_POLL_S", 0.02)
        connector = SlowConnector(delay=0.08)
        progress = FakeProgress()

        acquire._cone_search_with_timeout(connector, cone, 5, "SLOW", progress)

        assert any("Querying SLOW catalog" in message for message in progress.messages)

    def test_cancellation_is_checked_while_waiting_not_only_before_and_after(
            self, isolated_root, cone, monkeypatch):
        """Before this fix, Cancel was inert for the entire duration of one
        cone_search call -- up to astroquery's own internal timeout."""
        from astra.jobs import JobCancelled

        monkeypatch.setattr(acquire, "CONE_SEARCH_POLL_S", 0.02)
        connector = SlowConnector(delay=5.0)
        progress = FakeProgress(cancel_after=2)

        with pytest.raises(JobCancelled):
            acquire._cone_search_with_timeout(connector, cone, 5, "SLOW", progress)

        # Cancelled well before the connector's own 5s delay would have
        # elapsed -- this is the whole point of polling instead of blocking.
        assert progress.checks > 2

    def test_a_cancelled_cone_search_propagates_out_of_acquire_one(
            self, isolated_root, cone, monkeypatch):
        """JobCancelled must not be swallowed by the generic "archive
        failure" handler around cone_search -- it is a user action, not a
        data problem, and the caller (jobs.py) needs to see it to mark the
        job cancelled rather than failed."""
        from astra.jobs import JobCancelled

        monkeypatch.setattr(acquire, "CONE_SEARCH_POLL_S", 0.02)
        surveys._REGISTRY["slow"] = lambda **_kwargs: SlowConnector(delay=5.0)
        progress = FakeProgress(cancel_after=1)
        try:
            with pytest.raises(JobCancelled):
                acquire.acquire(cone, survey_names=["slow"], limit=5, progress=progress)
        finally:
            surveys._REGISTRY.pop("slow", None)
