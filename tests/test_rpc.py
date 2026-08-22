"""RPC dispatch contract between the Rust core and the Python engine."""

from __future__ import annotations

import io
import json

from astra import rpc


def test_ping_round_trip():
    response = rpc.dispatch({"id": 1, "method": "ping"})
    assert response["ok"] is True
    assert response["id"] == 1
    assert response["result"]["pong"] is True


class TestDeepModelsWithoutTorch:
    """Released installers ship a CPU-only engine with no PyTorch.

    Verified against the real 0.1.0 installer: `deep.train` returned a bare
    "No module named 'torch'", which reads like a broken installation rather
    than a deliberate build choice.
    """

    def _hide_torch(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    def test_deep_train_explains_the_limitation(self, monkeypatch):
        self._hide_torch(monkeypatch)
        response = rpc.dispatch({"id": 1, "method": "deep.train",
                                 "params": {"name": "x"}})

        assert response["ok"] is False
        error = response["error"]
        assert "No module named" not in error
        assert "CPU-only" in error
        assert "engine[gpu]" in error

    def test_deep_compare_without_deep_models_is_still_allowed(self, monkeypatch):
        """The baseline half of the study does not need PyTorch."""
        self._hide_torch(monkeypatch)
        response = rpc.dispatch({"id": 2, "method": "deep.compare",
                                 "params": {"include_deep": False}})

        # It may still fail for lack of data, but never for lack of PyTorch.
        if response["ok"] is False:
            assert "CPU-only" not in response["error"]

    def test_deep_compare_with_deep_models_is_refused(self, monkeypatch):
        self._hide_torch(monkeypatch)
        response = rpc.dispatch({"id": 3, "method": "deep.compare",
                                 "params": {"include_deep": True}})

        assert response["ok"] is False
        assert "CPU-only" in response["error"]

    def test_engine_survives_the_refusal(self, monkeypatch):
        """A refused request must not take the engine down."""
        self._hide_torch(monkeypatch)
        rpc.dispatch({"id": 1, "method": "deep.train", "params": {}})

        assert rpc.dispatch({"id": 2, "method": "ping"})["ok"] is True


def test_unknown_method_is_an_error_not_an_exception():
    response = rpc.dispatch({"id": 7, "method": "no.such.method"})
    assert response["ok"] is False
    assert response["id"] == 7
    assert "unknown method" in response["error"]


def test_handler_failure_is_reported_with_traceback(monkeypatch):
    def boom(_params):
        raise RuntimeError("detector offline")

    monkeypatch.setitem(rpc.HANDLERS, "boom", boom)
    response = rpc.dispatch({"id": 2, "method": "boom"})

    assert response["ok"] is False
    assert response["error"] == "detector offline"
    assert "RuntimeError" in response["traceback"]


def test_paths_handler_reports_the_layout():
    result = rpc.dispatch({"id": 3, "method": "paths"})["result"]
    assert {"root", "projects", "datasets", "models", "cache"} <= result.keys()


def test_event_handlers_round_trip_in_project_workspace(isolated_root):
    payload = {
        "event_id": "rpc-event",
        "packet_id": "rpc-packet",
        "event_time": "2026-08-20T12:00:00Z",
        "localization": {"ra_deg": 1.0, "dec_deg": 2.0},
    }
    response = rpc.dispatch({"id": 10, "method": "events.ingest", "params": {
        "provider": "generic", "payload": payload,
    }})
    assert response["ok"] is True
    packet = response["result"]
    listed = rpc.dispatch({"id": 11, "method": "events.list", "params": {}})
    assert listed["ok"] is True
    assert listed["result"][0]["event_id"] == "rpc-event"
    replay = rpc.dispatch({"id": 12, "method": "events.replay", "params": {}})
    assert replay["ok"] is True
    assert replay["result"][0]["packet_key"] == packet["packet_key"]


def test_significance_handlers_persist_explicit_interpretation_layer(isolated_root):
    response = rpc.dispatch({"id": 13, "method": "significance.calibrate", "params": {
        "scores": [0.1, 0.5, 0.9], "reference_scores": [0.1, 0.2, 0.3],
        "threshold": 0.8,
    }})
    assert response["ok"] is True
    assert response["result"]["estimated_fdr"] >= 0
    assert response["result"]["path"]


def test_followup_handler_is_draft_only():
    response = rpc.dispatch({"id": 14, "method": "followup.plan", "params": {
        "ra_deg": 180.0, "dec_deg": 22.0,
        "start_utc": "2026-08-20T00:00:00Z", "duration_hours": 2,
    }})
    assert response["ok"] is True
    assert response["result"]["mode"] == "draft_only"


def test_alert_poll_handler_accepts_bounded_payload(isolated_root):
    response = rpc.dispatch({"id": 16, "method": "alerts.poll", "params": {
        "provider": "gcn",
        "payload": {"alerts": [{"event_id": "rpc-alert", "packet_id": "p1"}]},
    }})
    assert response["ok"] is True
    assert response["result"]["ingested"] == 1
    assert response["result"]["new_packets"] == 1


def test_tap_query_handler_forwards_read_only_query(monkeypatch, isolated_root):
    class Response:
        headers = {"Content-Type": "text/csv"}
        text = "ra,dec\n1.0,2.0\n"

    monkeypatch.setattr(rpc.tap.netclient, "get",
                        lambda *args, **kwargs: Response())
    response = rpc.dispatch({"id": 17, "method": "tap.query", "params": {
        "service": "https://example.invalid/tap/sync",
        "adql": "SELECT ra, dec FROM sources",
        "max_rows": 3,
    }})
    assert response["ok"] is True
    assert response["result"]["rows"] == [{"ra": 1.0, "dec": 2.0}]
    assert response["result"]["query"]["limit"] == 3


def test_event_association_handler_is_conservative(monkeypatch, isolated_root):
    from astra.candidates import Candidate
    from astra import events

    events.ingest("gcn", {"event_id": "assoc-event", "packet_id": "assoc-packet",
                           "event_time": "2026-08-20T12:00:00Z",
                           "localization": {"ra_deg": 10.0, "dec_deg": 20.0}},
                  root=isolated_root.projects)
    candidate = Candidate(candidate_id="assoc-candidate", object_id="obj",
                          survey="ZTF", band="g", ra_deg=10.0, dec_deg=20.0,
                          features={"event_time": "2026-08-20T12:00:00Z"})
    monkeypatch.setattr(rpc.association.candidates, "load", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(rpc.association.candidates, "save", lambda *args, **kwargs: isolated_root.projects / "candidates.json")
    response = rpc.dispatch({"id": 19, "method": "events.associate", "params": {}})
    assert response["ok"] is True
    assert response["result"]["associations"] == 1


def test_physical_characterize_handler_returns_context_only():
    response = rpc.dispatch({"id": 20, "method": "physical.characterize", "params": {
        "photometry": {"gaia_bp": 15.1, "gaia_g": 14.8, "gaia_rp": 14.5,
                        "g": 15.0, "r": 14.8, "i": 14.7},
    }})
    assert response["ok"] is True
    assert response["result"]["quality"] == "usable"
    assert "temperature_k" in response["result"]


def test_review_next_uses_candidate_uncertainty(monkeypatch):
    monkeypatch.setattr(rpc.candidates_mod, "load", lambda name, root: [{
        "candidate_id": "a", "score": {"model_agreement": 1},
        "artifact": {"likelihood": 0.5}, "significance": {"tail_probability": 0.5},
        "features": {"x": 1},
    }])
    response = rpc.dispatch({"id": 15, "method": "review.next", "params": {"limit": 1}})
    assert response["ok"] is True
    assert response["result"][0]["candidate_id"] == "a"


def test_hardware_handler_always_returns_a_device():
    result = rpc.dispatch({"id": 4, "method": "hardware"})["result"]
    assert result["device"] in {"cpu", "cuda"}
    assert result["reason"]


def test_repeated_ablation_handler_forwards_its_parameters(monkeypatch):
    seen = {}

    def fake_run_repeated(*, fraction, seeds, survey=None):
        seen.update(fraction=fraction, seeds=seeds, survey=survey)
        return {"experiment_id": "repeated-test", "seeds": list(seeds)}

    monkeypatch.setattr(rpc.ablation, "run_repeated", fake_run_repeated)

    response = rpc.dispatch({
        "id": 8,
        "method": "ablation.repeated",
        "params": {"fraction": 0.2, "seeds": [3, 5]},
    })

    assert response == {
        "id": 8,
        "ok": True,
        "result": {"experiment_id": "repeated-test", "seeds": [3, 5]},
    }
    # survey defaults to None: unstratified, which is the previous behaviour.
    assert seen == {"fraction": 0.2, "seeds": (3, 5), "survey": None}


def test_serve_reads_lines_and_skips_blanks():
    stdin = io.StringIO('{"id":1,"method":"ping"}\n\n{"id":2,"method":"ping"}\n')
    stdout = io.StringIO()

    rpc.serve(stdin, stdout)

    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2]


def test_serve_survives_malformed_json():
    stdin = io.StringIO('not json\n{"id":5,"method":"ping"}\n')
    stdout = io.StringIO()

    rpc.serve(stdin, stdout)

    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert replies[0]["ok"] is False
    assert replies[1]["id"] == 5


class TestAcquireSurveyOptions:
    """B2: the acquire.cone RPC method must forward an optional
    survey_options dict (per-survey connector kwargs, e.g. an HLSP author
    choice for TESS) to acquire.acquire(), and pass None when the caller
    omits it -- reproducing today's call exactly. See docs/DEFERRED.txt
    Phase 8."""

    def _capture_acquire_call(self, monkeypatch):
        calls = []

        class FakeOutcome:
            def to_dict(self):
                return {"surveys": []}

        def fake_acquire(query, **kwargs):
            calls.append(kwargs)
            return FakeOutcome()

        monkeypatch.setattr(rpc.acquire, "acquire", fake_acquire)
        return calls

    def test_survey_options_are_forwarded_when_present(self, monkeypatch):
        calls = self._capture_acquire_call(monkeypatch)

        response = rpc.dispatch({
            "id": 1, "method": "acquire.cone",
            "params": {"ra_deg": 180.0, "dec_deg": 22.0,
                      "survey_options": {"tess": {"author": "QLP"}}},
        })

        assert response["ok"] is True
        assert calls[0]["survey_options"] == {"tess": {"author": "QLP"}}

    def test_survey_options_default_to_none_when_absent(self, monkeypatch):
        calls = self._capture_acquire_call(monkeypatch)

        response = rpc.dispatch({
            "id": 1, "method": "acquire.cone",
            "params": {"ra_deg": 180.0, "dec_deg": 22.0},
        })

        assert response["ok"] is True
        assert calls[0]["survey_options"] is None

    def test_skip_existing_defaults_true_and_forwards_when_overridden(self, monkeypatch):
        calls = self._capture_acquire_call(monkeypatch)

        rpc.dispatch({
            "id": 1, "method": "acquire.cone",
            "params": {"ra_deg": 180.0, "dec_deg": 22.0},
        })
        assert calls[0]["skip_existing"] is True

        rpc.dispatch({
            "id": 2, "method": "acquire.cone",
            "params": {"ra_deg": 180.0, "dec_deg": 22.0, "skip_existing": False},
        })
        assert calls[1]["skip_existing"] is False


class TestAcquireProject:
    """acquire.project runs one acquisition per region in a project's
    query_regions, via the same job-dispatch path as acquire.cone."""

    def _capture_acquire_project_call(self, monkeypatch):
        calls = []

        class FakeResult:
            def to_dict(self):
                return {"project_id": "proj", "regions": [], "totals": {}}

        def fake_acquire_project(project_id, **kwargs):
            calls.append({"project_id": project_id, **kwargs})
            return FakeResult()

        monkeypatch.setattr(rpc.acquire, "acquire_project", fake_acquire_project)
        return calls

    def test_dispatches_to_acquire_project_with_params(self, monkeypatch):
        calls = self._capture_acquire_project_call(monkeypatch)

        response = rpc.dispatch({
            "id": 1, "method": "acquire.project",
            "params": {"project_id": "proj", "surveys": ["ztf"], "limit": 5,
                      "skip_existing": False},
        })

        assert response["ok"] is True
        assert calls[0]["project_id"] == "proj"
        assert calls[0]["survey_names"] == ["ztf"]
        assert calls[0]["limit"] == 5
        assert calls[0]["skip_existing"] is False

    def test_skip_existing_defaults_to_true(self, monkeypatch):
        calls = self._capture_acquire_project_call(monkeypatch)

        rpc.dispatch({
            "id": 1, "method": "acquire.project",
            "params": {"project_id": "proj"},
        })

        assert calls[0]["skip_existing"] is True


class _BrokenPipeStdout:
    """A stdout double whose write always fails, like a pipe with no reader.

    This is the exact shape of the real captured crash: Windows raises
    `OSError: [Errno 22] Invalid argument` (not `BrokenPipeError`) when the
    reader side of a pipe is already gone -- see rpc.serve()'s docstring.
    """

    def __init__(self):
        self.flushed = False

    def write(self, _text):
        raise OSError(22, "Invalid argument")

    def flush(self):
        self.flushed = True


class TestBrokenPipeShutdown:
    """serve() must exit quietly on a broken write, not propagate a crash.

    The real crash-*.log this project captured was exactly an unhandled
    OSError from this write call taking down the whole engine process.
    """

    def test_a_broken_stdout_write_exits_serve_without_raising(self):
        stdin = io.StringIO('{"id":1,"method":"ping"}\n')
        stdout = _BrokenPipeStdout()

        rpc.serve(stdin, stdout)  # must return, not raise

    def test_a_broken_write_leaves_flush_unreached(self):
        """write() failing means flush() was never reached for that line --
        there is nothing left to flush once the write itself failed."""
        stdin = io.StringIO('{"id":1,"method":"ping"}\n')
        stdout = _BrokenPipeStdout()

        rpc.serve(stdin, stdout)

        assert stdout.flushed is False

    def test_a_later_broken_line_still_exits_cleanly(self):
        """Not just the first line: any line's write failing must exit
        serve() rather than raise, however many succeeded before it."""
        stdin = io.StringIO(
            '{"id":1,"method":"ping"}\n{"id":2,"method":"ping"}\n'
        )

        class _FailOnSecondWrite:
            def __init__(self):
                self.calls = 0

            def write(self, _text):
                self.calls += 1
                if self.calls == 2:
                    raise OSError(22, "Invalid argument")

            def flush(self):
                pass

        stdout = _FailOnSecondWrite()
        rpc.serve(stdin, stdout)  # must return, not raise
        assert stdout.calls == 2


class TestManifestListDoesNotCrashOnIdentityParams:
    """`_handle_manifest_list` used to reference an undefined `payload`
    variable whenever survey/release/object_id/band were supplied, raising
    `UnboundLocalError`. Caught by dispatch() so never a process crash, but
    a genuinely broken code path reachable via job.submit."""

    def test_survey_param_no_longer_raises(self, isolated_root):
        response = rpc.dispatch({
            "id": 1, "method": "manifest.list",
            "params": {"survey": "ZTF"},
        })
        assert response["ok"] is True
        assert response["result"] == []

    def test_all_four_identity_params_no_longer_raise(self, isolated_root):
        response = rpc.dispatch({
            "id": 2, "method": "manifest.list",
            "params": {"survey": "ZTF", "release": "dr24",
                      "object_id": "123", "band": "g"},
        })
        assert response["ok"] is True

    def test_project_id_still_selects_the_manifest_root(self, isolated_root):
        response = rpc.dispatch({
            "id": 3, "method": "manifest.list",
            "params": {"project_id": "does-not-exist"},
        })
        assert response["ok"] is True
        assert response["result"] == []
