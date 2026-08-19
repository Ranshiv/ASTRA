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
