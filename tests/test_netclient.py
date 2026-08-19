"""Resilient HTTP: retries, throttling and connection reuse.

Archive throttling used to become silent data loss — a 429 raised out of the
bare `requests.get`, the per-object handler caught it, and the object was
dropped with no count and no retry.
"""

from __future__ import annotations

import threading
import time

import pytest
import requests

from astra import netclient


@pytest.fixture(autouse=True)
def clean_client():
    netclient.reset()
    yield
    netclient.reset()


class TestRetryPolicy:
    def test_transient_statuses_are_retried(self):
        adapter = netclient.session().get_adapter("https://example.invalid")
        retry = adapter.max_retries

        for status in (429, 500, 502, 503, 504):
            assert status in retry.status_forcelist

    def test_client_errors_are_not_retried(self):
        """A 404 is a fact about the request, not a transient fault."""
        retry = netclient.session().get_adapter(
            "https://example.invalid").max_retries

        assert 404 not in retry.status_forcelist
        assert 400 not in retry.status_forcelist

    def test_backoff_and_budget_are_configured(self):
        retry = netclient.session().get_adapter(
            "https://example.invalid").max_retries

        assert retry.total == netclient.MAX_RETRIES
        assert retry.backoff_factor == netclient.BACKOFF_FACTOR
        assert retry.respect_retry_after_header is True

    def test_only_idempotent_methods_retry(self):
        retry = netclient.session().get_adapter(
            "https://example.invalid").max_retries
        assert "GET" in retry.allowed_methods
        assert "POST" not in retry.allowed_methods


class TestSessionReuse:
    def test_session_is_shared(self):
        """A fresh TLS handshake per object is pure overhead at scale."""
        assert netclient.session() is netclient.session()

    def test_reset_replaces_the_session(self):
        first = netclient.session()
        netclient.reset()
        assert netclient.session() is not first


class TestThrottle:
    def test_successive_requests_are_spaced(self, monkeypatch):
        # Comfortably above Windows' ~16 ms timer granularity, which makes a
        # tight assertion on a 50 ms interval flaky.
        monkeypatch.setitem(netclient.REQUEST_INTERVAL_SECONDS, "test", 0.20)

        started = time.monotonic()
        netclient.throttle("test")
        netclient.throttle("test")
        elapsed = time.monotonic() - started

        assert elapsed >= 0.18

    def test_distinct_providers_do_not_block_each_other(self, monkeypatch):
        monkeypatch.setitem(netclient.REQUEST_INTERVAL_SECONDS, "a", 0.2)
        monkeypatch.setitem(netclient.REQUEST_INTERVAL_SECONDS, "b", 0.2)

        netclient.throttle("a")
        started = time.monotonic()
        netclient.throttle("b")

        assert time.monotonic() - started < 0.1

    def test_throttle_is_thread_safe(self, monkeypatch):
        """Jobs run on a pool; the original bare dict was racy."""
        monkeypatch.setitem(netclient.REQUEST_INTERVAL_SECONDS, "race", 0.02)
        errors: list[BaseException] = []

        def worker():
            try:
                for _ in range(10):
                    netclient.throttle("race")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []


class TestGet:
    def test_backoff_grows_between_attempts(self):
        """The retry policy is live, not merely declared.

        urllib3 computes the delay from the configured backoff factor, so a
        429 storm is spread out instead of hammering the archive.
        """
        retry = netclient.session().get_adapter(
            "https://example.invalid").max_retries

        delays = []
        current = retry
        for _ in range(3):
            current = current.increment(method="GET", url="/x", error=None,
                                        _pool=None, _stacktrace=None,
                                        response=None)
            delays.append(current.get_backoff_time())

        assert delays[-1] > delays[0]
        assert all(delay >= 0 for delay in delays)

    def test_get_raises_for_status(self, monkeypatch):
        class FailingResponse:
            def raise_for_status(self):
                raise requests.HTTPError("500")

        monkeypatch.setattr(netclient, "throttle", lambda provider: None)
        monkeypatch.setattr(netclient, "session",
                            lambda: type("S", (), {
                                "get": staticmethod(
                                    lambda *a, **k: FailingResponse())})())

        with pytest.raises(requests.HTTPError):
            netclient.get("https://example.invalid", {}, timeout=1)

    def test_ztf_uses_the_resilient_client(self, monkeypatch):
        """Regression: the connector must not call requests.get directly."""
        from astra.surveys import ztf

        calls: list[str] = []

        class FakeResponse:
            text = "oid,hjd,mag,magerr,filtercode\n1,2458000.5,18.0,0.1,zg\n"

        def fake_get(url, params, timeout, provider="irsa"):
            calls.append(provider)
            return FakeResponse()

        monkeypatch.setattr(ztf.netclient, "get", fake_get)
        rows = ztf.ZTFConnector()._request({"ID": "1"})

        assert calls == ["irsa"]
        assert len(rows) == 1


class TestStreamingDownload:
    class Response:
        def __init__(self, chunks, declared=None):
            self.chunks = list(chunks)
            self.headers = ({"Content-Length": str(declared)}
                            if declared is not None else {})
            self.closed = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=1):
            yield from self.chunks

        def close(self):
            self.closed = True

    def test_streams_checksum_and_closes_response(self, monkeypatch, tmp_path):
        response = self.Response([b"abc", b"def"], declared=6)

        class Session:
            def get(self, *args, **kwargs):
                assert kwargs["stream"] is True
                return response

        monkeypatch.setattr(netclient, "session", lambda: Session())
        target = tmp_path / "product.fits"
        result = netclient.download("https://example.invalid/product", target,
                                    timeout=1, provider="test")

        assert target.read_bytes() == b"abcdef"
        assert result.bytes_written == 6
        assert result.sha256 == __import__("hashlib").sha256(b"abcdef").hexdigest()
        assert response.closed is True

    def test_declared_truncation_removes_partial_file(self, monkeypatch, tmp_path):
        response = self.Response([b"abc"], declared=6)

        class Session:
            def get(self, *args, **kwargs):
                return response

        monkeypatch.setattr(netclient, "session", lambda: Session())
        target = tmp_path / "product.fits"
        with pytest.raises(netclient.DownloadIncompleteError):
            netclient.download("https://example.invalid/product", target,
                               timeout=1, provider="test")
        assert not target.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_content_length_and_stream_limits_are_enforced(self, monkeypatch, tmp_path):
        response = self.Response([b"0123456789"], declared=10)

        class Session:
            def get(self, *args, **kwargs):
                return response

        monkeypatch.setattr(netclient, "session", lambda: Session())
        target = tmp_path / "product.fits"
        with pytest.raises(netclient.DownloadTooLargeError):
            netclient.download("https://example.invalid/product", target,
                               timeout=1, provider="test", max_bytes=5)
        assert not target.exists()
