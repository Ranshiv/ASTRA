"""discard.scan RPC handler (Direction 2, "anomalies in the discard pile").
No network -- `ZTFConnector` is monkeypatched with a fake that duck-types
`fetch_light_curves_with_quality`, matching `test_ztf_artifact_patches.py`'s
fake-connector convention."""

from __future__ import annotations

import numpy as np

from astra import rpc
from astra.surveys.base import LightCurve


class _FakeZTFConnector:
    def __init__(self):
        pass

    def fetch_light_curves_with_quality(self, source):
        n = 60
        time = np.arange(n, dtype=float)
        value = np.full(n, 18.0)
        value[20:25] = [18.4, 18.5, 18.55, 18.5, 18.4]
        catflags = np.zeros(n, dtype=np.uint32)
        catflags[20:25] = 32768
        curve = LightCurve(
            source=source, release="dr24", band="g", value_kind="mag",
            time=time, value=value, value_err=np.full(n, 0.02), time_system="HJD_UTC",
        )
        return [(curve, catflags)]


def test_discard_scan_recovers_a_coherent_run(monkeypatch):
    monkeypatch.setattr("astra.surveys.ztf.ZTFConnector", _FakeZTFConnector)

    response = rpc.dispatch({"id": 1, "method": "discard.scan", "params": {
        "object_id": "728116300014796", "ra_deg": 10.0, "dec_deg": 20.0,
    }})

    assert response["ok"] is True
    assert response["result"]["object_id"] == "728116300014796"
    records = response["result"]["records"]
    assert len(records) == 1
    assert records[0]["flag_category"] == "flagged"
    assert records[0]["coherent"] is True


def test_discard_scan_respects_min_run_length(monkeypatch):
    monkeypatch.setattr("astra.surveys.ztf.ZTFConnector", _FakeZTFConnector)

    response = rpc.dispatch({"id": 2, "method": "discard.scan", "params": {
        "object_id": "1", "ra_deg": 0.0, "dec_deg": 0.0, "min_run_length": 10,
    }})

    assert response["ok"] is True
    assert response["result"]["records"] == []
