from __future__ import annotations

import numpy as np
import pytest

from astra import modalitymatrix


def payload(kind: str, value: float, object_id: str = "o1") -> dict:
    return {
        "schema_version": 1,
        "identity": {"survey": "ZTF", "release": "dr24", "object_id": object_id, "band": "g"},
        "source": {"sha256": "abc", "path": f"{kind}.fits"},
        "features": {"signal": value, "valid": True},
        "quality": {"method": "fixture"},
    }


def test_sidecar_round_trip_and_keyed_join(tmp_path):
    saved = modalitymatrix.save_payloads([payload("image", 3.0)], "image", root=tmp_path)
    table = modalitymatrix.load(saved.path)
    rows, report = modalitymatrix.join_rows([
        {"survey": "ZTF", "release": "dr24", "object_id": "o1", "band": "g"},
        {"survey": "ZTF", "release": "dr24", "object_id": "o2", "band": "g"},
    ], table, kind="image")
    assert report["matched_rows"] == 1
    assert rows[0]["image__signal"] == 3.0
    assert rows[1]["image__signal"] is None


def test_sidecar_rejects_wrong_kind(tmp_path):
    try:
        modalitymatrix.save_payloads([], "photometry", root=tmp_path)
    except ValueError as exc:
        assert "image, spectral, or multiband" in str(exc)
    else:
        raise AssertionError("wrong sidecar kind was accepted")


def test_empty_sidecar_keeps_kind_metadata(tmp_path):
    modalitymatrix.save_payloads([], "spectral", root=tmp_path)
    listed = modalitymatrix.list_sidecars(tmp_path)
    assert listed[0]["kind"] == "spectral"


def test_multiband_kind_is_accepted(tmp_path):
    """A multiband row spans a group of bands, so it is keyed with the
    __multiband__ sentinel rather than a real band name -- see multiband.py."""
    multiband_payload = {
        "schema_version": 1,
        "identity": {"survey": "ZTF", "release": "dr24", "object_id": "o1",
                    "band": "__multiband__"},
        "source": {"sha256": "", "path": ""},
        "features": {"best_period_days": 1.234, "period_snr": 12.0},
        "quality": {"bands_contributing": 2.0},
    }
    saved = modalitymatrix.save_payloads([multiband_payload], "multiband", root=tmp_path)
    table = modalitymatrix.load(saved.path)
    rows, report = modalitymatrix.join_rows([
        {"survey": "ZTF", "release": "dr24", "object_id": "o1", "band": "__multiband__"},
        {"survey": "ZTF", "release": "dr24", "object_id": "o2", "band": "__multiband__"},
    ], table, kind="multiband")

    assert report["matched_rows"] == 1
    assert rows[0]["multiband__best_period_days"] == pytest.approx(1.234)
    assert rows[1]["multiband__best_period_days"] is None
