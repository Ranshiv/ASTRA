"""candidates.spatial: RA/Dec/Gaia-distance for the 3D spatial view.

The live candidate pipeline never joins Gaia distance columns -- only the
offline ablation study does (featurematrix.join_gaia_columns). This handler
reuses that same join against a deliberately empty FeatureMatrix built only
from already-loaded candidates' identities, so these tests exercise the real
join and the real SNR gate rather than a reimplementation of either.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import candidates as candidates_mod
from astra import metadata, rpc, store
from astra.candidates import Candidate
from astra.surveys.base import LightCurve, SourceRef


def _ztf_curve(object_id: str, ra_deg: float, dec_deg: float) -> LightCurve:
    source = SourceRef(survey="ZTF", object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg)
    return LightCurve(source=source, release="dr24", band="g", value_kind="mag",
                      time=2458000.0 + np.arange(30, dtype=np.float64) * 0.5,
                      value=np.full(30, 18.0), value_err=np.full(30, 0.02))


def _candidate(candidate_id: str, path: str, ra_deg: float, dec_deg: float,
              score_total: float = 0.5) -> Candidate:
    return Candidate(candidate_id=candidate_id, object_id="obj", survey="ZTF",
                     band="g", ra_deg=ra_deg, dec_deg=dec_deg, path=path,
                     score={"total": score_total})


class TestCandidatesSpatial:
    def test_reliable_parallax_is_reported_with_a_distance(self, isolated_root, monkeypatch):
        curve = _ztf_curve("obj1", ra_deg=180.0, dec_deg=22.0)
        result = store.write_curve(curve)
        metadata.upsert_sources(isolated_root.projects, [{
            "source_key": "Gaia/dr3/1", "survey": "Gaia", "release": "dr3",
            "object_id": "1", "ra_deg": 180.0, "dec_deg": 22.0,
            # High SNR: 5 mas parallax, tiny error -> well above the 5.0 gate.
            "extra": {"parallax": 5.0, "parallax_error": 0.05, "phot_g_mean_mag": 15.0},
        }])
        candidate = _candidate("cand1", str(result.path), 180.0, 22.0)
        monkeypatch.setattr(candidates_mod, "load", lambda *a, **k: [candidate])

        response = rpc.dispatch({"id": 1, "method": "candidates.spatial", "params": {}})

        assert response["ok"] is True
        point = response["result"]["points"][0]
        assert point["distance_reliable"] is True
        assert point["gaia_distance_pc"] == pytest.approx(200.0)
        assert point["gaia_parallax_snr"] == pytest.approx(100.0)
        assert response["result"]["reliable"] == 1

    def test_low_snr_parallax_is_excluded_not_plotted(self, isolated_root, monkeypatch):
        """The same SNR<5 gate scoring.py already uses for luminosity checks."""
        curve = _ztf_curve("obj1", ra_deg=180.0, dec_deg=22.0)
        result = store.write_curve(curve)
        metadata.upsert_sources(isolated_root.projects, [{
            "source_key": "Gaia/dr3/1", "survey": "Gaia", "release": "dr3",
            "object_id": "1", "ra_deg": 180.0, "dec_deg": 22.0,
            # SNR = 5.0 / 1.2 ~= 4.17, just under the threshold.
            "extra": {"parallax": 5.0, "parallax_error": 1.2},
        }])
        candidate = _candidate("cand1", str(result.path), 180.0, 22.0)
        monkeypatch.setattr(candidates_mod, "load", lambda *a, **k: [candidate])

        response = rpc.dispatch({"id": 1, "method": "candidates.spatial", "params": {}})

        point = response["result"]["points"][0]
        assert point["distance_reliable"] is False
        # A measured but unreliable distance is still reported, not hidden --
        # the caller decides whether to plot it, the engine only flags it.
        assert point["gaia_distance_pc"] == pytest.approx(200.0)
        assert response["result"]["reliable"] == 0

    def test_unmatched_candidate_reports_no_distance(self, isolated_root, monkeypatch):
        curve = _ztf_curve("obj1", ra_deg=180.0, dec_deg=22.0)
        result = store.write_curve(curve)
        # No Gaia sources registered at all.
        candidate = _candidate("cand1", str(result.path), 180.0, 22.0)
        monkeypatch.setattr(candidates_mod, "load", lambda *a, **k: [candidate])

        response = rpc.dispatch({"id": 1, "method": "candidates.spatial", "params": {}})

        point = response["result"]["points"][0]
        assert point["gaia_distance_pc"] is None
        assert point["distance_reliable"] is False
        assert response["result"]["gaia_matched"] == 0

    def test_empty_candidate_list_returns_an_empty_but_valid_result(self, isolated_root, monkeypatch):
        monkeypatch.setattr(candidates_mod, "load", lambda *a, **k: [])

        response = rpc.dispatch({"id": 1, "method": "candidates.spatial", "params": {}})

        assert response["ok"] is True
        assert response["result"] == {
            "points": [], "total": 0, "reliable": 0,
            "snr_threshold": rpc.GAIA_PARALLAX_SNR_THRESHOLD,
            "gaia_matched": 0, "gaia_match_rate": None,
        }

    def test_score_total_travels_through_for_point_colouring(self, isolated_root, monkeypatch):
        curve = _ztf_curve("obj1", ra_deg=180.0, dec_deg=22.0)
        result = store.write_curve(curve)
        candidate = _candidate("cand1", str(result.path), 180.0, 22.0, score_total=0.87)
        monkeypatch.setattr(candidates_mod, "load", lambda *a, **k: [candidate])

        response = rpc.dispatch({"id": 1, "method": "candidates.spatial", "params": {}})

        assert response["result"]["points"][0]["score_total"] == pytest.approx(0.87)

    def test_top_limits_how_many_candidates_are_returned(self, isolated_root, monkeypatch):
        curve = _ztf_curve("obj1", ra_deg=180.0, dec_deg=22.0)
        result = store.write_curve(curve)
        candidates = [_candidate(f"cand{i}", str(result.path), 180.0, 22.0)
                     for i in range(5)]
        monkeypatch.setattr(candidates_mod, "load", lambda *a, **k: candidates)

        response = rpc.dispatch({"id": 1, "method": "candidates.spatial",
                                 "params": {"top": 2}})

        assert response["result"]["total"] == 2
        assert len(response["result"]["points"]) == 2
