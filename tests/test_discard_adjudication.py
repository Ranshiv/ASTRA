"""discard_adjudication.py: independent forced-photometry verdicts on
discard-pile events (Direction 2, step 3). No network -- every scene is a
synthetic circular-Gaussian image built by
`ztf_forced_photometry._synthetic_gaussian_image`, matching that module's
own test convention."""

from __future__ import annotations

import numpy as np
import pytest

from astra import discard_adjudication as da
from astra import ztf_forced_photometry as zfp
from astra.discard_pile import DiscardRecord

SHAPE = (21, 21)


def _record(magnitude_offset=0.4):
    return DiscardRecord(
        object_id="1", survey="ZTF", band="g", flag_category="flagged",
        epoch_count=3, time_start=10.0, time_end=12.0,
        magnitude_offset=magnitude_offset, max_step=0.05, coherent=True,
    )


def _cube(fluxes: list[float], *, noise_sigma=2.0, seed=0):
    positions = zfp.build_scene_positions((10.0, 10.0), [], shape=SHAPE)
    rng = np.random.default_rng(seed)
    cube = np.stack([
        zfp._synthetic_gaussian_image(SHAPE, positions, {"target": flux}, 2.0,
                                      background=10.0, noise_sigma=noise_sigma, rng=rng)
        for flux in fluxes
    ])
    return cube, positions


class TestAdjudicate:
    def test_a_real_significant_fade_matches_a_fainter_catalog_offset(self):
        # Baseline bright, then a real dip during the flagged epochs.
        fluxes = [5000.0, 5050.0, 4950.0, 1500.0, 1450.0, 1550.0, 5010.0, 4990.0]
        cube, positions = _cube(fluxes, noise_sigma=5.0)
        flagged_indices = [3, 4, 5]
        record = _record(magnitude_offset=0.4)  # fainter, matching a flux dip

        result = da.adjudicate(record, cube, positions, flagged_indices)

        assert result.verdict == "likely_real"
        assert result.flux_z_score is not None
        assert result.flagged_flux_mean < result.baseline_flux_mean

    def test_no_real_flux_change_is_an_artifact_verdict(self):
        fluxes = [5000.0 + 20.0 * i for i in range(8)]  # flat, tiny noise-scale drift
        cube, positions = _cube(fluxes, noise_sigma=3.0)
        flagged_indices = [3, 4, 5]
        record = _record(magnitude_offset=0.4)

        result = da.adjudicate(record, cube, positions, flagged_indices)

        assert result.verdict == "likely_artifact"

    def test_direction_mismatch_is_an_artifact_verdict(self):
        # Independent photometry gets BRIGHTER during the flagged window,
        # but the catalog claims the source got fainter there.
        fluxes = [1500.0, 1450.0, 1550.0, 5000.0, 5050.0, 4950.0, 1490.0, 1510.0]
        cube, positions = _cube(fluxes, noise_sigma=5.0)
        flagged_indices = [3, 4, 5]
        record = _record(magnitude_offset=0.4)  # catalog says fainter

        result = da.adjudicate(record, cube, positions, flagged_indices)

        assert result.verdict == "likely_artifact"
        assert "disagrees in direction" in result.reasons[0]

    def test_too_few_flagged_epochs_is_inconclusive(self):
        fluxes = [5000.0, 5050.0, 4950.0, 1500.0]
        cube, positions = _cube(fluxes, noise_sigma=3.0)
        record = _record()

        result = da.adjudicate(record, cube, positions, flagged_indices=[3],
                               min_valid_epochs=2)

        assert result.verdict == "inconclusive"
        assert result.valid_flagged_epochs == 1

    def test_too_few_baseline_epochs_is_inconclusive(self):
        fluxes = [5000.0, 1500.0, 1450.0]
        cube, positions = _cube(fluxes, noise_sigma=3.0)
        record = _record()

        result = da.adjudicate(record, cube, positions, flagged_indices=[1, 2],
                               min_valid_epochs=2)

        assert result.verdict == "inconclusive"
        assert result.valid_baseline_epochs == 1

    def test_missing_target_position_is_inconclusive(self):
        # A scene built without a "target"-labelled position at all.
        from astra.ztf_forced_photometry import ScenePosition
        cube = np.zeros((3, *SHAPE))
        positions = [ScenePosition("n1", 15.0, 10.0)]

        result = da.adjudicate(_record(), cube, positions, flagged_indices=[0])

        assert result.verdict == "inconclusive"
        assert "no target position" in result.reasons[0]

    def test_to_dict_includes_the_record_and_verdict(self):
        fluxes = [5000.0] * 4 + [1500.0] * 4
        cube, positions = _cube(fluxes, noise_sigma=5.0)
        result = da.adjudicate(_record(), cube, positions, flagged_indices=[4, 5, 6, 7])

        payload = result.to_dict()
        assert payload["record"]["object_id"] == "1"
        assert payload["verdict"] in ("likely_real", "likely_artifact", "inconclusive")
