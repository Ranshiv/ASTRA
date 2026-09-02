"""astronomy_adapter.py: crossmatch.py re-expressed via corroborate.core,
checked for agreement against crossmatch.group_sources directly (Direction
3)."""

from __future__ import annotations

from astra import crossmatch
from astra.corroborate import astronomy_adapter as adapter
from astra.surveys.base import SourceRef

FIXED_EPOCH = 2024.5


def _by_survey():
    return {
        "ZTF": [
            SourceRef(survey="ZTF", object_id="z1", ra_deg=10.0, dec_deg=20.0),
            SourceRef(survey="ZTF", object_id="z2", ra_deg=50.0, dec_deg=-10.0),
        ],
        "Gaia": [
            SourceRef(survey="Gaia", object_id="g1", ra_deg=10.0003, dec_deg=20.0003),
            SourceRef(survey="Gaia", object_id="g2", ra_deg=99.0, dec_deg=5.0),
        ],
    }


class TestGroupSourcesViaCore:
    def test_matches_crossmatch_group_sources_membership(self):
        by_survey = _by_survey()
        legacy_groups = crossmatch.group_sources(by_survey, epoch=FIXED_EPOCH,
                                                  anchor_survey="ZTF")
        core_groups = adapter.group_sources_via_core(by_survey, epoch=FIXED_EPOCH,
                                                      anchor_survey="ZTF")

        legacy_membership = sorted(
            [tuple(sorted(group.to_dict()["members"].items())) for group in legacy_groups])
        core_membership = sorted(
            [tuple(sorted(adapter.group_to_source_membership(group).items()))
            for group in core_groups])
        assert legacy_membership == core_membership

    def test_matches_crossmatch_default_anchor_choice(self):
        by_survey = _by_survey()
        legacy_groups = crossmatch.group_sources(by_survey, epoch=FIXED_EPOCH)
        core_groups = adapter.group_sources_via_core(by_survey, epoch=FIXED_EPOCH)
        assert len(legacy_groups) == len(core_groups)

    def test_beam_width_blending_matches_for_a_coarse_survey(self):
        by_survey = {
            "ZTF": [SourceRef(survey="ZTF", object_id="z1", ra_deg=10.0, dec_deg=20.0)],
            "TESS": [SourceRef(survey="TESS", object_id="t1", ra_deg=10.001, dec_deg=20.001)],
        }
        legacy_groups = crossmatch.group_sources(by_survey, radius_arcsec=30.0,
                                                  epoch=FIXED_EPOCH, anchor_survey="ZTF")
        core_groups = adapter.group_sources_via_core(by_survey, radius_arcsec=30.0,
                                                      epoch=FIXED_EPOCH, anchor_survey="ZTF")
        assert "TESS" in legacy_groups[0].blended
        assert "TESS" in core_groups[0].blended

    def test_empty_input_matches(self):
        assert crossmatch.group_sources({}) == adapter.group_sources_via_core({})

    def test_proper_motion_is_applied_before_grouping(self):
        # A Gaia source with large enough proper motion to have drifted well
        # outside a tight radius by the query epoch, unless corrected.
        by_survey = {
            "ZTF": [SourceRef(survey="ZTF", object_id="z1", ra_deg=10.0, dec_deg=0.0)],
            "Gaia": [SourceRef(survey="Gaia", object_id="g1", ra_deg=10.0, dec_deg=0.0,
                              extra={"pmra": 500.0, "pmdec": 0.0})],
        }
        epoch = crossmatch.GAIA_EPOCH + 10.0  # 10 years of drift at 500 mas/yr = 5 arcsec
        legacy_groups = crossmatch.group_sources(by_survey, radius_arcsec=2.0, epoch=epoch,
                                                  anchor_survey="ZTF")
        core_groups = adapter.group_sources_via_core(by_survey, radius_arcsec=2.0, epoch=epoch,
                                                      anchor_survey="ZTF")
        legacy_has_gaia = "Gaia" in legacy_groups[0].members
        core_has_gaia = "Gaia" in core_groups[0].members
        assert legacy_has_gaia == core_has_gaia == False
