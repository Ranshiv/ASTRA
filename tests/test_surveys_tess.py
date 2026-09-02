"""TESS survey helper contract: `_row_float`/`_record_sector` masked-cell
handling. No broader `TESSConnector` mock suite here (see
`test_surveys_kepler.py`'s module docstring) -- this covers only the real
bug found live and fixed this session.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.tess import TESSConnector, _record_sector, _row_float


class TestRowFloat:
    def test_masked_cell_falls_through_to_fallback(self):
        # Real bug found live against a real TESS-field query: a masked
        # (missing) coordinate cell -- common in real MAST search results,
        # astropy masked columns -- raises numpy.ma.MaskError on
        # `float(...)`, which the original `except (TypeError, ValueError)`
        # did not catch, crashing cone_search() for every target in the
        # batch over one target's missing coordinate.
        row = {"s_ra": np.ma.masked, "ra": 180.5}
        columns = {"s_ra", "ra"}
        # s_ra (masked) is skipped; ra is used instead, matching the
        # existing "fall through to the next candidate column" contract.
        assert _row_float(row, columns, ("s_ra", "ra"), fallback=0.0) == pytest.approx(180.5)

    def test_all_candidates_masked_uses_cone_fallback(self):
        row = {"s_ra": np.ma.masked, "ra": np.ma.masked}
        columns = {"s_ra", "ra"}
        assert _row_float(row, columns, ("s_ra", "ra"), fallback=42.0) == pytest.approx(42.0)


class TestRecordSector:
    def test_masked_sequence_number_is_skipped_not_fatal(self):
        source = SourceRef(survey="TESS", object_id="1", ra_deg=0.0, dec_deg=0.0,
                           extra={"sectors": []})
        # Must not raise -- a masked cell is "unknown sector", not a crash.
        _record_sector(source, {"sequence_number": np.ma.masked}, {"sequence_number"})
        assert source.extra["sectors"] == []

    def test_real_sector_is_recorded(self):
        source = SourceRef(survey="TESS", object_id="1", ra_deg=0.0, dec_deg=0.0,
                           extra={"sectors": []})
        _record_sector(source, {"sequence_number": 23}, {"sequence_number"})
        assert source.extra["sectors"] == [23]


@pytest.mark.live
class TestConeSearchLive:
    """Confirmed live this session: a real TESS-field cone search at the
    Kepler field centre (RA=291.41, Dec=41.5) hit the masked-cell bug
    above and crashed before the fix; it returns real sources now."""

    def test_returns_real_sources_without_crashing(self):
        sources = TESSConnector().cone_search(
            ConeQuery(ra_deg=291.41, dec_deg=41.5, radius_arcsec=600), limit=20)
        assert len(sources) > 0
