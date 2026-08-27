"""Live, on-demand contract check for real GCN per-instrument notice filtering.

Not run by pytest's default collection (see the `live` marker registered in
engine/pyproject.toml -- `addopts = "-m 'not live'"` excludes it). Run
explicitly with:

    pytest tests/test_alerts_gcn_live.py -m live

`alerts.py`'s `DEFAULT_ENDPOINTS` routes `icecube`/`fermi`/`swift` through
the same generic `gcn` endpoint (`https://gcn.nasa.gov/alerts`) with no
confirmed per-instrument filter -- the module comment above that dict
documents this as "documented, not yet confirmed against a live fetch".
Module 4 of the approved research-modules plan asks for exactly this
confirmation before `association.event_to_event_correlation` /
`calibrate_event_graph` (already implemented, see test_event_graph.py) can
be trusted end-to-end against a REAL multi-messenger notice stream, not just
synthetic event dicts.

This file does not change any transport code: `alerts.poll(provider,
params={...})` already accepts an arbitrary query-parameter override (the
same mechanism `surveys/alerce.py`'s `survey="lsst"` override already uses),
so once GCN's real filtering contract is confirmed by running this file, the
correct `params` shape can be passed through unchanged.
"""

from __future__ import annotations

import pytest

from astra import alerts

pytestmark = pytest.mark.live

# GCN Classic over the new gcn.nasa.gov REST API documents a `type=` query
# parameter for filtering by notice type (e.g. "ICECUBE_ASTROTRACK_GOLD",
# "FERMI_GBM_FIN_POS", "SWIFT_BAT_GRB_POS"). This is the DOCUMENTED, not yet
# live-confirmed, shape this test checks -- see the module docstring.
CANDIDATE_NOTICE_TYPE_PARAMS = {
    "icecube": {"type": "ICECUBE_ASTROTRACK_GOLD"},
    "fermi": {"type": "FERMI_GBM_FIN_POS"},
    "swift": {"type": "SWIFT_BAT_GRB_POS"},
}


class TestGcnNoticeTypeFiltering:
    @pytest.mark.parametrize("provider", ["icecube", "fermi", "swift"])
    def test_poll_with_a_notice_type_filter_does_not_error(self, isolated_root, provider):
        """Confirms the request succeeds and returns a well-formed response.

        This intentionally does NOT assert every returned packet actually
        matches the requested notice type -- that would require a live
        packet to inspect, which may not exist at test-run time. What it
        confirms is that GCN accepts the documented `type=` parameter
        without erroring, and that the resulting poll still reports the
        `duplicate_rate`/`latency_summary` metrics `alerts.poll` always
        computes -- if GCN's real filter parameter name differs from
        `CANDIDATE_NOTICE_TYPE_PARAMS` above, this test's failure (or an
        empty/error response) is the signal to update it.
        """
        result = alerts.poll(provider, limit=5,
                             params=CANDIDATE_NOTICE_TYPE_PARAMS[provider])

        assert result["state"] in {"ok", "partial"}, result
        assert "duplicate_rate" in result
        assert "latency_summary" in result
