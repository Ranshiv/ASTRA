"""Shared VizieR (CDS) Simple Cone Search connector for metadata-only
catalogues.

Eight connectors (chandra/swift/xmm/wise/twomass/galex/herschel/erosita)
each hit the same `vizier.cds.unistra.fr` Simple Cone Search endpoint with
the same request shape, and the same
cone_search -> parse_votable -> per-row try/except -> SourceRef shape --
differing only in which VizieR catalogue id, which column carries the
object id/RA/Dec, and which columns become `SourceRef.extra`. Factoring
that shape here means there is one place that knows how to reach VizieR and
build a `SourceRef` from a row, the same rationale `_mast_caom.py` already
applied to hubble.py/jwst.py sharing one archive's request shape.
"""

from __future__ import annotations

from .. import netclient
from ..tap import parse_votable
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"


def vizier_cone_rows(catalog: str, ra_deg: float, dec_deg: float,
                     radius_arcsec: float, limit: int) -> list[dict]:
    """One VizieR Simple Cone Search fetch, parsed to plain row dicts."""
    top = max(1, min(int(limit), 200))
    response = netclient.get(
        SCS_URL,
        {"-source": catalog, "RA": ra_deg, "DEC": dec_deg,
         "SR": radius_arcsec / 3600.0, "-out.max": top},
        timeout=60, provider="vizier",
    )
    return parse_votable(response.text, top)


class VizierConeConnector(SurveyConnector):
    """A metadata-only catalogue reached by VizieR's Simple Cone Search.

    Subclasses set `id_column` (and `ra_column`/`dec_column` if not the
    common `RAJ2000`/`DEJ2000`) and implement `extra_fields(row)` for the
    columns particular to their catalogue; the VizieR fetch and the
    row-to-`SourceRef` shape are shared. `fetch_light_curves` is likewise
    shared: every one of these catalogues is metadata/mean-photometry only.
    """

    id_column: str
    ra_column: str = "RAJ2000"
    dec_column: str = "DEJ2000"

    def __init__(self, release: str) -> None:
        self.release = release

    def _catalog(self) -> str:
        """VizieR catalogue id used for the fetch. Defaults to `self.release`
        (true for wise/twomass/galex/herschel/erosita, where the constructor's
        `release` argument *is* the catalogue id); chandra/swift/xmm override
        this because their `release` is a separate human-readable label (e.g.
        "csc2.1") decoupled from the catalogue's fixed VizieR id (e.g.
        "IX/70") -- see their module docstrings for why.
        """
        return self.release

    def extra_fields(self, row: dict) -> dict:
        """Which row columns become `SourceRef.extra`; catalogue-specific."""
        raise NotImplementedError

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        rows = vizier_cone_rows(self._catalog(), query.ra_deg, query.dec_deg,
                                query.radius_arcsec, top)
        sources: list[SourceRef] = []
        for row in rows:
            try:
                object_id = str(row[self.id_column])
                ra_deg = float(row[self.ra_column])
                dec_deg = float(row[self.dec_column])
            except (KeyError, TypeError, ValueError):
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id, ra_deg=ra_deg, dec_deg=dec_deg,
                extra=self.extra_fields(row),
            ))
        # Every row failing to parse is indistinguishable from a genuinely
        # empty cone unless this is logged: the per-row try/except above
        # exists so one malformed row doesn't drop the whole response, but
        # if VizieR silently renamed `id_column`/`ra_column`/`dec_column`
        # every row would hit that same except clause, and the connector
        # would report zero sources here forever, with no visible fault.
        if rows and not sources:
            import logging
            logging.getLogger(__name__).warning(
                "%s: VizieR returned %d row(s) for catalogue %r but none "
                "parsed as a source -- id_column=%r, ra_column=%r, "
                "dec_column=%r may no longer match the catalogue's real "
                "columns.", self.name, len(rows), self._catalog(),
                self.id_column, self.ra_column, self.dec_column)
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        return []
