"""Gaia — astrometry and mean photometry.

Gaia is a catalogue connector, not a time-domain one: the main table gives one
row per source, so `fetch_light_curves` returns nothing by design. Its value to
ASTRA is `fetch_source_record`, which supplies the parallax, proper motion and
colour used for the physical-inconsistency term in the candidate score
(plan section 16) and for the stellar characterisation in section 20.
"""

from __future__ import annotations

import numpy as np

from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

DEFAULT_RELEASE = "dr3"
DR4_EXPECTED_RELEASE = "2026-12-02"

# Requested explicitly rather than with SELECT *, because the main table has
# ~150 columns and the extra transfer is pure waste at survey scale.
COLUMNS = (
    "source_id", "ra", "dec", "parallax", "parallax_error",
    "pmra", "pmdec", "phot_g_mean_mag", "phot_bp_mean_mag",
    "phot_rp_mean_mag", "radial_velocity", "phot_variable_flag",
    "a_g_val", "ebpminrp_val",
)


class GaiaConnector(SurveyConnector):
    name = "Gaia"
    capabilities = ("catalogue", "astrometry", "mean_photometry")
    resolution_arcsec = 0.4

    def __init__(self, release: str = DEFAULT_RELEASE, row_limit: int = 200) -> None:
        self.release = release
        self.row_limit = row_limit

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.gaia import Gaia

        Gaia.ROW_LIMIT = min(limit, self.row_limit)
        coord = SkyCoord(query.ra_deg, query.dec_deg, unit="deg")
        table = Gaia.cone_search(
            coord, radius=query.radius_arcsec * u.arcsec
        ).get_results()

        if table is None or len(table) == 0:
            return []

        return [
            SourceRef(
                survey=self.name,
                object_id=str(row["source_id"]),
                ra_deg=float(row["ra"]),
                dec_deg=float(row["dec"]),
                extra=_row_to_extra(row, table.colnames),
            )
            for row in table[:limit]
        ]

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        """Gaia's main table has no time series; epoch photometry is a later phase."""
        return []

    def fetch_source_record(self, source: SourceRef) -> dict:
        """Astrometric and photometric properties already captured by the search."""
        return {"survey": self.name, "release": self.release,
                "object_id": source.object_id, **source.extra}


class GaiaEpochAdapter:
    """Offline-ready adapter contract for Gaia DR4 epoch photometry.

    DR4 is deliberately not enabled by default: the expected release and
    archive schema are future-facing, and the full epoch product is too large
    for an implicit desktop download.  Fixtures can nevertheless be parsed
    and validated now, allowing the ingestion path to be tested before the
    external release gate opens.
    """

    release = "dr4-epoch"
    expected_release = DR4_EXPECTED_RELEASE
    required_columns = ("source_id", "time", "g_flux", "g_flux_error")

    @classmethod
    def validate_chunk(cls, rows: list[dict]) -> dict:
        valid, rejected = [], []
        for row in rows:
            if any(key not in row for key in cls.required_columns):
                rejected.append({"source_id": row.get("source_id"), "reason": "missing_column"})
                continue
            try:
                values = [float(row[key]) for key in cls.required_columns[1:]]
                if not all(np.isfinite(value) for value in values):
                    raise ValueError("non-finite epoch value")
                valid.append({
                    "source_id": str(row["source_id"]),
                    "time": values[0], "g_flux": values[1],
                    "g_flux_error": abs(values[2]),
                })
            except (TypeError, ValueError):
                rejected.append({"source_id": row.get("source_id"), "reason": "invalid_value"})
        return {
            "release": cls.release,
            "expected_release": cls.expected_release,
            "rows": valid,
            "accepted": len(valid),
            "rejected": len(rejected),
            "rejections": rejected,
            "enabled": False,
        }


def _row_to_extra(row, colnames) -> dict:
    """Convert one Gaia row into plain Python, masking missing values as None.

    Gaia uses masked columns heavily — parallax is absent for a large fraction
    of sources — and a NaN that reads as a number would corrupt downstream
    physical-consistency checks.
    """
    extra: dict = {}
    for column in COLUMNS:
        if column in ("source_id", "ra", "dec") or column not in colnames:
            continue
        value = row[column]
        try:
            import numpy as np

            if value is None or (np.ma.is_masked(value)):
                extra[column] = None
                continue
            as_float = float(value)
            extra[column] = None if np.isnan(as_float) else as_float
        except (TypeError, ValueError):
            extra[column] = str(value)
    return extra


def derived_properties(extra: dict) -> dict:
    """Quantities the ranking stage uses, computed only where inputs exist."""
    parallax = extra.get("parallax")
    parallax_error = extra.get("parallax_error")
    bp = extra.get("phot_bp_mean_mag")
    rp = extra.get("phot_rp_mean_mag")
    g = extra.get("phot_g_mean_mag")

    result: dict = {"distance_pc": None, "bp_rp": None,
                    "abs_g_mag": None, "parallax_snr": None,
                    "a_g": extra.get("a_g_val"),
                    "ebv": extra.get("ebpminrp_val")}

    if parallax is not None and parallax > 0:
        result["distance_pc"] = 1000.0 / parallax
        if g is not None:
            # Standard distance modulus; only valid for a positive parallax.
            result["abs_g_mag"] = g - 5.0 * (
                __import__("math").log10(result["distance_pc"]) - 1.0
            )
    if parallax is not None and parallax_error:
        result["parallax_snr"] = parallax / parallax_error
    if bp is not None and rp is not None:
        result["bp_rp"] = bp - rp

    return result
