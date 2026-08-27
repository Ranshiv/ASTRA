"""Gaia — astrometry and mean photometry.

Gaia is a catalogue connector, not a time-domain one: the main table gives one
row per source, so `fetch_light_curves` returns nothing by design. Its value to
ASTRA is `fetch_source_record`, which supplies the parallax, proper motion and
colour used for the physical-inconsistency term in the candidate score
(plan section 16) and for the stellar characterisation in section 20.

`query_eclipsing_binary`/`query_astrophysical_parameters` (added for roadmap
item 17, `eclipsing_binary_dimensions.py`) query two further DR3 tables via
`astroquery.gaia.Gaia.launch_job`, the same client this file's `cone_search`
already uses. Both tables' real column names were confirmed live this
session (2026-08-24) via `tap_schema.columns` and a real query, not assumed
from documentation: `gaiadr3.vari_eclipsing_binary` has no direct `period`
column -- only `frequency` (unit `1/d`, so `period_days = 1.0 / frequency`)
plus `derived_primary_ecl_phase/duration/depth` and the `derived_secondary_*`
equivalents; `gaiadr3.astrophysical_parameters` has `teff_gspphot`/
`radius_gspphot` (photometric estimates) and `teff_msc1`/`teff_msc2` (a
two-component "multiple star classifier" solution, present only for a
minority of sources) alongside FLAME's `radius_flame`/`mass_flame`.
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
    # gaia_source has no direct "phot_*_mag_error" column -- the archive
    # publishes flux and flux_error, and the standard magnitude-error
    # formula (see `magnitude_error_from_flux`) converts one to the other.
    # Needed for photometric_calibration.py's zero-point/color-term fit,
    # which requires per-band uncertainties to weight matched pairs.
    "phot_g_mean_flux", "phot_g_mean_flux_error",
    "phot_bp_mean_flux", "phot_bp_mean_flux_error",
    "phot_rp_mean_flux", "phot_rp_mean_flux_error",
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

    `BAND_SCHEMAS` maps each supported band family to its per-row value
    columns.  Gaia has not published a DR4 epoch table schema, so every
    column name here (including the existing G-band ones) is a documented,
    unconfirmed placeholder -- revisit once a real (or even draft) data model
    exists, exactly as the module this class lives in already notes.  RVS
    reports a radial velocity rather than a flux, so its column is a
    dimensionally different quantity from BP/RP/G's; `validate_chunk` treats
    all four uniformly as "one numeric value plus its error" for validation
    purposes only, not as physically equivalent measurements.
    """

    release = "dr4-epoch"
    expected_release = DR4_EXPECTED_RELEASE
    BAND_SCHEMAS: dict[str, tuple[str, str]] = {
        "g": ("g_flux", "g_flux_error"),
        "bp": ("bp_flux", "bp_flux_error"),
        "rp": ("rp_flux", "rp_flux_error"),
        "rvs": ("rv", "rv_error"),
    }
    DEFAULT_BANDS = ("g",)
    # Preserved for callers (and tests) that still read the pre-multi-band
    # contract directly; always equal to columns_for_bands(DEFAULT_BANDS).
    required_columns = ("source_id", "time", "g_flux", "g_flux_error")

    @classmethod
    def columns_for_bands(cls, bands: tuple[str, ...]) -> tuple[str, ...]:
        if not bands:
            raise ValueError("at least one band must be selected")
        unknown = [band for band in bands if band not in cls.BAND_SCHEMAS]
        if unknown:
            raise ValueError(f"unknown Gaia epoch band(s): {unknown}")
        columns = ["source_id", "time"]
        for band in bands:
            columns.extend(cls.BAND_SCHEMAS[band])
        return tuple(columns)

    @classmethod
    def validate_chunk(cls, rows: list[dict], bands: tuple[str, ...] = DEFAULT_BANDS) -> dict:
        columns = cls.columns_for_bands(bands)
        valid, rejected = [], []
        for row in rows:
            if any(key not in row for key in columns):
                rejected.append({"source_id": row.get("source_id"), "reason": "missing_column"})
                continue
            try:
                time_value = float(row["time"])
                band_values: dict[str, tuple[float, float]] = {}
                for band in bands:
                    flux_column, error_column = cls.BAND_SCHEMAS[band]
                    band_values[band] = (float(row[flux_column]), float(row[error_column]))
                all_numbers = [time_value, *(value for pair in band_values.values() for value in pair)]
                if not all(np.isfinite(value) for value in all_numbers):
                    raise ValueError("non-finite epoch value")
                accepted_row = {"source_id": str(row["source_id"]), "time": time_value}
                for band, (value, error) in band_values.items():
                    flux_column, error_column = cls.BAND_SCHEMAS[band]
                    accepted_row[flux_column] = value
                    accepted_row[error_column] = abs(error)
                valid.append(accepted_row)
            except (TypeError, ValueError):
                rejected.append({"source_id": row.get("source_id"), "reason": "invalid_value"})
        return {
            "release": cls.release,
            "expected_release": cls.expected_release,
            "bands": list(bands),
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


MAG_ERROR_COEFFICIENT = 2.5 / np.log(10.0)


def magnitude_error_from_flux(flux: float | None, flux_error: float | None) -> float | None:
    """Standard flux-ratio approximation of a magnitude uncertainty.

    gaia_source has no direct per-band magnitude-error column; ESA's own
    documentation derives one this way from the published flux/flux_error
    pair. Returns None rather than raising when either input is missing or
    the flux is non-positive (a magnitude is undefined there), matching
    this codebase's "unavailable, not fabricated" convention for optional
    photometric quantities.
    """
    if flux is None or flux_error is None:
        return None
    try:
        flux = float(flux)
        flux_error = float(flux_error)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(flux) and np.isfinite(flux_error)) or flux <= 0:
        return None
    return float(MAG_ERROR_COEFFICIENT * (flux_error / flux))


def photometric_errors(extra: dict) -> dict:
    """Per-band magnitude uncertainties derived from Gaia's flux columns."""
    return {
        "phot_g_mean_mag_error": magnitude_error_from_flux(
            extra.get("phot_g_mean_flux"), extra.get("phot_g_mean_flux_error")),
        "phot_bp_mean_mag_error": magnitude_error_from_flux(
            extra.get("phot_bp_mean_flux"), extra.get("phot_bp_mean_flux_error")),
        "phot_rp_mean_mag_error": magnitude_error_from_flux(
            extra.get("phot_rp_mean_flux"), extra.get("phot_rp_mean_flux_error")),
    }


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


def _validated_source_id(source_id: int | str) -> int:
    try:
        return int(source_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"source_id must be an integer, got {source_id!r}") from exc


def query_eclipsing_binary(source_id: int | str) -> dict | None:
    """Gaia DR3's own eclipsing-binary geometric-model solution for one
    source, for cross-checking a light-curve fit's period/eclipse depths.

    Returns `None` when the source has no `vari_eclipsing_binary` row (most
    sources don't -- this is a targeted classification table, not a
    catalogue of every variable), never a fabricated zero-filled record.
    """
    from astroquery.gaia import Gaia

    sid = _validated_source_id(source_id)
    job = Gaia.launch_job(
        "SELECT source_id, frequency, derived_primary_ecl_phase, "
        "derived_primary_ecl_duration, derived_primary_ecl_depth, "
        "derived_secondary_ecl_phase, derived_secondary_ecl_duration, "
        "derived_secondary_ecl_depth FROM gaiadr3.vari_eclipsing_binary "
        f"WHERE source_id = {sid}")
    table = job.get_results()
    if table is None or len(table) == 0:
        return None
    row = table[0]
    frequency = float(row["frequency"])
    if not np.isfinite(frequency) or frequency <= 0:
        return None
    return {
        "source_id": sid, "period_days": 1.0 / frequency,
        "primary_eclipse_phase": float(row["derived_primary_ecl_phase"]),
        "primary_eclipse_duration_phase": float(row["derived_primary_ecl_duration"]),
        "primary_eclipse_depth": float(row["derived_primary_ecl_depth"]),
        "secondary_eclipse_phase": float(row["derived_secondary_ecl_phase"]),
        "secondary_eclipse_duration_phase": float(row["derived_secondary_ecl_duration"]),
        "secondary_eclipse_depth": float(row["derived_secondary_ecl_depth"]),
    }


def query_astrophysical_parameters(source_id: int | str) -> dict | None:
    """Gaia DR3's photometric (`*_gspphot`) and FLAME (`*_flame`) stellar
    parameter estimates for one source -- the physical-radius anchor
    `eclipsing_binary_dimensions.anchor_physical_radius` prefers.

    Returns `None` when the source has no `astrophysical_parameters` row;
    individual fields within a returned row may still be `None` when Gaia's
    own pipeline did not populate them for that source (never fabricated).
    """
    from astroquery.gaia import Gaia

    sid = _validated_source_id(source_id)
    job = Gaia.launch_job(
        "SELECT source_id, teff_gspphot, radius_gspphot, mass_flame, "
        "radius_flame FROM gaiadr3.astrophysical_parameters "
        f"WHERE source_id = {sid}")
    table = job.get_results()
    if table is None or len(table) == 0:
        return None
    row = table[0]

    def _optional_float(column: str) -> float | None:
        value = row[column]
        if value is None or np.ma.is_masked(value):
            return None
        as_float = float(value)
        return as_float if np.isfinite(as_float) else None

    return {
        "source_id": sid,
        "teff_gspphot": _optional_float("teff_gspphot"),
        "radius_gspphot": _optional_float("radius_gspphot"),
        "mass_flame": _optional_float("mass_flame"),
        "radius_flame": _optional_float("radius_flame"),
    }


def query_extinction_estimate(source_id: int | str) -> dict | None:
    """Real astrometry plus Gaia DR3's own single-star photometric
    extinction estimate (`ag_gspphot`, `gaiadr3.astrophysical_parameters`,
    confirmed live this session via `tap_schema.columns`) for one source
    -- an independent, real reference `dust_3d.extinction_residual_vs_
    reference` can cross-check the 3-D dust-map marginalization against,
    the same "diagnostic cross-check, never a correction" restraint every
    other `*_residuals`/`compare_to_published` function in this codebase
    already applies. `ag_gspphot` is Gaia's own G-band extinction
    estimate (not exactly A0(550nm)), so a real comparison is expected to
    agree only roughly, not exactly -- two different real methods for the
    same physical quantity along the same sightline.
    """
    from astroquery.gaia import Gaia

    sid = _validated_source_id(source_id)
    job = Gaia.launch_job(
        "SELECT gs.source_id, gs.ra, gs.dec, gs.parallax, gs.parallax_error, "
        "ap.ag_gspphot FROM gaiadr3.gaia_source gs "
        "JOIN gaiadr3.astrophysical_parameters ap ON gs.source_id = ap.source_id "
        f"WHERE gs.source_id = {sid}")
    table = job.get_results()
    if table is None or len(table) == 0:
        return None
    row = table[0]

    def _optional_float(column: str) -> float | None:
        value = row[column]
        if value is None or np.ma.is_masked(value):
            return None
        as_float = float(value)
        return as_float if np.isfinite(as_float) else None

    parallax_mas = _optional_float("parallax")
    if parallax_mas is None or parallax_mas <= 0:
        return None
    return {
        "source_id": sid, "ra_deg": float(row["ra"]), "dec_deg": float(row["dec"]),
        "parallax_mas": parallax_mas, "parallax_error_mas": _optional_float("parallax_error"),
        "ag_gspphot_mag": _optional_float("ag_gspphot"),
    }
