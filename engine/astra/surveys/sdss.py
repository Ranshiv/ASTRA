"""SDSS optical catalogue/spectroscopy connector.

`cone_search` queries `SpecObjAll` (the spectroscopic object table), not
`PhotoObj`: an earlier version of this connector selected `plate, mjd,
fiberID` `FROM PhotoObj`, which does not have those columns -- confirmed live
against the real SkyServer SQL endpoint (`HTTP 500`, invalid column). Getting
real `plate`/`mjd`/`fiberID`/`run2d` is required to build a real spectrum
download URL, so this was a genuine bug, not a style choice.

`cone_search` also LEFT-joins `PhotoObj` on `bestObjID` for `ugriz` model
magnitudes and errors (`u/g/r/i/z`, `err_u/err_g/err_r/err_i/err_z`), closing
a real gap `photometric_calibration.py` used to document explicitly: this
connector previously returned identifiers/positions only, so SDSS pairs were
silently unavailable to that module's zero-point/color-term fit. A spectrum
with `bestObjID="0"` (no photometric counterpart -- see `photometric_match`
above) gets NULL photometry columns from the join, surfaced as `None` in
`extra`, never a fabricated value.

`fetch_spectrum`/`extract_sdss_spectrum` add real spectrum acquisition,
verified live while building this: the public, unauthenticated download URL
is `https://data.sdss.org/sas/dr17/sdss/spectro/redux/{run2d}/spectra/lite/
{plate:04d}/spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits` (confirmed by
downloading a real 168.8 KB file for a real SpecObjAll row and inspecting it
with astropy). SDSS's `COADD` table stores `loglam` (log10-Angstrom), not a
`wavelength` column, so `spectral_features.from_fits()`'s generic
`wavelength/wave/lambda` column matcher does not apply here --
`extract_sdss_spectrum` converts `loglam` itself and calls
`spectral_features.extract()` directly, reusing that module's continuum/S/N/
line statistics unchanged. The same real file also carries a `SPECOBJ` row
with SDSS's own legacy ELODIE cross-correlation stellar parameters (a real,
pipeline-computed `CLASS='STAR'`, `ELODIE_TEFF=6890.0 K` for that object) --
real spectroscopic evidence this connector surfaces, not a new ASTRA fit.
Those columns are only populated for SEGUE-era stellar spectra and are
frequently absent; a non-finite or non-positive `ELODIE_TEFF` is treated as
"not available", never a fabricated 0 K.

SDSS-V (roadmap item 24) needed NO new connector, confirmed live this
session rather than assumed: `release="dr19"` (SDSS-V's current DR) works
against this connector completely unchanged. `https://skyserver.sdss.org/
dr19/SkyServerWS/SearchTools/SqlSearch` is live, serves the identical
`SpecObjAll` schema with the same leading `#Table1` comment line
`parse_csv` already strips, and returns real `z`/`zErr`/`plate`/`mjd`/
`fiberID`/`run2d` rows. `SpecObjAll` on DR19 was also confirmed this
session to carry ONLY legacy-generation (pre-FPS, e.g. `run2d='v5_13_2'`)
spectra -- a live query for `run2d='v6_1_3'` (SDSS-V's own new fiber-
positioner-era reduction, confirmed to exist as a real directory under
`data.sdss.org/sas/dr19/spectro/boss/redux/`) returned zero rows. So
"SDSS-V support" today genuinely means "legacy SDSS spectra visible
through DR19's catalog," not new FPS-era observations -- `SpecObjAll`
doesn't expose those yet. `fetch_spectrum`'s dr17-hosted `SPECTRUM_URL`
was independently confirmed correct for a real DR19-cataloged row too
(`plate=7261, mjd=56603, fiber=458, run2d=v5_13_2` -> real `HTTP 200`):
SDSS spectrum files live at one stable, DR-independent SAS location once
published, not duplicated per release, which is also why this module never
needed a `release`-dependent `SPECTRUM_URL`. The new FPS-era files DO
exist on disk (confirmed: `dr19/spectro/boss/redux/v6_1_3/.../spec-
{fieldid}-{mjd}-{catalogid_or_coord}.fits`, a materially different naming
convention -- a long CatalogID or a coordinate string in the fiber slot,
not a small zero-padded fiber number) but are unreachable from this
connector: `SpecObjAll` carries none of those rows, so there is no
`catalogid` this module could read to build that URL. Stated as a real,
open `[GAP]` rather than guessed at.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector

DEFAULT_RELEASE = "dr18"
SQL_URL = "https://skyserver.sdss.org/{release}/SkyServerWS/SearchTools/SqlSearch"

# Verified live (see module docstring): the real, public, unauthenticated
# per-object spec-lite spectrum download contract.
SPECTRUM_URL = (
    "https://data.sdss.org/sas/dr17/sdss/spectro/redux/{run2d}/spectra/lite/"
    "{plate:04d}/spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits"
)


def parse_csv(payload: str, limit: int = 100) -> list[dict[str, str]]:
    """Parse a SkyServer CSV response.

    A real bug, found and fixed via a live check while building
    `query_spectroscopic_redshifts` for roadmap item 19: the live
    SqlSearch endpoint's CSV response is prefixed with a `#Table1` comment
    line before the real header row -- confirmed by a direct request
    against `https://skyserver.sdss.org/dr18/...`. Without stripping it,
    `csv.DictReader` treats `#Table1` itself as the field-name row, so
    every real row silently mismatches every column lookup in this file
    (`cone_search` included) and is dropped as if zero sources matched --
    a genuine "false empty result" bug that predates this session and had
    no live test to catch it. Any leading `#`-prefixed line is stripped
    (SkyServer's own convention for a comment/table-name line), not just
    this exact string, since the endpoint may return `#Table2` etc. for a
    different query shape.
    """
    lines = payload.splitlines()
    while lines and lines[0].startswith("#"):
        lines = lines[1:]
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    return [dict(row) for row in list(reader)[:limit]]


class SDSSConnector(SurveyConnector):
    name = "SDSS"
    capabilities = ("catalogue", "spectrum_metadata")
    resolution_arcsec = 1.0
    enabled_by_default = True

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        top = max(1, min(int(limit), 200))
        # The fixed query returns only identifiers/positions/photometry and
        # does not allow user-provided SQL, which keeps the connector
        # injection-safe. `PhotoObj` is LEFT-joined on `bestObjID` for
        # `ugriz` model magnitudes/errors -- the photometry closes a real
        # gap `photometric_calibration.py` documents (SDSS pairs were
        # silently unavailable to zero-point/color-term fitting because
        # this connector carried no magnitude fields at all). A spectrum
        # with `bestObjID=0` (no photometric counterpart, see the class
        # docstring above) simply gets NULL photometry columns here, the
        # same "missing, not fabricated" contract `spectrum_ready` already
        # uses for that case.
        sql = (
            f"SELECT TOP {top} s.bestObjID, s.specObjID, s.ra, s.dec, "
            "s.plate, s.mjd, s.fiberID, s.run2d, s.class, "
            "p.u, p.g, p.r, p.i, p.z, "
            "p.err_u, p.err_g, p.err_r, p.err_i, p.err_z "
            "FROM SpecObjAll s LEFT OUTER JOIN PhotoObj p ON p.objID = s.bestObjID "
            f"WHERE dbo.fDistanceArcMinEq(s.ra, s.dec, {query.ra_deg}, {query.dec_deg}) "
            f"<= {query.radius_arcsec / 60.0:.8f}"
        )
        response = netclient.get(
            SQL_URL.format(release=self.release),
            {"cmd": sql, "format": "csv"}, timeout=60, provider="sdss",
        )
        rows = parse_csv(response.text, top)
        sources: list[SourceRef] = []
        for row in rows:
            try:
                ra_deg, dec_deg = float(row["ra"]), float(row["dec"])
            except (KeyError, TypeError, ValueError):
                continue
            # bestObjID is legitimately "0" when a spectrum has no matched
            # photometric counterpart -- confirmed live. Falling back to
            # specObjID keeps the row rather than silently dropping it.
            best_object_id = str(row.get("bestObjID") or "").strip()
            photometric_match = bool(best_object_id) and best_object_id != "0"
            object_id = best_object_id if photometric_match else str(row.get("specObjID") or "")
            if not object_id:
                continue
            sources.append(SourceRef(
                survey=self.name, object_id=object_id,
                ra_deg=ra_deg, dec_deg=dec_deg,
                extra={"plate": row.get("plate"), "mjd": row.get("mjd"),
                       "fiber_id": row.get("fiberID"), "run2d": row.get("run2d"),
                       "class": row.get("class"),
                       "photometric_match": photometric_match,
                       "spectrum_ready": bool(row.get("plate")),
                       "mag_u": row.get("u"), "mag_u_error": row.get("err_u"),
                       "mag_g": row.get("g"), "mag_g_error": row.get("err_g"),
                       "mag_r": row.get("r"), "mag_r_error": row.get("err_r"),
                       "mag_i": row.get("i"), "mag_i_error": row.get("err_i"),
                       "mag_z": row.get("z"), "mag_z_error": row.get("err_z")},
            ))
        if rows and not sources:
            import logging
            logging.getLogger(__name__).warning(
                "SDSS: SkyServer returned %d row(s) but none parsed as a "
                "source -- ra/dec/bestObjID/specObjID may no longer match "
                "SpecObjAll's real columns.", len(rows))
        return sources

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        # SDSS is not a time-series connector.  A future spectrum job consumes
        # source.extra and writes spectral products, not a fake light curve.
        return []


def query_spectroscopic_redshifts(ra_deg: float, dec_deg: float, radius_arcsec: float,
                                  release: str = DEFAULT_RELEASE, limit: int = 200) -> list[SourceRef]:
    """Real spectroscopic redshifts (`SpecObjAll.z`/`zErr`) near a position
    -- for `photo_z.py`'s calibration-sample cross-match, added for roadmap
    item 19 (a new, additive function; `cone_search` above is unchanged).

    Confirmed live this session: `z`/`zErr` are real, populated columns on
    `SpecObjAll` (unlike `cone_search`'s columns, this pulls the redshift
    itself, not just the plate/mjd/fiberID needed to locate a spectrum
    file). `bestObjID` is legitimately `"0"` for some rows here too, the
    same fallback to `specObjID` `cone_search` already documents.
    """
    top = max(1, min(int(limit), 200))
    sql = (
        f"SELECT TOP {top} bestObjID, specObjID, ra, dec, z, zErr, class "
        "FROM SpecObjAll "
        f"WHERE dbo.fDistanceArcMinEq(ra, dec, {ra_deg}, {dec_deg}) "
        f"<= {radius_arcsec / 60.0:.8f} AND z > 0"
    )
    response = netclient.get(
        SQL_URL.format(release=release),
        {"cmd": sql, "format": "csv"}, timeout=60, provider="sdss",
    )
    sources: list[SourceRef] = []
    for row in parse_csv(response.text, top):
        try:
            ra_val, dec_val, z = float(row["ra"]), float(row["dec"]), float(row["z"])
        except (KeyError, TypeError, ValueError):
            continue
        best_object_id = str(row.get("bestObjID") or "").strip()
        object_id = best_object_id if best_object_id and best_object_id != "0" \
            else str(row.get("specObjID") or "")
        if not object_id:
            continue
        try:
            z_err = float(row["zErr"])
        except (KeyError, TypeError, ValueError):
            z_err = None
        sources.append(SourceRef(
            survey="SDSS", object_id=object_id, ra_deg=ra_val, dec_deg=dec_val,
            extra={"z": z, "z_err": z_err, "class": row.get("class")},
        ))
    return sources


class SdssSpectrumError(ValueError):
    """A real spectrum could not be located or downloaded for this source."""


def spectrum_url(source: SourceRef) -> str:
    """The real, verified per-object spec-lite FITS download URL.

    Raises when `source.extra` lacks any of plate/mjd/fiber_id/run2d -- e.g.
    a `SourceRef` built before `cone_search`'s fix above, or a source with no
    matched spectrum at all.
    """
    extra = source.extra or {}
    try:
        plate = int(extra["plate"])
        mjd = int(extra["mjd"])
        fiber = int(extra["fiber_id"])
        run2d = str(extra["run2d"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SdssSpectrumError(
            "source is missing plate/mjd/fiber_id/run2d; re-run cone_search "
            "with the current connector to get a spectrum-capable SourceRef"
        ) from exc
    return SPECTRUM_URL.format(run2d=run2d, plate=plate, mjd=mjd, fiber=fiber)


def fetch_spectrum(source: SourceRef, dataset_root: Path | None = None,
                   *, overwrite: bool = False) -> Path:
    """Download the real spec-lite FITS spectrum for `source`.

    Uses `netclient.download` -- the same bounded, atomic, SHA-256-checked
    transfer primitive the TESS/ZTF product pipelines already use -- rather
    than a bare `requests.get`, so a failed or interrupted transfer never
    leaves a partial FITS file at its canonical path.
    """
    from .. import config

    url = spectrum_url(source)
    extra = source.extra or {}
    plate, mjd, fiber = int(extra["plate"]), int(extra["mjd"]), int(extra["fiber_id"])
    root = dataset_root or config.PATHS.datasets
    destination = (root / "SDSS" / "spectra"
                  / f"spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits")
    if destination.exists() and not overwrite:
        return destination
    result = netclient.download(url, destination, provider="sdss", overwrite=overwrite)
    return result.path


def extract_sdss_spectrum(path: str | Path) -> dict:
    """Continuum/line features plus SDSS's own legacy stellar parameters.

    Does NOT reuse `spectral_features.from_fits()`'s generic FITS reader:
    that function matches columns named `wavelength`/`wave`/`lambda`, and
    SDSS's `COADD` table stores `loglam` (log10-Angstrom) instead -- verified
    live by downloading and inspecting a real spec-lite file. This function
    converts `loglam` itself, drops points SDSS's own `and_mask` flags as bad
    pixels, then calls the existing, unmodified `spectral_features.extract()`
    for the generic continuum/S/N/line statistics.
    """
    import hashlib

    import numpy as np
    from astropy.io import fits

    from .. import spectral_features

    source_path = Path(path).resolve()
    with fits.open(source_path, memmap=True) as hdul:
        coadd = hdul["COADD"].data
        loglam = np.asarray(coadd["loglam"], dtype=np.float64)
        flux = np.asarray(coadd["flux"], dtype=np.float64)
        ivar = np.asarray(coadd["ivar"], dtype=np.float64)
        and_mask = np.asarray(coadd["and_mask"], dtype=np.int64)

        good = (and_mask == 0)
        wavelength = 10.0 ** loglam[good]
        flux = flux[good]
        error = np.where(ivar[good] > 0, 1.0 / np.sqrt(np.clip(ivar[good], 1e-30, None)), 0.0)

        stellar_parameters = None
        if "SPECOBJ" in hdul and len(hdul["SPECOBJ"].data) > 0:
            spec_row = hdul["SPECOBJ"].data[0]
            teff = float(spec_row["ELODIE_TEFF"])
            if np.isfinite(teff) and teff > 0:
                stellar_parameters = {
                    "teff_k": teff,
                    "log_g": float(spec_row["ELODIE_LOGG"]),
                    "feh": float(spec_row["ELODIE_FEH"]),
                    "spt": str(spec_row["ELODIE_SPTYPE"]).strip(),
                    "source": "sdss_elodie_legacy_pipeline",
                }
            object_class = str(spec_row["CLASS"]).strip()
            subclass = str(spec_row["SUBCLASS"]).strip()
        else:
            object_class = subclass = None

    payload = spectral_features.extract(
        wavelength, flux, error, frame="observed",
        units="1e-17 erg/s/cm^2/Angstrom",
        source={"path": str(source_path),
               "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
    )
    payload["class"] = object_class
    payload["subclass"] = subclass
    payload["stellar_parameters"] = stellar_parameters
    return payload


def extract_sdss_line_measurements(path: str | Path) -> list[dict]:
    """SDSS's own pipeline emission/absorption-line fits (the `SPZLINE`
    extension), for roadmap item 25's "line-parameter residuals against
    released values" metric -- an additive function, `extract_sdss_spectrum`
    above is unchanged.

    Confirmed live this session by downloading and inspecting a real
    spec-lite file (`spec-0266-51630-0336.fits`): `SPZLINE` is real, present
    in the SAME file `fetch_spectrum` already downloads (no separate VAC
    query needed, unlike this codebase's earlier assumption that a
    `galSpecLine`-shaped table might require one), with columns `LINENAME`,
    `LINEWAVE` (rest wavelength), `LINEZ` (that line's own fitted redshift,
    NOT the object's global `z`), `LINESIGMA`/`LINESIGMA_ERR` (velocity
    dispersion in km/s, not Angstrom), `LINEAREA`/`LINEAREA_ERR` (integrated
    flux), `LINEEW`/`LINEEW_ERR`, `LINECONTLEVEL`, `LINECHI2`.

    A row for every line in SDSS's template list is always present, whether
    or not that line was actually detected in this spectrum -- an
    undetected/out-of-range line reports `LINEZ=LINESIGMA=LINEAREA=0.0`
    exactly (confirmed live: `Ly_alpha`/`N_V`/`C_IV` etc. for a low-z galaxy
    spectrum where those UV lines fall outside the observed window). Rows
    with `sigma_kms == 0.0 and area == 0.0` are treated as NOT FIT and
    excluded, the same "not available, not a fabricated zero" discipline
    `extract_sdss_spectrum`'s `ELODIE_TEFF` handling already established --
    a real non-detection would still report a small nonzero area/error, not
    an exact 0.0.

    `observed_wavelength_angstrom` (`LINEWAVE * (1 + LINEZ)`) and
    `sigma_angstrom` (`LINESIGMA_kms / c * observed_wavelength`) are
    provided pre-converted to the same units `line_profile.LineProfileParams`
    uses, so a caller does not need to know SDSS's own `LINESIGMA`
    velocity-space convention to compare against them.
    """
    import numpy as np
    from astropy.io import fits

    SPEED_OF_LIGHT_KMS = 299_792.458

    source_path = Path(path).resolve()
    with fits.open(source_path, memmap=True) as hdul:
        if "SPZLINE" not in hdul:
            return []
        rows = hdul["SPZLINE"].data

    measurements: list[dict] = []
    for row in rows:
        sigma_kms = float(row["LINESIGMA"])
        area = float(row["LINEAREA"])
        if sigma_kms == 0.0 and area == 0.0:
            continue
        rest_wave = float(row["LINEWAVE"])
        line_z = float(row["LINEZ"])
        observed_wave = rest_wave * (1.0 + line_z)
        measurements.append({
            "name": str(row["LINENAME"]).strip(),
            "rest_wavelength": rest_wave,
            "line_z": line_z,
            "observed_wavelength_angstrom": observed_wave,
            "sigma_kms": sigma_kms,
            "sigma_angstrom": sigma_kms / SPEED_OF_LIGHT_KMS * observed_wave,
            "area": area,
            "area_err": float(row["LINEAREA_ERR"]),
            "equivalent_width": float(row["LINEEW"]),
            "continuum_level": float(row["LINECONTLEVEL"]),
            "chi2": float(row["LINECHI2"]),
        })
    return measurements
