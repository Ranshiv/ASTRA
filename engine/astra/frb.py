"""Optional, cached fast-radio-burst coincidence check for candidate review.

Shaped like `catalogs.py` and `gw.py`: this is evidence, not a dependency of
discovery, and candidate generation never calls it. Unlike a GW event, most
CHIME/FRB bursts genuinely fit the point-plus-error-radius shape
`catalogs.CatalogQuery` already uses -- the base catalogue reports `ra`,
`ra_err`, `dec`, `dec_err` directly per burst (several arcminutes typical),
not a probability map needing reconstruction. Only a minority of bursts
(baseband-localized) additionally carry a precomputed sparse HEALPix
confidence-level map, and even that map is already finished -- unlike the GW
posterior-samples case, nothing here needs histogram-binning.

This is still not a `SurveyConnector`: a burst detection is a one-off event
review, not a survey with discoverable point sources to `cone_search` and
fetch light curves for.

The officially documented client, checked directly while planning this
module, turned out to be dead: `cfod` (CHIME/FRB Open Data's own package)
depends on `healpy`, which fails to build on Windows (confirmed with a real
install attempt), and its hardcoded catalogue download URL -- a Google Cloud
Storage bucket -- returned 403 "billing account ... disabled" on a direct
check. `cfod`'s last release was 2021; it is abandoned. The current data
lives on CADC/CANFAR as a DOI-registered public dataset (Catalog 2, DOI
10.11570/25.0066), whose browsable web listing was confirmed reachable, but
whose machine-readable VOSpace backend (`ws-uv.canfar.net`, including the
IVOA service registry every other CADC endpoint depends on for discovery)
was returning 503 across every check made while building this module --
a real infrastructure outage on CADC's side, not a wrong URL: the plain
website worked throughout. `fetch_burst_catalog` is therefore written
against the documented VOSpace convention and degrades to "unavailable"
rather than crashing on a failed request, exactly like every other lookup in
this module -- but it has NOT been exercised against a live, healthy CADC
service. Re-verify the download path once the outage clears, the same
"not live-validated yet" caveat already carried by the Chandra/Swift/XMM/
DES/Hubble/JWST connectors.

Deliberately NOT wired into scoring.WEIGHTS/combine(), for the identical
reason gw.py gives: this is unvalidated evidence with no track record in
this project, and adopting it would bump scoring.WEIGHT_VERSION and change
what every historically stored candidate score means. It lands as visible
evidence (`Candidate.frb`, `explanation["frb_coincidence"]`), not a ranking
change.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import config, netclient

# CADC/CANFAR VOSpace convention for a public DOI-registered dataset. Not
# live-validated -- see the module docstring. Re-check against the live
# service (the browsable listing at
# https://www.canfar.net/storage/list/AstroDataCitationDOI/CISTI.CANFAR/
# 25.0066/data was reachable; the machine API was not, during an outage) and
# correct this if the real shape differs.
CADC_VAULT_BASE = "https://ws-uv.canfar.net/vault"
DEFAULT_DOI_PATH = "AstroDataCitationDOI/CISTI.CANFAR/25.0066/data"
CATALOG_FILENAME = "chimefrbcat2.csv"

# New catalogue VERSIONS are infrequent (roughly annual), unlike GW's
# day-scale new-event cadence, so this cache lives longer than gw.py's.
CATALOG_TTL_DAYS = 7

# ra_err/dec_err are reported uncertainties, not explicit confidence-level
# radii, so "inside the ellipse" is a documented, tunable sigma multiple
# rather than a claimed confidence percentage -- do not read this as a 90%
# credible region the way gw.py's HEALPix path can.
DEFAULT_SIGMA_THRESHOLD = 3.0
DEFAULT_WINDOW_DAYS = 1.0
DEFAULT_NSIDE = 4096  # matches CHIME/FRB's published sparse localization maps


class FrbError(RuntimeError):
    """An FRB lookup could not produce a scientifically usable answer."""


@dataclass
class FrbBurst:
    tns_name: str
    repeater_name: str
    ra_deg: float
    ra_err_deg: float
    dec_deg: float
    dec_err_deg: float
    mjd_400: float
    localization_id: str | None = None

    def to_dict(self) -> dict:
        return {"tns_name": self.tns_name, "repeater_name": self.repeater_name,
               "ra_deg": self.ra_deg, "ra_err_deg": self.ra_err_deg,
               "dec_deg": self.dec_deg, "dec_err_deg": self.dec_err_deg,
               "mjd_400": self.mjd_400, "localization_id": self.localization_id}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _catalog_cache_path(root: Path) -> Path:
    return root / "chimefrb" / "catalog2.json"


def _parse_catalog_csv(text: str) -> list[FrbBurst]:
    """Defensive CSV parsing: a malformed or missing row is skipped, not fatal.

    Mirrors the per-row try/except discipline every survey connector in this
    codebase already uses (see swift.py/xmm.py's cone_search) -- a field-name
    mismatch against the real CADC file (not live-validated yet) degrades to
    "fewer bursts parsed", not a crash.
    """
    bursts: list[FrbBurst] = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            excluded = str(row.get("excluded_flag", "0")).strip() in ("1", "True", "true")
            if excluded:
                continue
            bursts.append(FrbBurst(
                tns_name=str(row["tns_name"]),
                repeater_name=str(row.get("repeater_name") or ""),
                ra_deg=float(row["ra"]), ra_err_deg=float(row["ra_err"]),
                dec_deg=float(row["dec"]), dec_err_deg=float(row["dec_err"]),
                mjd_400=float(row["mjd_400"]),
                localization_id=row.get("localization_id") or None,
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return bursts


def fetch_burst_catalog(refresh: bool = False, offline: bool = False,
                        root: Path | None = None) -> list[FrbBurst]:
    """List published CHIME/FRB bursts, cached for a week.

    A day-scale TTL like gw.py's would be wasteful here: new catalogue
    releases are roughly annual, not continuous.
    """
    root = root or config.PATHS.cache
    cache_path = _catalog_cache_path(root)

    if not refresh and cache_path.exists():
        age = _now() - datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        if age < timedelta(days=CATALOG_TTL_DAYS):
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [FrbBurst(**row) for row in payload["bursts"]]

    if offline:
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [FrbBurst(**row) for row in payload["bursts"]]
        return []

    url = f"{CADC_VAULT_BASE}/{DEFAULT_DOI_PATH}/{CATALOG_FILENAME}"
    try:
        response = netclient.get(url, {}, timeout=60, provider="cadc")
        bursts = _parse_catalog_csv(response.text)
    except Exception as exc:  # noqa: BLE001 - unavailable is data, not fatal
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [FrbBurst(**row) for row in payload["bursts"]]
        raise FrbError(f"could not fetch the CHIME/FRB catalogue: {exc}") from exc

    _atomic_json_write(cache_path, {
        "fetched_utc": _now().isoformat(timespec="seconds"),
        "bursts": [burst.to_dict() for burst in bursts],
    })
    return bursts


def within_error_ellipse(candidate_ra_deg: float, candidate_dec_deg: float,
                         burst: FrbBurst, sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD
                         ) -> tuple[bool, float]:
    """Normalised offset from a burst's reported position, in error-widths.

    RA offset is scaled by cos(dec), matching the published methodology's own
    convention ("R.A. errors scaled by cos(dec)"). Returns (inside, offset):
    `offset` is a normalised elliptical distance, not a confidence-level
    fraction -- 1.0 means exactly one reported error-width away, not "the 68%
    region". A burst's `ra_err`/`dec_err` of 0 would make this undefined;
    such a row already failed `_parse_catalog_csv`'s ValueError guard only if
    non-numeric, so a genuine zero is floored to avoid a division error.
    """
    delta_ra_deg = (candidate_ra_deg - burst.ra_deg) * np.cos(np.radians(burst.dec_deg))
    delta_dec_deg = candidate_dec_deg - burst.dec_deg
    ra_err = max(burst.ra_err_deg, 1e-6)
    dec_err = max(burst.dec_err_deg, 1e-6)
    offset = float(np.sqrt((delta_ra_deg / ra_err) ** 2 + (delta_dec_deg / dec_err) ** 2))
    return offset <= sigma_threshold, offset


def _localization_cache_path(burst: FrbBurst, root: Path) -> Path | None:
    if not burst.localization_id:
        return None
    return root / "chimefrb" / f"loc_{burst.localization_id}.h5"


def localization_membership(burst: FrbBurst, ra_deg: float, dec_deg: float,
                            *, nside: int = DEFAULT_NSIDE, root: Path | None = None
                            ) -> dict | None:
    """Precise sparse-HEALPix membership for a baseband-localized burst.

    None for a burst with no baseband localization product -- the common
    case, since most bursts have only the ra_err/dec_err ellipse -- or one
    whose map could not be fetched. Unlike gw.py's skymap, this map is
    already the finished confidence-level product (`ipix`/`CL` pairs); no
    binning or reconstruction is needed, only a pixel lookup.
    """
    root = root or config.PATHS.cache
    path = _localization_cache_path(burst, root)
    if path is None:
        return None

    if not path.exists():
        url = f"{CADC_VAULT_BASE}/{DEFAULT_DOI_PATH}/localization/{burst.localization_id}.h5"
        try:
            netclient.download(url, path, timeout=120.0, provider="cadc", overwrite=False)
        except Exception:  # noqa: BLE001 - a failed download is "no map", not a crash
            return None
        if not path.exists():
            return None

    import astropy.units as u
    import h5py
    from astropy_healpix import HEALPix

    try:
        with h5py.File(path, "r") as handle:
            ipix = handle["ipix"][:]
            confidence = handle["CL"][:]
    except Exception:  # noqa: BLE001 - a corrupt/unexpected file is "no map"
        return None
    if len(ipix) == 0:
        return None

    healpix = HEALPix(nside=nside, order="nested")
    pixel = int(healpix.lonlat_to_healpix(ra_deg * u.deg, dec_deg * u.deg))
    match = np.where(ipix == pixel)[0]
    if len(match) == 0:
        # Outside every pixel this sparse map reports at all -- the least
        # confident region possible, not "unknown".
        return {"confidence_level": 1.0, "in_90pct_region": False}

    level = float(confidence[match[0]])
    return {"confidence_level": level, "in_90pct_region": level <= 0.90}


def _event_mjd_to_jd(mjd: float) -> float:
    return mjd + 2_400_000.5


def _candidate_time_bounds_jd(path: Path) -> tuple[float, float] | None:
    """A stored candidate's observation span, as approximate JD bounds.

    Identical approach to gw.py's -- day-scale windows make the JD/HJD/BJD/
    UTC distinctions (at most ~8 minutes) immaterial here too.
    """
    from . import store

    try:
        curve = store.read_curve(path)
    except Exception:  # noqa: BLE001 - an unreadable curve has no time bounds
        return None
    tidy = curve.dropna()
    if len(tidy) == 0:
        return None
    time_min, time_max = float(tidy.time.min()), float(tidy.time.max())
    if curve.time_system == "MJD_UTC":
        time_min += 2_400_000.5
        time_max += 2_400_000.5
    return time_min, time_max


def _temporally_coincident(time_bounds: tuple[float, float], burst: FrbBurst,
                           window_days: float) -> bool:
    burst_jd = _event_mjd_to_jd(burst.mjd_400)
    time_min, time_max = time_bounds
    return (time_min - window_days) <= burst_jd <= (time_max + window_days)


def enrich_candidate_frb(candidate: Any, bursts: list[FrbBurst], *,
                         window_days: float = DEFAULT_WINDOW_DAYS,
                         sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD,
                         nside: int = DEFAULT_NSIDE, root: Path | None = None
                         ) -> dict:
    """One candidate's FRB coincidence evidence.

    Time filter first, on purpose: most candidate/burst pairs fail on time
    alone, and there is no reason to check position (let alone fetch a
    localization map) to learn that.
    """
    time_bounds = _candidate_time_bounds_jd(Path(candidate.path))
    if time_bounds is None:
        return {"checked_bursts": 0, "coincident": [], "state": "unavailable",
                "reason": "candidate light curve unreadable or empty"}

    coincident_bursts = [burst for burst in bursts
                         if _temporally_coincident(time_bounds, burst, window_days)]

    coincident: list[dict] = []
    for burst in coincident_bursts:
        inside, offset = within_error_ellipse(
            candidate.ra_deg, candidate.dec_deg, burst, sigma_threshold)
        if not inside:
            continue
        entry = {"burst": burst.tns_name, "repeater_name": burst.repeater_name,
                 "mjd_400": burst.mjd_400, "sigma_offset": offset,
                 "sigma_threshold": sigma_threshold, "position_source": "ellipse"}
        localization = localization_membership(burst, candidate.ra_deg,
                                               candidate.dec_deg, nside=nside, root=root)
        if localization is not None:
            entry.update(localization)
            entry["position_source"] = "healpix"
        coincident.append(entry)

    state = "match" if coincident else "no_match"
    return {"checked_bursts": len(bursts), "temporally_coincident": len(coincident_bursts),
           "coincident": coincident, "state": state, "window_days": window_days,
           "sigma_threshold": sigma_threshold}


def _apply_to_candidate(candidate: Any, evidence: dict) -> None:
    """Attach FRB evidence without moving the composite score -- see module docstring."""
    candidate.frb = evidence
    if evidence.get("coincident"):
        names = ", ".join(item["burst"] for item in evidence["coincident"])
        candidate.explanation["frb_coincidence"] = (
            f"Temporally and spatially coincident with FRB {names}."
        )
        actions = [a for a in candidate.explanation.get("recommended_actions", [])
                  if "fast radio burst" not in a.lower()]
        actions.append(
            f"Check for a fast radio burst counterpart near {names}; "
            "this coincidence is not yet part of the ranking score.")
        candidate.explanation["recommended_actions"] = actions
    elif evidence.get("state") == "unavailable":
        candidate.explanation["frb_coincidence"] = (
            "FRB coincidence not checked: candidate light curve unreadable.")
    else:
        candidate.explanation["frb_coincidence"] = (
            f"No FRB coincidence in {evidence.get('checked_bursts', 0)} checked bursts.")


def top_k_counterpart_recall(queries: list[dict], bursts: list[FrbBurst], *, k: int = 3,
                             sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD) -> dict:
    """Validate that the true burst counterpart ranks within the top-k candidates.

    Each query is `{"ra_deg", "dec_deg", "true_tns_name"}` -- a position with
    a known true counterpart (from injection or a labelled review set). This
    measures ranking quality of `within_error_ellipse`'s offset, not just
    whether the closest match happens to be correct: a crowded field can push
    the true counterpart to rank 2 or 3 while still being a usable candidate
    list for a researcher to check by eye.

    This is validation of the evidence-gathering machinery, mirroring
    `evaluate.score_method`'s precision@k/recall@k shape -- it does not by
    itself justify wiring FRB coincidence into scoring.WEIGHTS.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    total = 0
    hits = 0
    per_query: list[dict] = []
    for query in queries:
        true_name = query.get("true_tns_name")
        if true_name is None:
            continue
        ra = float(query["ra_deg"])
        dec = float(query["dec_deg"])
        total += 1
        ranked = sorted(
            ((burst.tns_name, within_error_ellipse(ra, dec, burst, sigma_threshold)[1])
             for burst in bursts),
            key=lambda item: item[1],
        )
        top_names = [name for name, _ in ranked[:k]]
        hit = true_name in top_names
        hits += int(hit)
        rank = next((i for i, name in enumerate(top_names, start=1) if name == true_name), None)
        per_query.append({"true_tns_name": true_name, "hit": hit, "rank": rank})
    return {
        "k": k, "queries": total, "hits": hits,
        "recall_at_k": (hits / total) if total else float("nan"),
        "per_query": per_query,
    }


def credible_region_containment(*, nside: int = DEFAULT_NSIDE, trials: int = 500,
                                levels: tuple[float, ...] = (0.5, 0.9),
                                sigma_deg: float = 0.1, seed: int = 42,
                                root: Path | None = None) -> dict:
    """Monte-Carlo check that reported credible levels mean what they claim.

    Builds a synthetic sparse localization map the same way a real
    baseband-localized burst would carry one -- a finished ipix/CL product,
    not a reconstruction -- by histogram-binning samples from a known 2D
    Gaussian (same discipline `gw.build_skymap_from_samples` already uses,
    and for the identical reason: no bandwidth-selection algorithm to get
    wrong). It then draws many TRUE positions from that SAME Gaussian and
    checks the empirical fraction `localization_membership` reports inside
    each nominal credible level. A well-calibrated 90% region should contain
    the truth in ~90% of trials, not 60% or 99% -- this turns the one-off
    manual spot-check already described in docs/LIMITATIONS.md into a
    repeatable, quantitative test.
    """
    import h5py
    import astropy.units as u
    from astropy_healpix import HEALPix

    root = root or config.PATHS.cache
    rng = np.random.default_rng(seed)
    center_ra, center_dec = 180.0, 10.0
    healpix = HEALPix(nside=nside, order="nested")

    ra_sigma_deg = sigma_deg / np.cos(np.radians(center_dec))
    sample_ra = rng.normal(center_ra, ra_sigma_deg, 200_000)
    sample_dec = rng.normal(center_dec, sigma_deg, 200_000)
    sample_pixels = healpix.lonlat_to_healpix(sample_ra * u.deg, sample_dec * u.deg)
    unique_pixels, counts = np.unique(np.asarray(sample_pixels), return_counts=True)
    probability = counts / counts.sum()
    order = np.argsort(-probability)
    cumulative = np.cumsum(probability[order])
    credible_level = np.empty_like(cumulative)
    credible_level[order] = cumulative

    burst = FrbBurst(tns_name="SYNTHETIC_CONTAINMENT_CHECK", repeater_name="",
                     ra_deg=center_ra, ra_err_deg=0.05, dec_deg=center_dec,
                     dec_err_deg=0.05, mjd_400=58800.0,
                     localization_id="containment_check")
    map_path = _localization_cache_path(burst, root)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(map_path, "w") as handle:
            handle.create_dataset("ipix", data=unique_pixels.astype(np.int64))
            handle.create_dataset("CL", data=credible_level.astype(np.float32))

        true_ra = rng.normal(center_ra, ra_sigma_deg, trials)
        true_dec = rng.normal(center_dec, sigma_deg, trials)

        contained = {level: 0 for level in levels}
        for ra, dec in zip(true_ra, true_dec):
            result = localization_membership(burst, float(ra), float(dec),
                                             nside=nside, root=root)
            observed = result["confidence_level"] if result is not None else 1.0
            for level in levels:
                if observed <= level:
                    contained[level] += 1
    finally:
        map_path.unlink(missing_ok=True)

    return {
        "trials": trials, "nominal_levels": list(levels),
        "empirical_containment": {level: contained[level] / trials for level in levels},
    }


def enrich_candidates_frb(name: str = "default", *,
                          window_days: float = DEFAULT_WINDOW_DAYS,
                          sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD,
                          nside: int = DEFAULT_NSIDE, root: Path | None = None,
                          refresh: bool = False, offline: bool = False) -> dict:
    """Enrich an existing candidate file with FRB coincidence evidence."""
    from . import candidates

    root = root or config.PATHS.projects
    cache_root = config.PATHS.cache
    built = candidates.load(name, root)
    bursts = fetch_burst_catalog(refresh=refresh, offline=offline, root=cache_root)

    counts: dict[str, int] = {"match": 0, "no_match": 0, "unavailable": 0}
    for candidate in built:
        evidence = enrich_candidate_frb(candidate, bursts, window_days=window_days,
                                        sigma_threshold=sigma_threshold, nside=nside,
                                        root=cache_root)
        _apply_to_candidate(candidate, evidence)
        counts[evidence["state"]] = counts.get(evidence["state"], 0) + 1

    candidates.save(built, name, root)
    return {"bursts_checked": len(bursts), "candidates": len(built), "counts": counts}
