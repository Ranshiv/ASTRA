"""Optional, cached gravitational-wave coincidence check for candidate review.

Shaped like `catalogs.py` deliberately: this is evidence, not a dependency of
discovery, and candidate generation never calls it -- a researcher explicitly
runs enrichment after candidates exist. But it is NOT a `SurveyConnector`.
A GW event has no single position to build a `ConeQuery` from; its position
is a sky-localization probability map, frequently multi-modal, covering tens
to thousands of square degrees. Forcing that into `cone_search(ConeQuery) ->
list[SourceRef]` would either lie about a position or force a false uniform-
disc assumption onto a genuinely non-uniform, sometimes bimodal distribution
-- and `crossmatch.match_catalogs`'s unindexed O(n*m) point-radius search is
computationally unsound at that sky-area scale regardless.

The premise that GWOSC hosts a ready-made skymap FITS was checked directly
against the live service while planning this module, and was wrong: GWOSC's
event API (structured v2, legacy eventapi, and a full key enumeration of a
real event record) carries no RA/Dec, no localization area, and no skymap
file -- only physical/detection parameters and strain data. GraceDB, the
usual skymap host, requires authentication for file access (confirmed with a
direct 401) -- the same credential wall that blocks the Rubin/LSST connector.

What IS genuinely public and unauthenticated is the parameter-estimation
posterior-samples file GWOSC links to for events with a public PE release
(a `dcc.ligo.org/public/...` URL). Its `right_ascension`/`declination`
columns are exactly what a sample-based skymap needs, so this module builds
one itself by histogram-binning those samples into HEALPix pixels --
deliberately not a smoothed KDE (no bandwidth-selection algorithm to get
wrong) and deliberately not `healpy` (no native Windows wheel; `astropy_healpix`
has one and does everything this module needs).

Not every event has a public PE release; `skymap_path` returns None rather
than raising for those, and that must read as "position not available", not
"no coincidence". And some releases hold sky position FIXED to a known
electromagnetic counterpart rather than sampling it (GW170817's public
posterior has right_ascension/declination std of essentially zero, because
that re-analysis conditioned on the already-known host galaxy) -- reporting
that as an ordinary GW-derived localization would silently claim an
impossibly precise position for a check that is trivial by construction for
that one event, so it is flagged as `position_source: "em_counterpart_fixed"`
rather than presented as a normal skymap.

Deliberately NOT wired into scoring.WEIGHTS/combine() in this version. Doing
so would bump scoring.WEIGHT_VERSION and change what every historically
stored candidate score means, for evidence with no track record yet in this
project -- the same restraint already applied to the calibrated artifact
weights, which are measured and built but not adopted into the production
score for the same reason. GW coincidence lands as visible evidence
(`Candidate.gw`, `explanation["gw_coincidence"]`), not a ranking change.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import config, netclient

EVENT_CATALOG_TTL_DAYS = 1
DEFAULT_CATALOG = "GWTC-1-confident"
DEFAULT_WINDOW_DAYS = 30.0
DEFAULT_NSIDE = 64
# A public posterior this flat is conditioned on a known position (an EM
# counterpart), not sampling sky location -- see the module docstring.
_FIXED_POSITION_STD_DEG = 1e-6


class GwError(RuntimeError):
    """A GW lookup could not produce a scientifically usable answer."""


@dataclass
class GwEvent:
    name: str
    catalog: str
    gps_time: float

    def to_dict(self) -> dict:
        return {"name": self.name, "catalog": self.catalog, "gps_time": self.gps_time}


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


def _events_cache_path(catalog: str, root: Path) -> Path:
    safe = catalog.replace("/", "_")
    return root / "gwosc" / f"events_{safe}.json"


def fetch_event_catalog(catalog: str = DEFAULT_CATALOG, *, refresh: bool = False,
                        offline: bool = False, root: Path | None = None
                        ) -> list[GwEvent]:
    """List published events in one GWOSC catalog, cached for a day.

    A day-scale TTL is deliberate: new events are published periodically, not
    continuously, so this need not hit the API on every call, but a
    researcher should not wait longer than a day to notice a new one.
    """
    root = root or config.PATHS.cache
    cache_path = _events_cache_path(catalog, root)

    if not refresh and cache_path.exists():
        age = _now() - datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        if age < timedelta(days=EVENT_CATALOG_TTL_DAYS):
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [GwEvent(**row) for row in payload["events"]]

    if offline:
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [GwEvent(**row) for row in payload["events"]]
        return []

    from gwosc import datasets

    try:
        names = datasets.find_datasets(type="event", catalog=catalog)
        events = [GwEvent(name=name, catalog=catalog,
                          gps_time=float(datasets.event_gps(name, catalog=catalog)))
                 for name in names]
    except Exception as exc:  # noqa: BLE001 - a failed listing is unavailable, not fatal
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [GwEvent(**row) for row in payload["events"]]
        raise GwError(f"could not fetch the GWOSC event catalog: {exc}") from exc

    _atomic_json_write(cache_path, {
        "catalog": catalog, "fetched_utc": _now().isoformat(timespec="seconds"),
        "events": [event.to_dict() for event in events],
    })
    return events


def resolve_pe_data_url(event: str, catalog: str, *, root: Path | None = None
                        ) -> str | None:
    """The public posterior-samples file URL for one event, if it has one.

    Not every event has a public parameter-estimation release -- GW-only
    detections without deep PE exist -- so None is a normal answer, not a
    failure. Cached indefinitely: a published release does not change.
    """
    root = root or config.PATHS.cache
    cache_path = root / "gwosc" / f"pe_url_{event}_{catalog}.json".replace("/", "_")
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached.get("data_url")

    from gwosc.api import v2

    try:
        version = v2.fetch_event_version(event, catalog=catalog)
        parameters_url = version.get("parameters_url")
        if not parameters_url:
            data_url = None
        else:
            response = netclient.get(parameters_url, {}, timeout=30, provider="gwosc")
            products = response.json().get("results", [])
            preferred = next((p for p in products if p.get("is_preferred")), None)
            data_url = (preferred or {}).get("data_url") or None
            data_url = data_url or None
    except Exception:  # noqa: BLE001 - unresolvable is "no position", not a crash
        data_url = None

    _atomic_json_write(cache_path, {"event": event, "catalog": catalog, "data_url": data_url})
    return data_url


def skymap_path(event: str, catalog: str, *, root: Path | None = None) -> Path | None:
    """Download (once) and return the local posterior-samples file, or None."""
    data_url = resolve_pe_data_url(event, catalog, root=root)
    if not data_url:
        return None

    root = root or config.PATHS.cache
    destination = root / "gwosc" / f"{event}_{catalog}.hdf5".replace("/", "_")
    if destination.exists():
        return destination
    try:
        netclient.download(data_url, destination, timeout=120.0, provider="gwosc",
                           overwrite=False)
    except Exception:  # noqa: BLE001 - a failed download is "no position", not a crash
        return None
    return destination if destination.exists() else None


def _select_posterior_positions(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """RA/Dec samples in degrees from the first `_posterior` dataset found.

    Public PE releases bundle several waveform-family variants per event; this
    takes the first rather than choosing among them, a deliberate
    simplification flagged here rather than asserted as the scientifically
    preferred choice.
    """
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        group_name = next((name for name in handle if name.endswith("_posterior")), None)
        if group_name is None:
            return None
        dataset = handle[group_name]
        names = dataset.dtype.names or ()
        if "right_ascension" not in names or "declination" not in names:
            return None
        rows = dataset[:]
        ra_deg = np.degrees(rows["right_ascension"].astype(np.float64))
        dec_deg = np.degrees(rows["declination"].astype(np.float64))
    return ra_deg, dec_deg


def build_skymap_from_samples(hdf5_path: Path, nside: int = DEFAULT_NSIDE
                              ) -> tuple[np.ndarray, str] | None:
    """Per-pixel probability array from posterior samples, and its source.

    Returns (probability, position_source) where position_source is
    "gw_posterior" for a genuinely sampled position or "em_counterpart_fixed"
    when the release held sky position fixed to a known counterpart -- see
    the module docstring. None when the file has no usable position samples.
    """
    import astropy.units as u
    from astropy_healpix import HEALPix

    positions = _select_posterior_positions(hdf5_path)
    if positions is None:
        return None
    ra_deg, dec_deg = positions
    if len(ra_deg) == 0:
        return None

    source = ("em_counterpart_fixed"
             if float(np.std(ra_deg)) < _FIXED_POSITION_STD_DEG
             and float(np.std(dec_deg)) < _FIXED_POSITION_STD_DEG
             else "gw_posterior")

    healpix = HEALPix(nside=nside, order="nested")
    pixels = healpix.lonlat_to_healpix(ra_deg * u.deg, dec_deg * u.deg)
    counts = np.bincount(pixels, minlength=healpix.npix).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return None
    return counts / total, source


def credible_membership(event: str, catalog: str, ra_deg: float, dec_deg: float,
                        *, nside: int = DEFAULT_NSIDE, root: Path | None = None
                        ) -> dict | None:
    """Probability density and credible-region membership at one position.

    None when the event has no usable public position (no PE release, or a
    release with no recognisable position samples) -- distinct from a
    computed-and-low answer, matching the None-vs-zero contract `scoring.py`
    already establishes for every other evidence component in this codebase.

    Edge case, verified directly: when `position_source` is
    "em_counterpart_fixed" the posterior is a delta function (all mass in one
    pixel), and `credible_level` at that exact pixel is 1.0, not a small
    number -- correct, not a bug: a single-pixel map has no smaller-than-100%
    region containing any nonzero-probability pixel. `in_90pct_region` is
    therefore always False for a fixed position even at an exact match; a
    caller must read `position_source` to know the credible-region numbers
    are not meaningful there. Verified sane on a genuinely scattered
    synthetic posterior: a true-position pixel came back at credible_level
    0.032 (well inside 90%), a 1-sigma offset at 0.486, and a 15-sigma-away
    point at ~1.0.
    """
    import astropy.units as u
    from astropy_healpix import HEALPix

    path = skymap_path(event, catalog, root=root)
    if path is None:
        return None
    built = build_skymap_from_samples(path, nside=nside)
    if built is None:
        return None
    probability, position_source = built

    healpix = HEALPix(nside=nside, order="nested")
    pixel = int(healpix.lonlat_to_healpix(ra_deg * u.deg, dec_deg * u.deg))
    density = float(probability[pixel])

    order = np.argsort(probability)[::-1]
    cumulative = np.cumsum(probability[order])
    position_in_order = int(np.where(order == pixel)[0][0])
    credible_level = float(cumulative[position_in_order])

    return {
        "probability_density": density,
        "credible_level": credible_level,
        "in_90pct_region": credible_level <= 0.90,
        "position_source": position_source,
    }


def _event_gps_to_jd_utc(gps_time: float) -> float:
    """Event GPS time as a Julian Date in UTC.

    Full barycentric correction (up to roughly 8 minutes) is not applied: at
    the day-scale windows this coincidence check uses, that offset is
    negligible, the same reasoning already documented for HJD elsewhere in
    this codebase. This is a coincidence screen, not a timing analysis.
    """
    from astropy.time import Time

    return float(Time(gps_time, format="gps").utc.jd)


def _candidate_time_bounds_jd(path: Path) -> tuple[float, float] | None:
    """A stored candidate's observation span, as approximate JD bounds.

    MJD is converted to JD (+2400000.5); JD/HJD/BJD are used as-is -- the
    day-scale window this check runs at makes the JD/HJD/BJD/UTC distinctions
    (at most ~8 minutes) immaterial, unlike `timeframe.py`'s exact conversions,
    which exist for a different, higher-precision purpose.
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


def _temporally_coincident(time_bounds: tuple[float, float], event: GwEvent,
                           window_days: float) -> bool:
    event_jd = _event_gps_to_jd_utc(event.gps_time)
    time_min, time_max = time_bounds
    return (time_min - window_days) <= event_jd <= (time_max + window_days)


def enrich_candidate_gw(candidate: Any, events: list[GwEvent], *,
                        window_days: float = DEFAULT_WINDOW_DAYS,
                        nside: int = DEFAULT_NSIDE, root: Path | None = None
                        ) -> dict:
    """One candidate's GW coincidence evidence.

    Runs the (nearly free) time filter before the (FITS-download-and-lookup)
    spatial one, on purpose -- most candidate/event pairs fail on time alone,
    and there is no reason to fetch a skymap to learn that.
    """
    time_bounds = _candidate_time_bounds_jd(Path(candidate.path))
    if time_bounds is None:
        return {"checked_events": 0, "coincident": [], "state": "unavailable",
                "reason": "candidate light curve unreadable or empty"}

    coincident_events = [event for event in events
                         if _temporally_coincident(time_bounds, event, window_days)]

    coincident: list[dict] = []
    for event in coincident_events:
        membership = credible_membership(
            event.name, event.catalog, candidate.ra_deg, candidate.dec_deg,
            nside=nside, root=root)
        if membership is None:
            continue
        coincident.append({"event": event.name, "catalog": event.catalog,
                           "gps_time": event.gps_time, **membership})

    state = "match" if any(item["in_90pct_region"] for item in coincident) else "no_match"
    return {"checked_events": len(events), "temporally_coincident": len(coincident_events),
           "coincident": coincident, "state": state, "window_days": window_days}


def _apply_to_candidate(candidate: Any, evidence: dict) -> None:
    """Attach GW evidence without moving the composite score -- see module docstring."""
    candidate.gw = evidence
    in_region = [item for item in evidence.get("coincident", [])
                if item.get("in_90pct_region")]
    if in_region:
        names = ", ".join(item["event"] for item in in_region)
        candidate.explanation["gw_coincidence"] = (
            f"Temporally and spatially coincident with {names} "
            f"(within its 90% credible region)."
        )
        actions = [a for a in candidate.explanation.get("recommended_actions", [])
                  if "gravitational-wave" not in a.lower()]
        actions.append(
            f"Check archival/follow-up imaging around {names} for a counterpart; "
            "this coincidence is not yet part of the ranking score.")
        candidate.explanation["recommended_actions"] = actions
    elif evidence.get("state") == "unavailable":
        candidate.explanation["gw_coincidence"] = (
            "GW coincidence not checked: candidate light curve unreadable.")
    else:
        candidate.explanation["gw_coincidence"] = (
            f"No GW event coincidence in {evidence.get('checked_events', 0)} "
            "checked events.")


def enrich_candidates_gw(name: str = "default", *, catalog: str = DEFAULT_CATALOG,
                         window_days: float = DEFAULT_WINDOW_DAYS,
                         nside: int = DEFAULT_NSIDE, root: Path | None = None,
                         refresh: bool = False, offline: bool = False) -> dict:
    """Enrich an existing candidate file with GW coincidence evidence."""
    from . import candidates

    root = root or config.PATHS.projects
    cache_root = config.PATHS.cache
    built = candidates.load(name, root)
    events = fetch_event_catalog(catalog, refresh=refresh, offline=offline,
                                 root=cache_root)

    counts: dict[str, int] = {"match": 0, "no_match": 0, "unavailable": 0}
    for candidate in built:
        evidence = enrich_candidate_gw(candidate, events, window_days=window_days,
                                       nside=nside, root=cache_root)
        _apply_to_candidate(candidate, evidence)
        counts[evidence["state"]] = counts.get(evidence["state"], 0) + 1

    candidates.save(built, name, root)
    return {"catalog": catalog, "events_checked": len(events),
           "candidates": len(built), "counts": counts}
