"""Converting survey timestamps to a common frame (plan section 15).

Cross-survey timing is meaningless until every survey's clock means the same
thing. The surveys ASTRA starts with do not agree:

  ZTF   HJD_UTC   heliocentric arrival time, on the UTC scale
  TESS  BJD_TDB   barycentric arrival time, on the TDB scale

Two separate corrections stand between them:

1. Time scale. TDB - UTC = 32.184 s + (TAI - UTC), which is about 69 s today
   and changes whenever a leap second is introduced. This is the larger term.
2. Reference point. HJD is referred to the centre of the Sun, BJD to the
   solar-system barycentre. The Sun orbits that barycentre at up to roughly
   two solar radii, so the light-travel difference reaches only a few seconds.

Note this is *not* the +-8.3 minute correction people usually quote — that is
the geocentric-to-barycentric term, and HJD has already absorbed almost all of
it. `measure_frame_offset` computes the real number for a given position and
date rather than relying on any rule of thumb.

BJD_TDB is the target frame because it is the standard for precise timing and
is what TESS already publishes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surveys.base import LightCurve, TimeSystem

TARGET_SYSTEM: TimeSystem = "BJD_TDB"

# Observatory positions. The site registry needs network access, so the
# observatories ASTRA queries are recorded here directly.
SITES: dict[str, dict[str, float]] = {
    # Palomar Observatory, home of ZTF on the Samuel Oschin 48-inch.
    "ZTF": {"lon_deg": -116.8650, "lat_deg": 33.3563, "height_m": 1712.0},
    # Gaia and TESS are space-based; their products are already barycentric.
    "GAIA": {"lon_deg": 0.0, "lat_deg": 0.0, "height_m": 0.0},
    "TESS": {"lon_deg": 0.0, "lat_deg": 0.0, "height_m": 0.0},
}

MJD_TO_JD = 2400000.5


@dataclass(frozen=True)
class FrameOffset:
    """Measured difference between a source frame and BJD_TDB, in seconds."""

    scale_seconds: float       # UTC -> TDB
    reference_seconds: float   # heliocentric -> barycentric light travel
    total_seconds: float

    def to_dict(self) -> dict:
        return {
            "scale_seconds": round(self.scale_seconds, 4),
            "reference_seconds": round(self.reference_seconds, 4),
            "total_seconds": round(self.total_seconds, 4),
        }


def _earth_location(survey: str):
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    site = SITES.get(survey.upper(), SITES["ZTF"])
    return EarthLocation.from_geodetic(
        lon=site["lon_deg"] * u.deg,
        lat=site["lat_deg"] * u.deg,
        height=site["height_m"] * u.m,
    )


def to_bjd_tdb(time: np.ndarray, time_system: TimeSystem,
               ra_deg: float, dec_deg: float,
               survey: str = "ZTF") -> np.ndarray:
    """Convert a time array to BJD_TDB.

    Already-barycentric input is returned untouched. Conversion is exact for
    the frames ASTRA handles rather than an approximation, because the whole
    point is to stop small systematic offsets from masquerading as physics.
    """
    time = np.asarray(time, dtype=np.float64)
    if time.size == 0 or time_system == TARGET_SYSTEM:
        return time

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    location = _earth_location(survey)

    if time_system == "MJD_UTC":
        jd = time + MJD_TO_JD
        stamps = Time(jd, format="jd", scale="utc", location=location)
    elif time_system in ("JD_UTC", "HJD_UTC"):
        stamps = Time(time, format="jd", scale="utc", location=location)
    else:
        raise ValueError(f"unsupported time system: {time_system!r}")

    if time_system == "HJD_UTC":
        # Undo the heliocentric correction to recover geocentric arrival,
        # then apply the barycentric one. Going straight from helio to bary
        # would double-count the Earth's orbital term.
        stamps = stamps - stamps.light_travel_time(coord, "heliocentric")

    barycentric = stamps.tdb + stamps.light_travel_time(coord, "barycentric")
    return np.asarray(barycentric.jd, dtype=np.float64)


def measure_frame_offset(time_system: TimeSystem, ra_deg: float, dec_deg: float,
                         reference_jd: float, survey: str = "ZTF") -> FrameOffset:
    """Measure the real correction at one position and date.

    Exists so the size of the effect is a measurement in the record rather
    than a remembered rule of thumb.
    """
    if time_system == TARGET_SYSTEM:
        return FrameOffset(0.0, 0.0, 0.0)

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    location = _earth_location(survey)
    stamps = Time(reference_jd, format="jd", scale="utc", location=location)

    scale_seconds = float((stamps.tdb.jd - stamps.utc.jd) * 86400.0)

    helio = stamps.light_travel_time(coord, "heliocentric")
    bary = stamps.light_travel_time(coord, "barycentric")
    reference_seconds = float((bary - helio).to(u.s).value)

    converted = to_bjd_tdb(np.array([reference_jd]), time_system,
                           ra_deg, dec_deg, survey)
    total_seconds = float((converted[0] - reference_jd) * 86400.0)

    return FrameOffset(scale_seconds, reference_seconds, total_seconds)


def align(curve: LightCurve) -> LightCurve:
    """Return the curve with its times converted to BJD_TDB.

    Curves already in the target frame are returned unchanged, so this is safe
    to apply to everything before any cross-survey comparison.
    """
    if curve.time_system == TARGET_SYSTEM or len(curve) == 0:
        return curve

    converted = to_bjd_tdb(curve.time, curve.time_system,
                           curve.source.ra_deg, curve.source.dec_deg,
                           curve.source.survey)

    return LightCurve(
        source=curve.source,
        release=curve.release,
        band=curve.band,
        value_kind=curve.value_kind,
        time=converted,
        value=curve.value,
        value_err=curve.value_err,
        time_system=TARGET_SYSTEM,
    )


def overlap_days(first: LightCurve, second: LightCurve) -> float:
    """Length of the interval both curves cover, after frame alignment.

    Returns 0 when they never observed the source at the same time, which is
    itself evidence: two surveys covering disjoint epochs cannot corroborate
    a single transient event, only a persistent behaviour.
    """
    if len(first) == 0 or len(second) == 0:
        return 0.0

    a = align(first)
    b = align(second)

    start = max(float(a.time[0]), float(b.time[0]))
    end = min(float(a.time[-1]), float(b.time[-1]))
    return max(0.0, end - start)
