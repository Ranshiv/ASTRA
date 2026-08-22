"""Local, draft-only observability planning for candidate follow-up."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

SCHEMA_VERSION = 2
MAX_DURATION_HOURS = 24 * 7
MAX_SLOTS = 10_000


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _altitude(ra_deg: float, dec_deg: float, when: datetime,
              latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
    # Greenwich mean sidereal time approximation, sufficient for a planning
    # draft.  The report labels this as approximate and should not be used as
    # an observatory control command without a facility-specific ephemeris.
    jd = (when.timestamp() / 86400.0) + 2440587.5
    d = jd - 2451545.0
    lst = (280.46061837 + 360.98564736629 * d + longitude_deg) % 360.0
    hour_angle = math.radians((lst - ra_deg + 540.0) % 360.0 - 180.0)
    lat = math.radians(latitude_deg)
    dec = math.radians(dec_deg)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hour_angle)
    altitude = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))
    cos_az = ((math.sin(dec) - math.sin(math.radians(altitude)) * math.sin(lat))
              / max(math.cos(math.radians(altitude)) * math.cos(lat), 1e-9))
    azimuth = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if math.sin(hour_angle) > 0:
        azimuth = 360.0 - azimuth
    return altitude, azimuth


def _solar_position(when: datetime) -> tuple[float, float]:
    """Low-precision apparent solar RA/Dec, adequate for a draft filter."""
    jd = (when.timestamp() / 86400.0) + 2440587.5
    days = jd - 2451545.0
    mean_longitude = math.radians((280.460 + 0.9856474 * days) % 360.0)
    anomaly = math.radians((357.528 + 0.9856003 * days) % 360.0)
    ecliptic = mean_longitude + math.radians(1.915) * math.sin(anomaly) \
        + math.radians(0.020) * math.sin(2.0 * anomaly)
    obliquity = math.radians(23.439 - 0.0000004 * days)
    ra = math.degrees(math.atan2(math.cos(obliquity) * math.sin(ecliptic),
                                 math.cos(ecliptic))) % 360.0
    dec = math.degrees(math.asin(math.sin(obliquity) * math.sin(ecliptic)))
    return ra, dec


def _moon_position(when: datetime) -> tuple[float, float]:
    """Compact Paul Schlyter-style lunar ephemeris for planning filters."""
    jd = (when.timestamp() / 86400.0) + 2440587.5
    days = jd - 2451543.5
    node = math.radians((125.1228 - 0.0529538083 * days) % 360.0)
    inclination = math.radians(5.1454)
    periapsis = math.radians((318.0634 + 0.1643573223 * days) % 360.0)
    eccentricity = 0.0549
    mean_anomaly = math.radians((115.3654 + 13.0649929509 * days) % 360.0)
    eccentric_anomaly = mean_anomaly + eccentricity * math.sin(mean_anomaly) \
        * (1.0 + eccentricity * math.cos(mean_anomaly))
    x_orbit = 60.2666 * (math.cos(eccentric_anomaly) - eccentricity)
    y_orbit = 60.2666 * math.sqrt(1.0 - eccentricity * eccentricity) \
        * math.sin(eccentric_anomaly)
    true_anomaly = math.atan2(y_orbit, x_orbit)
    radius = math.hypot(x_orbit, y_orbit)
    argument = true_anomaly + periapsis
    x_ecliptic = radius * (math.cos(node) * math.cos(argument)
                           - math.sin(node) * math.sin(argument) * math.cos(inclination))
    y_ecliptic = radius * (math.sin(node) * math.cos(argument)
                           + math.cos(node) * math.sin(argument) * math.cos(inclination))
    z_ecliptic = radius * math.sin(argument) * math.sin(inclination)
    longitude = math.atan2(y_ecliptic, x_ecliptic)
    latitude = math.atan2(z_ecliptic, math.hypot(x_ecliptic, y_ecliptic))
    obliquity = math.radians(23.4393)
    ra = math.degrees(math.atan2(
        math.sin(longitude) * math.cos(obliquity)
        - math.tan(latitude) * math.sin(obliquity), math.cos(longitude))) % 360.0
    dec = math.degrees(math.asin(
        math.sin(latitude) * math.cos(obliquity)
        + math.cos(latitude) * math.sin(obliquity) * math.sin(longitude)))
    return ra, dec


def _moon_illumination(when: datetime) -> float:
    sun_ra, sun_dec = _solar_position(when)
    moon_ra, moon_dec = _moon_position(when)
    elongation = math.radians(angular_separation_deg(sun_ra, sun_dec, moon_ra, moon_dec))
    return (1.0 - math.cos(elongation)) / 2.0


def angular_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    first_ra, first_dec, second_ra, second_dec = map(
        math.radians, (ra1, dec1, ra2, dec2))
    cosine = math.sin(first_dec) * math.sin(second_dec) + math.cos(first_dec) \
        * math.cos(second_dec) * math.cos(first_ra - second_ra)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _weather_value(weather: list[dict[str, Any]] | dict[str, Any] | None,
                   when: datetime) -> bool | None:
    if weather is None:
        return None
    if isinstance(weather, dict):
        entries = [{"utc": key, "usable": value} for key, value in weather.items()]
    elif isinstance(weather, list):
        entries = [item for item in weather if isinstance(item, dict)]
    else:
        return None
    parsed = []
    for entry in entries:
        text = entry.get("utc") or entry.get("time")
        try:
            value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
            value = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        parsed.append((abs((value - when).total_seconds()), bool(entry.get("usable", entry.get("ok", False)))))
    if not parsed:
        return None
    distance, value = min(parsed, key=lambda item: item[0])
    return value if distance <= 90 * 60 else None


def plan(*, ra_deg: float, dec_deg: float, start_utc: str | None = None,
         duration_hours: float = 12.0, latitude_deg: float = 43.65,
         longitude_deg: float = -79.38, min_altitude_deg: float = 30.0,
         cadence_minutes: int = 10, target_id: str | None = None,
         twilight_sun_altitude_deg: float = -18.0,
         min_moon_separation_deg: float = 0.0,
         max_moon_illumination: float = 1.0,
         max_airmass: float | None = None,
         weather: list[dict[str, Any]] | dict[str, Any] | None = None,
         facility_name: str | None = None,
         facility_constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return visible windows and best slot; never submits an observation."""
    ra, dec = float(ra_deg), float(dec_deg)
    if not 0.0 <= ra < 360.0 or not -90.0 <= dec <= 90.0:
        raise ValueError("target coordinates are out of range")
    duration = float(duration_hours)
    if not 0.0 < duration <= MAX_DURATION_HOURS:
        raise ValueError(f"duration_hours must be in (0, {MAX_DURATION_HOURS}]" )
    cadence = int(cadence_minutes)
    if not 1 <= cadence <= 60:
        raise ValueError("cadence_minutes must be between 1 and 60")
    if not -90.0 <= float(latitude_deg) <= 90.0:
        raise ValueError("latitude_deg is out of range")
    if not -180.0 <= float(longitude_deg) <= 180.0:
        raise ValueError("longitude_deg is out of range")
    if not 0.0 <= float(min_altitude_deg) < 90.0:
        raise ValueError("min_altitude_deg must be in [0, 90)")
    if not -90.0 <= float(twilight_sun_altitude_deg) <= 10.0:
        raise ValueError("twilight_sun_altitude_deg is out of range")
    if not 0.0 <= float(min_moon_separation_deg) <= 180.0:
        raise ValueError("min_moon_separation_deg must be in [0, 180]")
    if not 0.0 <= float(max_moon_illumination) <= 1.0:
        raise ValueError("max_moon_illumination must be in [0, 1]")
    if max_airmass is not None and (not math.isfinite(float(max_airmass)) or float(max_airmass) < 1.0):
        raise ValueError("max_airmass must be at least 1 when supplied")
    facility = dict(facility_constraints or {})
    if facility_name:
        facility.setdefault("name", facility_name)
    if "min_altitude_deg" in facility:
        min_altitude_deg = max(float(min_altitude_deg), float(facility["min_altitude_deg"]))
    if "max_airmass" in facility:
        facility_airmass = float(facility["max_airmass"])
        max_airmass = facility_airmass if max_airmass is None else min(float(max_airmass), facility_airmass)
    if "min_moon_separation_deg" in facility:
        min_moon_separation_deg = max(float(min_moon_separation_deg), float(facility["min_moon_separation_deg"]))
    if "twilight_sun_altitude_deg" in facility:
        twilight_sun_altitude_deg = float(facility["twilight_sun_altitude_deg"])
    if not -90.0 <= float(twilight_sun_altitude_deg) <= 10.0:
        raise ValueError("twilight_sun_altitude_deg is out of range")
    if not 0.0 <= float(min_moon_separation_deg) <= 180.0:
        raise ValueError("min_moon_separation_deg must be in [0, 180]")
    if max_airmass is not None and (not math.isfinite(float(max_airmass)) or float(max_airmass) < 1.0):
        raise ValueError("max_airmass must be at least 1 when supplied")
    start = _parse_time(start_utc)
    slots = min(MAX_SLOTS, int(duration * 60 / cadence) + 1)
    samples = []
    rejected = {"below_altitude": 0, "airmass": 0, "twilight": 0,
                "moon_separation": 0, "moon_illumination": 0, "weather": 0}
    for index in range(slots):
        when = start + timedelta(minutes=index * cadence)
        altitude, azimuth = _altitude(ra, dec, when, float(latitude_deg), float(longitude_deg))
        airmass = 1.0 / max(math.sin(math.radians(altitude)), 1e-6)
        sun_ra, sun_dec = _solar_position(when)
        moon_ra, moon_dec = _moon_position(when)
        sun_altitude, _ = _altitude(sun_ra, sun_dec, when, float(latitude_deg), float(longitude_deg))
        moon_separation = angular_separation_deg(ra, dec, moon_ra, moon_dec)
        moon_illumination = _moon_illumination(when)
        weather_ok = _weather_value(weather, when)
        if altitude < float(min_altitude_deg):
            rejected["below_altitude"] += 1
            continue
        if max_airmass is not None and airmass > float(max_airmass):
            rejected["airmass"] += 1
            continue
        if sun_altitude > float(twilight_sun_altitude_deg):
            rejected["twilight"] += 1
            continue
        if moon_separation < float(min_moon_separation_deg):
            rejected["moon_separation"] += 1
            continue
        if moon_illumination > float(max_moon_illumination):
            rejected["moon_illumination"] += 1
            continue
        if weather_ok is False:
            rejected["weather"] += 1
            continue
        samples.append({"utc": when.isoformat(), "altitude_deg": round(altitude, 3),
                        "azimuth_deg": round(azimuth, 3), "airmass": round(airmass, 3),
                        "sun_altitude_deg": round(sun_altitude, 3),
                        "moon_separation_deg": round(moon_separation, 3),
                        "moon_illumination": round(moon_illumination, 4),
                        "weather_ok": weather_ok})
    windows: list[dict[str, Any]] = []
    if samples:
        begin = samples[0]
        previous = samples[0]
        for current in samples[1:]:
            if datetime.fromisoformat(current["utc"]) - datetime.fromisoformat(previous["utc"]) > timedelta(minutes=cadence * 1.5):
                windows.append({"start_utc": begin["utc"], "end_utc": previous["utc"],
                                "slots": 0})
                begin = current
            previous = current
        windows.append({"start_utc": begin["utc"], "end_utc": previous["utc"], "slots": 0})
        for window in windows:
            window["slots"] = sum(window["start_utc"] <= item["utc"] <= window["end_utc"]
                                  for item in samples)
    best = min(samples, key=lambda item: item["airmass"]) if samples else None
    return {
        "schema_version": SCHEMA_VERSION,
        "target_id": target_id,
        "target": {"ra_deg": ra, "dec_deg": dec},
        "site": {"latitude_deg": float(latitude_deg), "longitude_deg": float(longitude_deg)},
        "constraints": {"min_altitude_deg": float(min_altitude_deg),
                        "cadence_minutes": cadence,
                        "twilight_sun_altitude_deg": float(twilight_sun_altitude_deg),
                        "min_moon_separation_deg": float(min_moon_separation_deg),
                        "max_moon_illumination": float(max_moon_illumination),
                        "max_airmass": float(max_airmass) if max_airmass is not None else None,
                        "facility": facility or None,
                        "weather_supplied": weather is not None},
        "start_utc": start.isoformat(), "duration_hours": duration,
        "visible": bool(samples), "windows": windows, "best_slot": best,
        "samples": samples[:MAX_SLOTS],
        "rejected_slots": rejected,
        "mode": "draft_only",
        "caveats": ["Approximate sidereal-time, solar, and lunar geometry; verify with the facility ephemeris.",
                     "Weather is only applied when caller-supplied forecast samples are provided; no forecast is fetched automatically.",
                     "Facility instrument limits, conflicts, and scheduling ownership still require observatory validation.",
                     "No observation request was submitted."],
    }
