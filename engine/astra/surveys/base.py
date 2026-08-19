"""Common vocabulary for every survey connector (plan sections 6, 7 and 12).

Surveys disagree about almost everything: ZTF reports magnitudes, TESS reports
electron flux, Gaia reports a static astrometric solution rather than a time
series. Normalising into one schema here is what makes the cross-survey
matching in section 15 possible without special-casing each archive.

Adding a survey means adding a connector, never editing this module.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

ValueKind = Literal["mag", "flux"]

# Surveys timestamp differently: ZTF publishes heliocentric JD, TESS publishes
# barycentric TESS JD. These disagree by up to ~8 minutes, which is larger than
# a TESS 2-minute cadence, so the frame travels with the data rather than being
# assumed. Full barycentric correction to a single frame is deferred to the
# cross-survey engine (plan section 15); until then this field is what stops
# two surveys being silently misaligned.
TimeSystem = Literal["JD_UTC", "HJD_UTC", "BJD_TDB", "MJD_UTC"]

# Time is always float64. Barycentric Julian dates look like 2457000.123456,
# and float32 carries ~7 significant digits, which would destroy the timing
# resolution every period and flare-duration feature depends on.
TIME_DTYPE = np.float64
# Photometry genuinely does not need more than float32, and halving it here
# halves the canonical store.
VALUE_DTYPE = np.float32


@dataclass(frozen=True)
class ConeQuery:
    """A positional search, in the form every archive accepts."""

    ra_deg: float
    dec_deg: float
    radius_arcsec: float

    def __post_init__(self) -> None:
        if not -360.0 <= self.ra_deg <= 360.0:
            raise ValueError(f"ra_deg out of range: {self.ra_deg}")
        if not -90.0 <= self.dec_deg <= 90.0:
            raise ValueError(f"dec_deg out of range: {self.dec_deg}")
        if self.radius_arcsec <= 0:
            raise ValueError(f"radius_arcsec must be positive: {self.radius_arcsec}")

    @property
    def radius_deg(self) -> float:
        return self.radius_arcsec / 3600.0

    def key(self) -> str:
        """Stable identity, so the same cone always hashes to the same manifest."""
        return f"{self.ra_deg:.6f}_{self.dec_deg:.6f}_{self.radius_arcsec:.3f}"


@dataclass(frozen=True)
class SourceRef:
    """One object as a survey names it, before any data is downloaded."""

    survey: str
    object_id: str
    ra_deg: float
    dec_deg: float
    extra: dict = field(default_factory=dict)

    def storage_key(self, release: str) -> str:
        """Content address for the canonical store.

        Keyed by (survey, release, object_id) so seven experiment groups over
        overlapping object sets share one copy instead of seven.
        """
        raw = f"{self.survey}/{release}/{self.object_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class LightCurve:
    """A normalised time series, whatever survey it came from."""

    source: SourceRef
    release: str
    band: str
    value_kind: ValueKind
    time: np.ndarray
    value: np.ndarray
    value_err: np.ndarray
    time_system: TimeSystem = "JD_UTC"

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=TIME_DTYPE)
        self.value = np.asarray(self.value, dtype=VALUE_DTYPE)
        self.value_err = np.asarray(self.value_err, dtype=VALUE_DTYPE)

        lengths = {len(self.time), len(self.value), len(self.value_err)}
        if len(lengths) != 1:
            raise ValueError(
                "time, value and value_err must be the same length; "
                f"got {len(self.time)}, {len(self.value)}, {len(self.value_err)}"
            )

    def __len__(self) -> int:
        return len(self.time)

    def finite_mask(self) -> np.ndarray:
        """Points usable for analysis — archives pad gaps with NaN."""
        return (
            np.isfinite(self.time)
            & np.isfinite(self.value)
            & np.isfinite(self.value_err)
        )

    def _with_points(self, index: np.ndarray) -> "LightCurve":
        return LightCurve(
            source=self.source,
            release=self.release,
            band=self.band,
            value_kind=self.value_kind,
            time=self.time[index],
            value=self.value[index],
            value_err=self.value_err[index],
            time_system=self.time_system,
        )

    def dropna(self) -> "LightCurve":
        return self._with_points(self.finite_mask())

    def sorted_by_time(self) -> "LightCurve":
        return self._with_points(np.argsort(self.time, kind="stable"))

    def time_span_days(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(np.nanmax(self.time) - np.nanmin(self.time))


class SurveyConnector(ABC):
    """One archive. Network access is confined to the two fetch methods.

    Implementations must not cache to disk themselves — `astra.store` owns
    persistence, so the cache cap in `astra.cache` stays authoritative.
    """

    name: str
    release: str
    # Capability metadata is part of the connector contract.  It lets the UI
    # and acquisition planner reject an impossible request before network I/O
    # (for example, asking a static catalogue for a light curve).
    capabilities: tuple[str, ...] = ("catalogue", "light_curve")
    credential_required: bool = False
    resolution_arcsec: float | None = None
    enabled_by_default: bool = True

    @abstractmethod
    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        """Return objects near a position without downloading their data."""

    @abstractmethod
    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        """Download and normalise one object's time series, one per band."""

    def describe(self) -> dict:
        return {"name": self.name, "release": self.release,
                "class": type(self).__name__,
                "capabilities": list(self.capabilities),
                "credential_required": self.credential_required,
                "resolution_arcsec": self.resolution_arcsec,
                "enabled_by_default": self.enabled_by_default}


def normalise_band(survey: str, raw: object) -> str:
    """Map survey-specific filter codes onto readable band names."""
    text = str(raw).strip().lower()
    table = {
        "ztf": {"1": "g", "2": "r", "3": "i", "zg": "g", "zr": "r", "zi": "i"},
        "tess": {"": "TESS", "none": "TESS", "tess": "TESS"},
        "gaia": {"g": "G", "bp": "BP", "rp": "RP"},
    }
    return table.get(survey.lower(), {}).get(text, text or "unknown")


def to_arrays(rows: Iterable[tuple[float, float, float]]) -> tuple[np.ndarray, ...]:
    """Build correctly typed columns from row tuples, preserving time precision."""
    materialised = list(rows)
    if not materialised:
        return (
            np.empty(0, dtype=TIME_DTYPE),
            np.empty(0, dtype=VALUE_DTYPE),
            np.empty(0, dtype=VALUE_DTYPE),
        )
    times, values, errors = zip(*materialised)
    return (
        np.asarray(times, dtype=TIME_DTYPE),
        np.asarray(values, dtype=VALUE_DTYPE),
        np.asarray(errors, dtype=VALUE_DTYPE),
    )
