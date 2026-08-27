"""OGLE Early Warning System — real microlensing events with real published
model parameters (backlog item 15).

Unusually valuable among this codebase's connectors, and the reason item 15
can measure "parameter bias" against a REAL published baseline rather than
only against synthetic injections: OGLE's EWS season index publishes its
own fitted Paczynski parameters for every event (Tmax, tau, umin, Amax,
fbl, Ibl, I0) alongside the position, and each event has real, downloadable
photometry. So a fit produced by `microlensing_fit.py` can be compared to
what the survey that discovered the event actually published for it.

Both contracts were verified live while writing this module, the same
discipline `alerce.py`/`sdss.py`/`tess_pixels.py` already document:
  * the season index at
    `ogle.astrouw.edu.pl/ogle4/ews/{year}/ews.html` -- a real HTML table,
    one row per event, confirmed for the 2019 season (1,526 events);
  * per-event photometry at
    `www.astrouw.edu.pl/ogle/ogle4/ews/{year}/blg-NNNN/phot.dat` -- real
    plain text, five whitespace-separated columns (HJD, I magnitude,
    magnitude error, seeing, sky level), ~3,800 rows for the event checked.
Note the two live on DIFFERENT host paths; that is OGLE's own layout, not
a typo. Credential-free, no API key, no rate-limit documentation -- so this
module uses `netclient`'s throttle like every other connector rather than
hammering a university web server.

`cone_search` filters a fetched season index by position rather than
querying server-side: OGLE publishes no cone-search endpoint, and saying
so plainly is better than a fake one. `fetch_light_curves` needs an event
already identified (its `extra` carries the year and event number).
"""

from __future__ import annotations

import re

import numpy as np

from .. import netclient
from .base import ConeQuery, LightCurve, SourceRef, SurveyConnector, to_arrays

EWS_INDEX_URL = "https://ogle.astrouw.edu.pl/ogle4/ews/{year}/ews.html"
PHOTOMETRY_URL = "https://www.astrouw.edu.pl/ogle/ogle4/ews/{year}/blg-{number:04d}/phot.dat"

DEFAULT_RELEASE = "ogle4-ews"
# OGLE-IV EWS seasons. 2010 is the first OGLE-IV season; the connector does
# not guess beyond a caller-supplied year, it only rejects obviously
# impossible ones.
MIN_YEAR = 2010
MAX_YEAR = 2100

_EVENT_RE = re.compile(r"(\d{4})-BLG-(\d{4})")


def _finite(text: object) -> float | None:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _sexagesimal_to_degrees(ra_text: str, dec_text: str) -> tuple[float, float] | None:
    """OGLE publishes positions as HH:MM:SS.s / +DD:MM:SS.s."""
    try:
        ra_parts = [float(part) for part in str(ra_text).strip().split(":")]
        dec_parts = [float(part) for part in str(dec_text).strip().split(":")]
        if len(ra_parts) != 3 or len(dec_parts) != 3:
            return None
        ra_deg = 15.0 * (ra_parts[0] + ra_parts[1] / 60.0 + ra_parts[2] / 3600.0)
        # The sign belongs to the whole declination, not just its degrees.
        sign = -1.0 if str(dec_text).strip().startswith("-") else 1.0
        dec_deg = sign * (abs(dec_parts[0]) + dec_parts[1] / 60.0 + dec_parts[2] / 3600.0)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= ra_deg < 360.0 and -90.0 <= dec_deg <= 90.0):
        return None
    return ra_deg, dec_deg


def _row_cells(raw_row: str) -> list[str]:
    """Cell text from one EWS table row.

    Split on the OPENING `<td>`/`<th>` tag rather than matching paired
    tags: the real EWS page is HTML 3.2-style and does not close its cells
    (verified against the live 2019 index -- rows look like
    `<TD>BLG500.01  <TD ALIGN="RIGHT"> 179275`). A paired-tag regex finds
    nothing at all on the real page, which is exactly the silent
    zero-rows failure this codebase's connector notes keep warning about.
    """
    pieces = re.split(r"<t[dh][^>]*>", raw_row, flags=re.IGNORECASE)
    # pieces[0] is whatever preceded the first cell, not a cell itself.
    return [re.sub(r"<[^>]+>", " ", piece).strip() for piece in pieces[1:]]


def parse_event_table(html: str, limit: int = 5000) -> list[dict]:
    """Rows of the real EWS season index.

    Columns are located RELATIVE to the cell holding the event name rather
    than by absolute index: the real table carries a leading empty cell
    before the event link, and anchoring on the event name makes the
    parser robust to that (and to it changing) instead of silently
    reading every parameter one column off.
    """
    rows: list[dict] = []
    for raw_row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL):
        cells = _row_cells(raw_row)

        anchor = None
        for index, cell in enumerate(cells):
            match = _EVENT_RE.search(cell)
            if match:
                anchor, event_match = index, match
                break
        if anchor is None or len(cells) < anchor + 14:
            continue

        def cell(offset: int) -> str:
            return cells[anchor + offset]

        position = _sexagesimal_to_degrees(cell(3), cell(4))
        if position is None:
            continue
        ra_deg, dec_deg = position

        rows.append({
            "event": f"{event_match.group(1)}-BLG-{event_match.group(2)}",
            "year": int(event_match.group(1)),
            "number": int(event_match.group(2)),
            "field": cell(1),
            "star": cell(2),
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            # OGLE's OWN published fit for this event -- the reference
            # `microlensing_eval.parameter_bias` compares against.
            # Offsets: 5=Tmax(HJD) 6=Tmax(UT) 7=tau 8=Umin 9=Amax
            #          10=Dmag 11=f_bl 12=I_bl 13=I_0
            "t0_hjd": _finite(cell(5)),
            "tE_days": _finite(cell(7)),
            "u0": _finite(cell(8)),
            "amax": _finite(cell(9)),
            "dmag": _finite(cell(10)),
            "f_bl": _finite(cell(11)),
            "I_bl": _finite(cell(12)),
            "I0": _finite(cell(13)),
        })
        if len(rows) >= limit:
            break
    return rows


def parse_photometry(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The real 5-column `phot.dat`: HJD, I mag, mag error, seeing, sky.

    Only the first three columns are kept; seeing and sky level are real
    observing metadata but this codebase's `LightCurve` has no per-point
    slot for them (`store.SCHEMA` is time/value/value_err), so they are
    dropped rather than silently misfiled into a column that means
    something else.
    """
    points: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            hjd, mag, mag_err = (float(parts[0]), float(parts[1]), float(parts[2]))
        except (TypeError, ValueError):
            continue
        if not np.isfinite([hjd, mag, mag_err]).all() or mag_err <= 0:
            continue
        points.append((hjd, mag, mag_err))
    return to_arrays(points)


class OGLEConnector(SurveyConnector):
    name = "OGLE"
    capabilities = ("catalogue", "light_curve", "published_model")
    resolution_arcsec = 0.4
    credential_required = False
    # Opt-in, like every connector added since sdss.py.
    enabled_by_default = False

    def __init__(self, release: str = DEFAULT_RELEASE) -> None:
        self.release = release

    def list_events(self, year: int, limit: int = 5000) -> list[SourceRef]:
        """Every real EWS event for a season, carrying OGLE's published fit."""
        year = int(year)
        if not MIN_YEAR <= year <= MAX_YEAR:
            raise ValueError(f"year must be between {MIN_YEAR} and {MAX_YEAR}")

        response = netclient.get(EWS_INDEX_URL.format(year=year), {},
                                 timeout=120, provider="ogle")
        sources: list[SourceRef] = []
        for row in parse_event_table(response.text, limit=limit):
            sources.append(SourceRef(
                survey=self.name, object_id=row["event"],
                ra_deg=row["ra_deg"], dec_deg=row["dec_deg"],
                extra={key: row[key] for key in (
                    "year", "number", "field", "star", "t0_hjd", "tE_days",
                    "u0", "amax", "dmag", "f_bl", "I_bl", "I0")},
            ))
        return sources

    def cone_search(self, query: ConeQuery, limit: int = 100) -> list[SourceRef]:
        """Positional filter over a fetched season index.

        OGLE publishes no cone-search endpoint, so this fetches the season
        index and filters locally. `query.extra_year` (or the current year)
        selects the season; a caller wanting several seasons calls this
        once per season rather than this module guessing a range.
        """
        year = int(getattr(query, "year", 0) or 0)
        if not year:
            from datetime import datetime, timezone
            year = datetime.now(timezone.utc).year

        events = self.list_events(year)
        radius_deg = query.radius_arcsec / 3600.0
        cos_dec = np.cos(np.radians(query.dec_deg))

        matched: list[SourceRef] = []
        for source in events:
            delta_ra = (source.ra_deg - query.ra_deg) * cos_dec
            delta_dec = source.dec_deg - query.dec_deg
            if np.hypot(delta_ra, delta_dec) <= radius_deg:
                matched.append(source)
            if len(matched) >= limit:
                break
        return matched

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        """The real per-event photometry, as a single I-band curve."""
        extra = source.extra or {}
        try:
            year = int(extra["year"])
            number = int(extra["number"])
        except (KeyError, TypeError, ValueError):
            match = _EVENT_RE.search(str(source.object_id))
            if not match:
                raise ValueError(
                    "OGLE light curves need an event identified by list_events() "
                    "or an object_id of the form YYYY-BLG-NNNN"
                ) from None
            year, number = int(match.group(1)), int(match.group(2))

        response = netclient.get(
            PHOTOMETRY_URL.format(year=year, number=number), {},
            timeout=120, provider="ogle")
        time, value, value_err = parse_photometry(response.text)
        if len(time) == 0:
            return []

        return [LightCurve(
            source=source, release=self.release, band="I", value_kind="mag",
            time=time, value=value, value_err=value_err, time_system="HJD_UTC",
        )]


def published_parameters(source: SourceRef) -> dict | None:
    """OGLE's own fitted parameters for an event, or None when absent.

    This is the real reference baseline `microlensing_eval.parameter_bias`
    compares a fit against. Returns None rather than a partial dict when
    the core three are not all present, so a caller can never silently
    compare against a half-missing reference.
    """
    extra = source.extra or {}
    t0, tE, u0 = extra.get("t0_hjd"), extra.get("tE_days"), extra.get("u0")
    if t0 is None or tE is None or u0 is None:
        return None
    return {"t0": float(t0), "tE": float(tE), "u0": float(u0),
            "f_bl": extra.get("f_bl"), "I_bl": extra.get("I_bl")}
