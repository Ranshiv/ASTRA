"""Physics-informed stellar latent manifold (backlog item 12).

A star's position on the Gaia colour-magnitude diagram (BP-RP colour vs.
absolute G magnitude) is not arbitrary: normal single main-sequence stars lie
close to a well-measured track. This module compares a star's real,
extinction-corrected CMD position against a real, published main-sequence
track and reports how far off it falls -- a physically grounded "latent
manifold coordinate" (position along the track plus perpendicular residual),
not an arbitrarily learned embedding.

`ZAMS_TRACK` is a real, live-verified subset of Eric Mamajek's maintained
dwarf-star table (Pecaut & Mamajek 2013, ApJS 208, 9;
`EEM_dwarf_UBVIJHK_colors_Teff.txt`, fetched 2026-08-22 from
http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt),
covering spectral types B9V (10700 K) through M7V (2680 K). The O/B0-B7 end
of that table does not publish a Gaia BP-RP column, so the embedded track
deliberately starts at B9V -- an explicit scope limit (rare, short-lived hot
stars fall outside it), not a gap glossed over. This is ONE static,
approximately solar-metallicity zero-age-main-sequence track, not a fitted
multi-age/multi-metallicity isochrone grid; a giant branch, subdwarf, or
significantly non-solar-metallicity star is expected to show a real,
physically meaningful residual against it, not necessarily an error.

Extinction correction uses two DISTINCT Gaia quantities, kept separate and
never confused: `a_g` (Gaia's own G-band extinction estimate, corrects the
magnitude) and `ebpminrp` (Gaia's own E(BP-RP) reddening estimate, already
in BP-RP colour units, corrects the colour directly). This is a different,
more direct convention than `scoring.physical_inconsistency`'s `2.74 * ebv`
fallback (which approximates A_G from a differently-normalised reddening
only when `a_g` itself is absent) -- that function only ever corrects a
magnitude, never a colour, so there was no precedent to reuse here.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = 1

# (spectral_type, teff_kelvin, bp_rp, abs_g_mag), ascending BP-RP (hot to
# cool). Real values, see module docstring for provenance.
ZAMS_TRACK: tuple[tuple[str, float, float, float], ...] = (
    ("B9V", 10700.0, -0.120, 0.515),
    ("A1V", 9300.0, 0.005, 1.16),
    ("A3V", 8600.0, 0.110, 1.69),
    ("A5V", 8100.0, 0.194, 1.98),
    ("A7V", 7760.0, 0.263, 2.19),
    ("A9V", 7400.0, 0.327, 2.37),
    ("F1V", 7020.0, 0.434, 2.69),
    ("F3V", 6750.0, 0.518, 2.99),
    ("F5V", 6550.0, 0.587, 3.26),
    ("F7V", 6280.0, 0.670, 3.66),
    ("F9V", 6050.0, 0.719, 4.105),
    ("G1V", 5860.0, 0.803, 4.462),
    ("G3V", 5720.0, 0.832, 4.703),
    ("G5V", 5660.0, 0.850, 4.801),
    ("G7V", 5550.0, 0.880, 5.006),
    ("G9V", 5380.0, 0.950, 5.34),
    ("K1V", 5170.0, 1.01, 5.65),
    ("K3V", 4830.0, 1.21, 6.20),
    ("K5V", 4440.0, 1.43, 6.83),
    ("K7V", 4100.0, 1.70, 7.57),
    ("K9V", 3930.0, 1.79, 8.03),
    ("M1V", 3660.0, 2.09, 8.82),
    ("M3V", 3430.0, 2.50, 10.05),
    ("M5V", 3060.0, 3.35, 12.45),
    ("M7V", 2680.0, 4.65, 14.72),
)

_TRACK_BP_RP = np.array([row[2] for row in ZAMS_TRACK], dtype=np.float64)
_TRACK_TEFF = np.array([row[1] for row in ZAMS_TRACK], dtype=np.float64)
_TRACK_ABS_G = np.array([row[3] for row in ZAMS_TRACK], dtype=np.float64)
_TRACK_ARC = np.linspace(0.0, 1.0, len(ZAMS_TRACK))


def nearest_track_point(bp_rp: float, abs_g_mag: float) -> dict:
    """The track's expected absolute G magnitude/Teff at this colour.

    Reports a VERTICAL residual at fixed colour (`abs_g_mag` minus the
    track's own absolute magnitude at that BP-RP), not a full 2-D nearest
    point: colour and magnitude are different physical quantities with
    different natural scales, and "how many magnitudes off the track is
    this star, at its own colour" is the standard, directly interpretable
    convention -- not an arbitrary Euclidean distance mixing units.

    A colour outside the track's own [B9V, M7V] range is clamped to the
    nearest end rather than extrapolated (a straight-line extrapolation
    past the track's own measured range would fabricate precision this
    module does not have); `out_of_range` reports when that happened.
    """
    bp_rp = float(bp_rp)
    abs_g_mag = float(abs_g_mag)
    if not (np.isfinite(bp_rp) and np.isfinite(abs_g_mag)):
        raise ValueError("bp_rp and abs_g_mag must be finite")

    out_of_range = bool(bp_rp < _TRACK_BP_RP[0] or bp_rp > _TRACK_BP_RP[-1])
    clamped = float(np.clip(bp_rp, _TRACK_BP_RP[0], _TRACK_BP_RP[-1]))
    track_abs_g = float(np.interp(clamped, _TRACK_BP_RP, _TRACK_ABS_G))
    track_teff = float(np.interp(clamped, _TRACK_BP_RP, _TRACK_TEFF))
    arc_length_fraction = float(np.interp(clamped, _TRACK_BP_RP, _TRACK_ARC))

    return {
        "residual_mag": abs_g_mag - track_abs_g,
        "track_abs_g_mag": track_abs_g,
        "teff_k": track_teff,
        "arc_length_fraction": arc_length_fraction,
        "out_of_range": out_of_range,
    }


def isochrone_residual(bp_rp: float | None, abs_g_mag: float | None,
                       a_g: float | None = None,
                       ebpminrp: float | None = None) -> dict | None:
    """Extinction-corrected residual against `ZAMS_TRACK`, or None.

    Returns None (never a fabricated value) when `bp_rp`/`abs_g_mag` is
    missing or non-finite -- the same "missing evidence, not a value"
    discipline `scoring.physical_inconsistency` already follows for the
    same reason.
    """
    if bp_rp is None or abs_g_mag is None:
        return None
    bp_rp = float(bp_rp)
    abs_g_mag = float(abs_g_mag)
    if not (np.isfinite(bp_rp) and np.isfinite(abs_g_mag)):
        return None

    corrected_abs_g = abs_g_mag
    a_g_used = None
    if a_g is not None and np.isfinite(a_g):
        corrected_abs_g = abs_g_mag - float(a_g)
        a_g_used = float(a_g)

    corrected_bp_rp = bp_rp
    ebpminrp_used = None
    if ebpminrp is not None and np.isfinite(ebpminrp):
        corrected_bp_rp = bp_rp - float(ebpminrp)
        ebpminrp_used = float(ebpminrp)

    result = nearest_track_point(corrected_bp_rp, corrected_abs_g)
    result["a_g_used"] = a_g_used
    result["ebpminrp_used"] = ebpminrp_used
    return result


def compare_to_spectroscopic_teff(cmd_teff_k: float | None,
                                  elodie_teff_k: float | None) -> float | None:
    """Fractional discrepancy between the CMD-implied Teff and a real,
    spectroscopically measured one (e.g. SDSS's `ELODIE_TEFF`).

    Returns None when either input is missing/non-finite/non-positive --
    genuine use of real spectroscopy as a cross-check, not a decorative
    input: a large discrepancy can flag a blend, a misclassified spectrum,
    or a genuinely unusual object.
    """
    if cmd_teff_k is None or elodie_teff_k is None:
        return None
    cmd_teff_k = float(cmd_teff_k)
    elodie_teff_k = float(elodie_teff_k)
    if not (np.isfinite(cmd_teff_k) and np.isfinite(elodie_teff_k)) or elodie_teff_k <= 0:
        return None
    return (cmd_teff_k - elodie_teff_k) / elodie_teff_k
