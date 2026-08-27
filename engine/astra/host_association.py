"""Probabilistic host-galaxy association (roadmap item 31, P1).

Ranks candidate host galaxies for a transient position with a Bayesian
combination of positional offset, luminosity, and redshift evidence,
following the structure of the PATH algorithm (Gagliano et al. 2021, ApJ
908, 170 -- "Probabilistic Association of Transients to their Hosts") in
its bounded, Sersic-fit-free fallback mode: ASTRA has no image-based
profile-fitting stage (that is `tess_psf.py`'s domain, and only for TESS
target pixels, not deep-imaging cutouts), so each candidate galaxy is
modelled as a single circular exponential-disk light profile rather than a
fitted Sersic ellipse. This is a real, stated scope limit, not a hidden
simplification -- it is the same fallback PATH's own paper uses when a
per-galaxy Sersic fit is unavailable.

For each candidate galaxy `i`, the (unnormalized) posterior weight is

    P(host = i) proportional to P(offset | i) * P(luminosity | i, z_i) * P(i is a galaxy)

normalized against the other candidates plus an explicit `prior_no_host`
probability mass reserved for "the true host is not in the candidate
list at all" -- the same residual term PATH's own model carries.

`P(offset | i)` is the closed-form radial marginal of a 2D exponential
surface-brightness profile of scale radius `r_e`. For a profile
`I(r) = I0 * exp(-r / r_e)`, the total 2D flux is
`integral_0^inf I(r) * 2*pi*r dr = I0 * 2*pi*r_e**2`
(substituting `u = r / r_e` reduces the integral to
`integral_0^inf u * exp(-u) du = Gamma(2) = 1`). Treating "probability the
transient landed at radius r" as proportional to the light enclosed in an
annulus at that radius gives the normalized radial density
`f(r) = (r / r_e**2) * exp(-r / r_e)`, which integrates to exactly 1 over
`r in [0, inf)` by the same substitution -- verified directly in this
module's own tests, not merely asserted here.

`P(luminosity | i, z_i)` is a Schechter (1976, ApJ 203, 297) luminosity
function evaluated at the candidate's absolute magnitude, using real,
citable SDSS r-band parameters (Blanton et al. 2003, ApJ 592, 819, Table
2, low-redshift sample: `M* = -20.44`, `alpha = -1.05`, in the paper's
`h = 1` convention). Only the shape term `x**(alpha+1) * exp(-x)` (with
`x = 10**(0.4*(M* - M))`) is computed -- the `phi*` normalization constant
is dropped because this function is only ever used for *relative* ranking
among candidates sharing the same cosmology and band, never as an
absolute number density.

`P(i is a galaxy)` is a hard veto (weight 0) applied when a candidate
cross-matches a Gaia source with a statistically significant parallax
(`|parallax_snr| >= FOREGROUND_PARALLAX_SNR_THRESHOLD`) -- Gaia's
astrometric solution is essentially only ever measured for foreground
Galactic point sources, so a significant parallax at a "galaxy"
candidate's position is real evidence the DES/Pan-STARRS detection is
actually a foreground star, not a host galaxy. A hard veto (rather than a
soft downweight) is a deliberate, stated choice: this module has no
further evidence to partially trust a flagged candidate against.

Data sources are all real, already-existing connectors, reused unchanged:
`surveys/des.py`/`surveys/panstarrs.py` (deep-imaging candidate
positions/photometry/size), `surveys/sdss.py`/`surveys/desi.py`
(spectroscopic redshift), `surveys/gaia.py` (foreground-star veto),
`surveys/alerce.py` (real ZTF transient positions, used by the caller to
supply `ra_deg`/`dec_deg`, not imported here). `crossmatch.
angular_separation_arcsec` supplies the offset geometry, reused unchanged.

Two scope limits stated up front, not glossed over: (1) when no candidate
size measurement is available, `DEFAULT_R_E_ARCSEC` is used as an explicit
FALLBACK assumption, not a real per-galaxy measurement -- `HostCandidate.
r_e_arcsec` stays `None` in that case so a caller can see the difference;
(2) when no redshift is matched for a candidate, the luminosity term is
left neutral (1.0) and the association degrades to offset-only evidence,
again visible via `HostCandidate.redshift is None` rather than silently
assumed. `find_and_associate_host`'s galaxy size proxy comes from DES
only (`surveys/des.py`'s `flux_radius_r`, confirmed live this session
against a real `TAP_SCHEMA.columns` query and converted from its real
pixel unit to arcsec -- see that module's docstring for the column-name
and unit bug this found and fixed). Pan-STARRS contributes no size proxy:
a live check this session confirmed its `mean` object endpoint (the one
this codebase's `PanSTARRSConnector` queries) publishes no Kron-radius or
other size column at all -- only the separate `stack` endpoint does,
which this connector does not query -- so every Pan-STARRS-only candidate
falls back to `DEFAULT_R_E_ARCSEC`, a real, confirmed (not merely
unverified) data-source gap, stated here rather than silently assumed
fixed.

Like every other opt-in research module in this codebase, NOT wired into
`rpc.py`, `scoring.WEIGHTS`, or `evidence.py`.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

from .crossmatch import angular_separation_arcsec
from .surveys.base import SourceRef

# Blanton et al. 2003 (ApJ 592, 819), Table 2, SDSS r-band, h=1 convention.
DEFAULT_M_STAR = -20.44
DEFAULT_ALPHA = -1.05

# Fallback half-light radius (arcsec) used only when no real size
# measurement is available for a candidate -- a stated assumption, not a
# measurement (see module docstring).
DEFAULT_R_E_ARCSEC = 1.5

# |parallax / parallax_error| at or above this is treated as a
# statistically significant astrometric solution -- a standard 3-sigma
# detection threshold, not a tuned value.
FOREGROUND_PARALLAX_SNR_THRESHOLD = 3.0

# Preference order for reading one scalar apparent magnitude out of a
# DES/Pan-STARRS SourceRef.extra dict -- r-band first, matching the band
# the Schechter parameters above are calibrated in.
_MAGNITUDE_KEYS = ("r_mean", "g_mean", "i_mean", "z_mean", "y_mean")

# Column populated by find_and_associate_host's DES candidates, already
# converted to arcsec by `surveys/des.py` (confirmed live this session:
# the real column is per-band and in pixels, not a bare arcsec value --
# see that module's docstring). Pan-STARRS supplies no size column: a
# live check this session confirmed the `mean` endpoint this connector
# queries has no Kron-radius field at all (only `stack`, which this
# connector does not query) -- see `surveys/panstarrs.py`'s docstring.
_SIZE_KEYS = ("flux_radius_r_arcsec",)


class HostAssociationError(ValueError):
    """A host-association input or computation was invalid."""


def default_cosmology():
    """The fallback cosmology used whenever a caller does not supply one --
    a plain `astropy.cosmology` instance, following `strong_lens.py`'s own
    convention that distances/cosmology are ultimately the caller's
    concern (`astropy.cosmology` is already a core dependency)."""
    from astropy.cosmology import Planck18
    return Planck18


def exponential_offset_likelihood(offset_arcsec: float, r_e_arcsec: float) -> float:
    """Normalized radial density of a 2D exponential light profile at a
    given offset -- `f(r) = (r / r_e**2) * exp(-r / r_e)`, derived and
    verified to integrate to 1 in the module docstring/tests."""
    if r_e_arcsec <= 0:
        raise HostAssociationError(f"r_e_arcsec must be positive, got {r_e_arcsec}")
    if offset_arcsec < 0:
        raise HostAssociationError(f"offset_arcsec must be non-negative, got {offset_arcsec}")
    return (offset_arcsec / r_e_arcsec ** 2) * math.exp(-offset_arcsec / r_e_arcsec)


def schechter_luminosity_prior(abs_mag: float, m_star: float = DEFAULT_M_STAR,
                               alpha: float = DEFAULT_ALPHA) -> float:
    """Unnormalized Schechter-function relative density at an absolute
    magnitude -- `phi*` is dropped; only relative ranking is needed."""
    if not math.isfinite(abs_mag):
        raise HostAssociationError(f"abs_mag must be finite, got {abs_mag}")
    x = 10.0 ** (0.4 * (m_star - abs_mag))
    return (x ** (alpha + 1.0)) * math.exp(-x)


def absolute_magnitude(apparent_mag: float, redshift: float, cosmology=None) -> float:
    """Standard distance-modulus conversion: `M = m - distmod(z)`."""
    if not (redshift > 0 and math.isfinite(redshift)):
        raise HostAssociationError(f"redshift must be a positive finite value, got {redshift}")
    cosmology = cosmology or default_cosmology()
    return apparent_mag - cosmology.distmod(redshift).value


def _apparent_magnitude(candidate: SourceRef) -> float | None:
    for key in _MAGNITUDE_KEYS:
        value = candidate.extra.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


@dataclass(frozen=True)
class HostCandidate:
    source: SourceRef
    offset_arcsec: float
    r_e_arcsec: float | None
    redshift: float | None
    redshift_source: str | None
    abs_mag: float | None
    likely_foreground_star: bool
    posterior_probability: float

    def to_dict(self) -> dict:
        return {
            "survey": self.source.survey, "object_id": self.source.object_id,
            "ra_deg": self.source.ra_deg, "dec_deg": self.source.dec_deg,
            "offset_arcsec": round(self.offset_arcsec, 4),
            "r_e_arcsec": self.r_e_arcsec,
            "redshift": self.redshift, "redshift_source": self.redshift_source,
            "abs_mag": self.abs_mag,
            "likely_foreground_star": self.likely_foreground_star,
            "posterior_probability": round(self.posterior_probability, 6),
        }


@dataclass(frozen=True)
class HostAssociationResult:
    transient: SourceRef
    candidates: list[HostCandidate] = field(default_factory=list)
    no_host_probability: float = 0.0

    def to_dict(self) -> dict:
        return {
            "transient": {"survey": self.transient.survey, "object_id": self.transient.object_id,
                          "ra_deg": self.transient.ra_deg, "dec_deg": self.transient.dec_deg},
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "no_host_probability": round(self.no_host_probability, 6),
        }


def associate_host(transient: SourceRef, candidates: list[SourceRef], *,
                   redshifts: dict[str, tuple[float, str]] | None = None,
                   r_e_arcsec: dict[str, float] | None = None,
                   foreground_flags: dict[str, bool] | None = None,
                   prior_no_host: float = 0.05,
                   m_star: float = DEFAULT_M_STAR, alpha: float = DEFAULT_ALPHA,
                   cosmology=None) -> HostAssociationResult:
    """The pure Bayesian-combine step: no network calls, every per-candidate
    input (redshift, size, foreground flag) is supplied by the caller as a
    plain dict keyed by `SourceRef.object_id` -- kept separate from
    `find_and_associate_host` so the math stays testable without mocking
    four connectors at once (the same split `sn_classification.py`'s MCMC
    fit keeps from its ALeRCE-fetching caller)."""
    if not 0.0 <= prior_no_host < 1.0:
        raise HostAssociationError(f"prior_no_host must be in [0, 1), got {prior_no_host}")
    redshifts = redshifts or {}
    r_e_arcsec = r_e_arcsec or {}
    foreground_flags = foreground_flags or {}

    built: list[HostCandidate] = []
    raw_scores: list[float] = []
    for candidate in candidates:
        offset = angular_separation_arcsec(
            transient.ra_deg, transient.dec_deg, candidate.ra_deg, candidate.dec_deg)

        measured_r_e = r_e_arcsec.get(candidate.object_id)
        effective_r_e = measured_r_e if measured_r_e and measured_r_e > 0 else DEFAULT_R_E_ARCSEC
        offset_term = exponential_offset_likelihood(offset, effective_r_e)

        redshift_val = redshift_source = abs_mag = None
        luminosity_term = 1.0
        z_info = redshifts.get(candidate.object_id)
        if z_info is not None:
            redshift_val, redshift_source = z_info
            apparent_mag = _apparent_magnitude(candidate)
            if apparent_mag is not None:
                abs_mag = absolute_magnitude(apparent_mag, redshift_val, cosmology)
                luminosity_term = schechter_luminosity_prior(abs_mag, m_star, alpha)

        is_star = bool(foreground_flags.get(candidate.object_id, False))
        galaxy_term = 0.0 if is_star else 1.0

        raw_scores.append(offset_term * luminosity_term * galaxy_term)
        built.append(HostCandidate(
            source=candidate, offset_arcsec=offset, r_e_arcsec=measured_r_e,
            redshift=redshift_val, redshift_source=redshift_source, abs_mag=abs_mag,
            likely_foreground_star=is_star, posterior_probability=0.0,
        ))

    total = sum(raw_scores)
    if total > 0:
        posteriors = [(1.0 - prior_no_host) * score / total for score in raw_scores]
        no_host_probability = prior_no_host
    else:
        # No candidate carries any positive evidence (e.g. every candidate
        # vetoed as a foreground star, or none supplied) -- all the
        # posterior mass is genuinely "the true host is not here."
        posteriors = [0.0 for _ in raw_scores]
        no_host_probability = 1.0

    finished = sorted(
        (dataclasses.replace(candidate, posterior_probability=p)
         for candidate, p in zip(built, posteriors)),
        key=lambda c: c.posterior_probability, reverse=True,
    )
    return HostAssociationResult(
        transient=transient, candidates=finished, no_host_probability=no_host_probability)


def _extract_r_e_arcsec(source: SourceRef) -> float | None:
    for key in _SIZE_KEYS:
        value = source.extra.get(key)
        if value is None:
            continue
        try:
            size = float(value)
        except (TypeError, ValueError):
            continue
        if size > 0:
            return size
    return None


def find_and_associate_host(ra_deg: float, dec_deg: float, radius_arcsec: float = 30.0, *,
                            redshift_radius_arcsec: float = 3.0,
                            foreground_radius_arcsec: float = 2.0,
                            prior_no_host: float = 0.05,
                            cosmology=None) -> HostAssociationResult:
    """Thin network-orchestration layer around `associate_host`: pulls
    candidate galaxies from DES/Pan-STARRS, a matching spec-z from
    SDSS/DESI, and a Gaia foreground-star check, per candidate -- one
    SDSS + one DESI + one Gaia query per candidate, bounded by how many
    candidates a small `radius_arcsec` field returns, not a batch
    endpoint. Acceptable for a research-scale, one-transient-at-a-time
    call; not swept into any bulk pipeline, matching this module's own
    stated opt-in scope."""
    from .surveys.base import ConeQuery
    from .surveys.des import DESConnector
    from .surveys.desi import DESIConnector
    from .surveys.gaia import GaiaConnector, derived_properties
    from .surveys.panstarrs import PanSTARRSConnector
    from .surveys.sdss import query_spectroscopic_redshifts

    transient = SourceRef(survey="query", object_id="transient", ra_deg=ra_deg, dec_deg=dec_deg)
    field_query = ConeQuery(ra_deg=ra_deg, dec_deg=dec_deg, radius_arcsec=radius_arcsec)

    candidates: list[SourceRef] = []
    r_e_arcsec: dict[str, float] = {}
    for connector in (DESConnector(), PanSTARRSConnector()):
        for candidate in connector.cone_search(field_query):
            candidates.append(candidate)
            size = _extract_r_e_arcsec(candidate)
            if size is not None:
                r_e_arcsec[candidate.object_id] = size

    redshifts: dict[str, tuple[float, str]] = {}
    for candidate in candidates:
        for source in query_spectroscopic_redshifts(
                candidate.ra_deg, candidate.dec_deg, redshift_radius_arcsec):
            z = source.extra.get("z")
            if z is not None:
                redshifts[candidate.object_id] = (float(z), "sdss")
                break
        if candidate.object_id in redshifts:
            continue
        near = ConeQuery(ra_deg=candidate.ra_deg, dec_deg=candidate.dec_deg,
                         radius_arcsec=redshift_radius_arcsec)
        for source in DESIConnector().cone_search(near):
            z = source.extra.get("z")
            if z is not None:
                redshifts[candidate.object_id] = (float(z), "desi")
                break

    foreground_flags: dict[str, bool] = {}
    for candidate in candidates:
        near = ConeQuery(ra_deg=candidate.ra_deg, dec_deg=candidate.dec_deg,
                         radius_arcsec=foreground_radius_arcsec)
        for source in GaiaConnector().cone_search(near, limit=1):
            snr = derived_properties(source.extra).get("parallax_snr")
            if snr is not None and abs(snr) >= FOREGROUND_PARALLAX_SNR_THRESHOLD:
                foreground_flags[candidate.object_id] = True

    return associate_host(
        transient, candidates, redshifts=redshifts, r_e_arcsec=r_e_arcsec,
        foreground_flags=foreground_flags, prior_no_host=prior_no_host, cosmology=cosmology)


__all__ = [
    "HostAssociationError", "HostCandidate", "HostAssociationResult",
    "exponential_offset_likelihood", "schechter_luminosity_prior", "absolute_magnitude",
    "default_cosmology", "associate_host", "find_and_associate_host",
    "DEFAULT_M_STAR", "DEFAULT_ALPHA", "DEFAULT_R_E_ARCSEC",
    "FOREGROUND_PARALLAX_SNR_THRESHOLD",
]
