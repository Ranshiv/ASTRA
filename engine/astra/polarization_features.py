"""Polarization feature pipeline (roadmap item 28, P2).

Neither an "optical polarimetry archive" nor a "radio Faraday data"
service in the generic sense this roadmap item names has one obvious
credential-free API the way DES/Pan-STARRS do -- checked live this
session via `astroquery.vizier.Vizier.find_catalogs` (the same discovery
tool `catalogs.py`/`vlass.py` already used), following this session's
"best available real substitute, documented" instruction. Two real,
VizieR-hosted catalogues confirmed live:

  - `J/ApJ/728/104` ("Optical polarization for 878 Hipparcos stars",
    Santos et al. 2011, ApJ 728, 104) -- real `PV`/`e_PV` (percent linear
    polarization) and `PA`/`e_PA` (position angle, degrees) columns,
    confirmed live via a real cone search.
  - `J/other/RAA/14.942` ("Rotation measures of radio point sources",
    Xu & Han 2014, RAA 14, 942) -- real `RM`/`e_RM` (rad/m^2) columns,
    confirmed live.

Both are catalogue-only (no time series), so they are exposed as bounded
query helpers (`query_optical_polarization`/`query_rotation_measure`)
following `surveys/vlass.py`'s `query_nvss_flux_1_4ghz` pattern -- a full
`SurveyConnector` is for light-curve/catalogue survey membership, not a
one-off cross-match helper, the same distinction that module already
draws for NVSS. TESS/ZTF timing (already real, already acquired
elsewhere in this codebase) is folded in only as an optional extra
feature dimension for the separability check below, not as a new
acquisition path.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import netclient
from .tap import parse_votable

SCHEMA_VERSION = 1
SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
OPTICAL_POLARIZATION_CATALOG = "J/ApJ/728/104"
# A second, independent real optical polarimetry catalogue -- "The linear
# polarization of nearby bright stars measured at the parts per million
# level" (Bailey, Lucas & Hough 2010, MNRAS 405, 2570; real title/authors
# confirmed live this session via the arXiv API, abs/1003.1753 -- VizieR's
# own catalogue metadata under-titles it "Linear polarization of nearby
# bright stars"). Units were confirmed live this session, not assumed: the
# paper's own abstract reports "BS 3982 (Regulus) has a polarization of
# ~37x10^-6"; this catalogue's real row for `HR=3982` (the same star --
# HR/BS are the same Bright Star/Harvard Revised numbering) has `Pol=36.7`
# -- matching the paper's quoted value to its own stated precision and
# confirming `Pol`/`Q/I`/`U/I` are all in units of 1e-6 (parts per
# million), NOT percent or 1e-4 as earlier guessed. `_PPM_TO_PERCENT`
# converts `Pol` into the same `p_percent` scale `query_optical_
# polarization` (Santos+2011) already uses, now that the conversion is
# real rather than assumed.
OPTICAL_POLARIZATION_CATALOG_SECONDARY = "J/MNRAS/405/2570"
_PPM_TO_PERCENT = 1e-4  # 1 part-per-million = 1e-6 = 1e-4 percent
ROTATION_MEASURE_CATALOG = "J/other/RAA/14.942"


def query_optical_polarization(ra_deg: float, dec_deg: float, radius_arcsec: float = 30.0
                               ) -> dict | None:
    """The nearest real optical linear-polarization measurement within
    `radius_arcsec`. Returns `None` (not a fabricated zero) when no
    catalogue source falls within the search radius -- the Santos+2011
    sample covers only 878 Hipparcos stars, so most sky positions have no
    match, which is expected sparsity, not a service failure."""
    response = netclient.get(
        SCS_URL,
        {"-source": OPTICAL_POLARIZATION_CATALOG, "RA": ra_deg, "DEC": dec_deg,
         "SR": radius_arcsec / 3600.0, "-out.max": 1, "-out.orderby": "_r"},
        timeout=60, provider="vizier",
    )
    rows = parse_votable(response.text, 1)
    if not rows:
        return None
    row = rows[0]
    try:
        p_percent, theta_deg = float(row["PV"]), float(row["PA"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "hip": row.get("HIP"), "p_percent": p_percent, "p_error_percent": row.get("e_PV"),
        "theta_deg": theta_deg, "theta_error_deg": row.get("e_PA"),
    }


def query_optical_polarization_secondary(ra_deg: float, dec_deg: float, radius_arcsec: float = 30.0
                                         ) -> dict | None:
    """The nearest real Bailey et al. (2010) polarimetry measurement
    within `radius_arcsec`. Returns both the catalogue's own raw `Pol`
    (parts-per-million, confirmed live -- see this module's catalogue-
    constant comment) AND `p_percent`/`theta_deg` already converted to
    the same scale `query_optical_polarization` uses, so callers can use
    either function's result interchangeably."""
    response = netclient.get(
        SCS_URL,
        {"-source": OPTICAL_POLARIZATION_CATALOG_SECONDARY, "RA": ra_deg, "DEC": dec_deg,
         "SR": radius_arcsec / 3600.0, "-out.max": 1, "-out.orderby": "_r"},
        timeout=60, provider="vizier",
    )
    rows = parse_votable(response.text, 1)
    if not rows:
        return None
    row = rows[0]
    try:
        pol_raw, theta_deg = float(row["Pol"]), float(row["theta"])
    except (KeyError, TypeError, ValueError):
        return None
    pol_error_raw = row.get("e_Pol")
    try:
        p_error_percent = float(pol_error_raw) * _PPM_TO_PERCENT if pol_error_raw is not None else None
    except (TypeError, ValueError):
        p_error_percent = None
    return {
        "hr": row.get("HR"), "pol_raw_ppm": pol_raw, "pol_raw_ppm_error": pol_error_raw,
        "p_percent": pol_raw * _PPM_TO_PERCENT, "p_error_percent": p_error_percent,
        "theta_deg": theta_deg, "theta_error_deg": row.get("e_theta"),
        "spectral_type": row.get("SpType"),
    }


def query_optical_polarization_any(ra_deg: float, dec_deg: float, radius_arcsec: float = 30.0
                                   ) -> dict | None:
    """Try the primary (Santos+2011) catalogue first, then the secondary
    (Bailey+2010) one -- broader real coverage than either catalogue
    alone. Both results now share a common `p_percent`/`theta_deg`
    schema (Bailey+2010's real ppm-to-percent conversion confirmed live
    this session), so `source_catalog` names only WHICH catalogue
    matched, not a schema difference callers must branch on."""
    primary = query_optical_polarization(ra_deg, dec_deg, radius_arcsec)
    if primary is not None:
        return {**primary, "source_catalog": OPTICAL_POLARIZATION_CATALOG}
    secondary = query_optical_polarization_secondary(ra_deg, dec_deg, radius_arcsec)
    if secondary is not None:
        return {**secondary, "source_catalog": OPTICAL_POLARIZATION_CATALOG_SECONDARY}
    return None


def query_rotation_measure(ra_deg: float, dec_deg: float, radius_arcsec: float = 60.0
                           ) -> dict | None:
    """The nearest real Faraday rotation-measure detection within
    `radius_arcsec`. `None` when no catalogue source is within range."""
    response = netclient.get(
        SCS_URL,
        {"-source": ROTATION_MEASURE_CATALOG, "RA": ra_deg, "DEC": dec_deg,
         "SR": radius_arcsec / 3600.0, "-out.max": 1, "-out.orderby": "_r"},
        timeout=60, provider="vizier",
    )
    rows = parse_votable(response.text, 1)
    if not rows:
        return None
    row = rows[0]
    try:
        rm_rad_m2 = float(row["RM"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"rm_rad_per_m2": rm_rad_m2, "rm_error_rad_per_m2": row.get("e_RM"),
           "telescope": row.get("Tel")}


# A second, much larger real RM catalogue -- Taylor, Stil & Sunstrum
# (2009, ApJ 702, 1230), the real all-sky NVSS RM catalogue of 37,543
# sources widely cited in the literature as "the" NVSS RM catalogue --
# found live this session via a direct `J/ApJ/702/1230` VizieR lookup
# after `Vizier.find_catalogs` text search did not surface it under
# several search-term phrasings tried. Confirmed live: real `RM`/`e_RM`
# columns, but no per-source `Tel` (telescope) column the way Xu+2014 has
# -- every row is an NVSS detection by construction, so this catalogue's
# own results never populate that field.
ROTATION_MEASURE_CATALOG_NVSS = "J/ApJ/702/1230"


def query_rotation_measure_nvss(ra_deg: float, dec_deg: float, radius_arcsec: float = 60.0
                                ) -> dict | None:
    """The nearest real Taylor+2009 NVSS rotation-measure detection
    within `radius_arcsec` -- the much larger (37,543-source) real
    catalogue, checked as the primary source by `query_rotation_
    measure_any` before falling back to the smaller Xu+2014 sample."""
    response = netclient.get(
        SCS_URL,
        {"-source": ROTATION_MEASURE_CATALOG_NVSS, "RA": ra_deg, "DEC": dec_deg,
         "SR": radius_arcsec / 3600.0, "-out.max": 1, "-out.orderby": "_r"},
        timeout=60, provider="vizier",
    )
    rows = parse_votable(response.text, 1)
    if not rows:
        return None
    row = rows[0]
    try:
        rm_rad_m2 = float(row["RM"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"rm_rad_per_m2": rm_rad_m2, "rm_error_rad_per_m2": row.get("e_RM"),
           "stokes_i_flux_mjy": row.get("Si"), "polarized_flux_mjy": row.get("Pk")}


def query_rotation_measure_any(ra_deg: float, dec_deg: float, radius_arcsec: float = 60.0
                               ) -> dict | None:
    """Try the large Taylor+2009 NVSS catalogue first, then fall back to
    Xu+2014 -- broader real coverage than either alone, same
    `source_catalog`-tagged pattern as `query_optical_polarization_any`."""
    primary = query_rotation_measure_nvss(ra_deg, dec_deg, radius_arcsec)
    if primary is not None:
        return {**primary, "source_catalog": ROTATION_MEASURE_CATALOG_NVSS}
    secondary = query_rotation_measure(ra_deg, dec_deg, radius_arcsec)
    if secondary is not None:
        return {**secondary, "source_catalog": ROTATION_MEASURE_CATALOG}
    return None


def stokes_from_p_theta(p_percent: float, theta_deg: float) -> tuple[float, float]:
    """Reduced Stokes parameters q=Q/I, u=U/I from the catalogue's
    (percent polarization, position angle) representation -- the standard
    `q = p*cos(2*theta)`, `u = p*sin(2*theta)` relation (e.g. Wardle &
    Kronberg 1974, ApJ 194, 249), `p` as a fraction."""
    p_fraction = p_percent / 100.0
    theta_rad = math.radians(theta_deg)
    return p_fraction * math.cos(2.0 * theta_rad), p_fraction * math.sin(2.0 * theta_rad)


def p_theta_from_stokes(q: float, u: float) -> tuple[float, float]:
    """Inverse of `stokes_from_p_theta`, for round-trip validation."""
    p_fraction = math.hypot(q, u)
    theta_deg = math.degrees(0.5 * math.atan2(u, q)) % 180.0
    return p_fraction * 100.0, theta_deg


def polarization_angle_error_deg(p_percent: float, p_error_percent: float) -> float | None:
    """The standard linear-polarization angle-error approximation
    (Serkowski 1958; Wardle & Kronberg 1974, ApJ 194, 249):
    `sigma_theta[deg] = 28.65 * sigma_p / p` (radian form `0.5/SNR`
    converted to degrees), valid for p well above its own noise floor.
    Returns `None` when `p_percent` is non-positive (angle is undefined
    at zero polarization, not a fabricated large error)."""
    if p_percent <= 0 or p_error_percent < 0:
        return None
    return round(28.65 * p_error_percent / p_percent, 3)


def _synthetic_polarization_features(rng: np.random.Generator, n: int, class_label: str
                                     ) -> tuple[np.ndarray, list[str]]:
    """Synthetic (p_percent, |RM|, variability_amplitude) feature triples
    per class -- a stated, class-conditional assumption for validating the
    CLASSIFIER MECHANISM, not a trained real-population model: an
    unpolarized/weakly-polarized normal star; a magnetically active/
    dusty variable with modest polarization and real photometric
    variability; a blazar-like source with high, RM-associated
    polarization and strong variability.
    """
    if class_label == "unpolarized_star":
        p = np.clip(rng.normal(0.3, 0.2, n), 0.0, None)
        rm = np.abs(rng.normal(5.0, 3.0, n))
        amplitude = np.abs(rng.normal(0.02, 0.01, n))
    elif class_label == "polarized_variable":
        p = np.clip(rng.normal(2.0, 0.8, n), 0.0, None)
        rm = np.abs(rng.normal(15.0, 8.0, n))
        amplitude = np.abs(rng.normal(0.15, 0.05, n))
    else:  # blazar_like
        p = np.clip(rng.normal(6.0, 2.0, n), 0.0, None)
        rm = np.abs(rng.normal(60.0, 25.0, n))
        amplitude = np.abs(rng.normal(0.4, 0.15, n))
    return np.column_stack([p, rm, amplitude]), [class_label] * n


def evaluate_class_separability(*, n_per_class: int = 150, seed: int = 23,
                                test_fraction: float = 0.3) -> dict[str, Any]:
    """Bounded logistic probe (same shape as
    `sed_energy_balance.evaluate_class_likelihood_calibration`) over
    (polarization degree, |RM|, photometric variability amplitude)
    features on synthetic class-conditional distributions -- the "source-
    class separability" metric this roadmap item names. Mechanism
    validation only, not a trained real-population classifier.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    classes = ["unpolarized_star", "polarized_variable", "blazar_like"]
    features_list, labels_list = [], []
    for class_label in classes:
        features, labels = _synthetic_polarization_features(rng, n_per_class, class_label)
        features_list.append(features)
        labels_list.extend(labels)
    features_all = np.vstack(features_list)
    labels_all = np.asarray(labels_list)

    n_total = len(labels_all)
    order = rng.permutation(n_total)
    n_test = max(1, int(round(n_total * test_fraction)))
    test_idx, train_idx = order[:n_test], order[n_test:]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(features_all[train_idx], labels_all[train_idx])
    predictions = clf.predict(features_all[test_idx])
    return {
        "classes": classes, "n_train": len(train_idx), "n_test": n_test,
        "macro_f1": round(float(f1_score(labels_all[test_idx], predictions, average="macro")), 4),
    }


def evaluate_real_class_separability(*, class_a: str = "AGN", class_b: str = "YSO",
                                     n_per_class: int = 15, rm_radius_arcsec: float = 120.0,
                                     test_fraction: float = 0.3, seed: int = 11
                                     ) -> dict[str, Any]:
    """The real-data counterpart to `evaluate_class_separability`: real
    ALeRCE-broker-classified objects (`surveys/alerce.py`'s
    `query_classified_objects`, the same real-label precedent
    `sed_energy_balance.evaluate_real_class_separability` already uses),
    each contributing two real features -- the nearest real NVSS rotation
    measure (`query_rotation_measure_nvss`, |RM| in rad/m^2, 0.0 when no
    real match exists within `rm_radius_arcsec`, since a genuine
    non-detection is real information a classifier can use, not a
    fabricated positive) and the real photometric scatter of the
    object's own ALeRCE-fetched ZTF light curve (std of the first
    available band's real magnitudes, a real variability-amplitude
    proxy). `class_a`/`class_b` default to "AGN"/"YSO", the same two
    classes `sed_energy_balance`'s real study already confirmed live to
    return real ALeRCE rows. Genuinely different real-data behaviour
    expected from that other study: AGN are the class most likely to have
    a real NVSS RM detection at all (radio-loud AGN are common NVSS
    sources); YSOs mostly are not, so many YSO rows are expected to carry
    `rm_rad_per_m2=0.0` -- a real class-conditional signal, not a bug.
    Small `n_per_class` default and not part of the offline suite for the
    same reason as `sed_energy_balance`'s real study: this hits multiple
    real, rate-limited services per object.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    from .surveys.alerce import ALeRCEConnector

    alerce = ALeRCEConnector()
    features_list: list[list[float]] = []
    labels_list: list[str] = []
    n_no_rm = 0
    n_no_curve = 0
    for class_label in (class_a, class_b):
        objects = alerce.query_classified_objects(class_label, limit=n_per_class)
        for obj in objects:
            rm_match = query_rotation_measure_nvss(obj.ra_deg, obj.dec_deg, rm_radius_arcsec)
            abs_rm = abs(rm_match["rm_rad_per_m2"]) if rm_match is not None else 0.0
            n_no_rm += rm_match is None

            curves = alerce.fetch_light_curves(obj)
            amplitude = 0.0
            if curves:
                finite = curves[0].dropna()
                if len(finite) >= 3:
                    amplitude = float(np.std(finite.value))
                else:
                    n_no_curve += 1
            else:
                n_no_curve += 1

            features_list.append([abs_rm, amplitude])
            labels_list.append(class_label)

    if len(set(labels_list)) < 2 or len(labels_list) < 6:
        return {"classes": [class_a, class_b], "n_used": len(labels_list),
               "n_no_rm_match": n_no_rm, "n_no_light_curve": n_no_curve,
               "macro_f1": None, "warnings": ["insufficient real data for a real fit"]}

    features = np.asarray(features_list)
    labels = np.asarray(labels_list)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    n_test = max(1, int(round(len(labels) * test_fraction)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    if len(set(labels[train_idx])) < 2:
        return {"classes": [class_a, class_b], "n_used": len(labels_list),
               "n_no_rm_match": n_no_rm, "n_no_light_curve": n_no_curve,
               "macro_f1": None, "warnings": ["training split did not contain both classes"]}

    clf = LogisticRegression(max_iter=1000)
    clf.fit(features[train_idx], labels[train_idx])
    predictions = clf.predict(features[test_idx])
    return {
        "classes": [class_a, class_b], "n_used": len(labels_list),
        "n_no_rm_match": n_no_rm, "n_no_light_curve": n_no_curve,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "macro_f1": round(float(f1_score(labels[test_idx], predictions, average="macro")), 4),
    }
