"""Parity tests: `healpix_common.pixel_probability` must reproduce the three
independent HEALPix computations already shipped in `gw.py`, `frb.py`, and
`association.py`, on shared synthetic fixtures, before any of them is
considered a candidate for switching over to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import association, healpix_common

NSIDE = 32


def _healpix():
    import astropy.units as u
    from astropy_healpix import HEALPix

    return HEALPix(nside=NSIDE, order="nested"), u


class TestDenseProbabilityMapMatchesGwMath:
    """Reproduces gw.py:credible_membership's inline pixel/order/cumsum math."""

    def _synthetic_map(self, rng, peak_pixel, npix):
        # A single dominant pixel plus a long, decaying tail -- enough
        # structure that "the peak pixel" and "everything else" are not
        # degenerate, matching the kind of map build_skymap_from_samples
        # actually produces.
        probability = rng.random(npix) * 0.01
        probability[peak_pixel] = 5.0
        return probability / probability.sum()

    def _gw_style_reference(self, probability, target, nside):
        density = float(probability[target])
        order = np.argsort(probability)[::-1]
        cumulative = np.cumsum(probability[order])
        position = int(np.where(order == target)[0][0])
        credible_level = float(cumulative[position])
        return {"probability_density": density, "credible_level": credible_level,
               "in_90pct_region": credible_level <= 0.90}

    def test_matches_gw_inline_computation_at_the_peak(self):
        healpix, u = _healpix()
        rng = np.random.default_rng(11)
        peak_pixel = 200
        probability = self._synthetic_map(rng, peak_pixel, healpix.npix)
        lon, lat = healpix.healpix_to_lonlat(peak_pixel)
        ra_deg, dec_deg = float(lon.to(u.deg).value), float(lat.to(u.deg).value)

        reference = self._gw_style_reference(probability, peak_pixel, NSIDE)
        result = healpix_common.pixel_probability(
            ra_deg, dec_deg, nside=NSIDE, probability_map=probability)

        assert result["pixel_probability"] == pytest.approx(reference["probability_density"])
        assert result["credible_level"] == pytest.approx(reference["credible_level"])
        assert result["in_credible_region"] == reference["in_90pct_region"]

    def test_matches_gw_inline_computation_at_an_arbitrary_point(self):
        healpix, u = _healpix()
        rng = np.random.default_rng(12)
        peak_pixel = 50
        probability = self._synthetic_map(rng, peak_pixel, healpix.npix)
        target = 137
        lon, lat = healpix.healpix_to_lonlat(target)
        ra_deg, dec_deg = float(lon.to(u.deg).value), float(lat.to(u.deg).value)

        reference = self._gw_style_reference(probability, target, NSIDE)
        result = healpix_common.pixel_probability(
            ra_deg, dec_deg, nside=NSIDE, probability_map=probability)

        assert result["pixel_probability"] == pytest.approx(reference["probability_density"])
        assert result["credible_level"] == pytest.approx(reference["credible_level"])
        assert result["in_credible_region"] == reference["in_90pct_region"]

    def test_empty_map_returns_none(self):
        assert healpix_common.pixel_probability(
            180.0, 10.0, nside=NSIDE, probability_map=np.array([])) is None


class TestSparsePrecomputedCredibleLevelsMatchesFrbMath:
    """Reproduces frb.py:localization_membership's ipix/CL lookup."""

    def test_true_pixel_reports_its_own_confidence_level(self):
        healpix, u = _healpix()
        pixel = 400
        lon, lat = healpix.healpix_to_lonlat(pixel)
        ra_deg, dec_deg = float(lon.to(u.deg).value), float(lat.to(u.deg).value)
        sparse = [{"index": pixel, "confidence_level": 0.1},
                 {"index": pixel + 1, "confidence_level": 0.5}]

        result = healpix_common.pixel_probability(
            ra_deg, dec_deg, nside=NSIDE, sparse_pixels=sparse,
            precomputed_credible_levels=True)

        assert result["credible_level"] == pytest.approx(0.1)
        assert result["in_credible_region"] is True
        assert result["pixel_probability"] is None  # not reported for this shape

    def test_position_outside_the_sparse_map_is_least_confident(self):
        healpix, u = _healpix()
        pixel = 10
        lon, lat = healpix.healpix_to_lonlat(pixel)
        ra_deg, dec_deg = float(lon.to(u.deg).value), float(lat.to(u.deg).value)
        far_lon, far_lat = healpix.healpix_to_lonlat(healpix.npix - 1)

        result = healpix_common.pixel_probability(
            float(far_lon.to(u.deg).value), float(far_lat.to(u.deg).value),
            nside=NSIDE, sparse_pixels=[{"index": pixel, "confidence_level": 0.1}],
            precomputed_credible_levels=True)

        assert result["credible_level"] == 1.0
        assert result["in_credible_region"] is False


class TestSparseRawProbabilityMatchesAssociationMath:
    """Reproduces association.py's own _healpix_probability, called directly."""

    def _sparse_localization(self, rng, n=40, nside=NSIDE):
        indices = rng.choice(12 * nside * nside, size=n, replace=False)
        raw = rng.random(n)
        probability = raw / raw.sum()
        return [{"index": int(i), "probability": float(p)} for i, p in zip(indices, probability)]

    def test_matches_association_inline_computation(self):
        healpix, u = _healpix()
        rng = np.random.default_rng(21)
        pixels = self._sparse_localization(rng)
        target = pixels[5]["index"]
        lon, lat = healpix.healpix_to_lonlat(target)
        ra_deg, dec_deg = float(lon.to(u.deg).value), float(lat.to(u.deg).value)
        localization = {"pixels": pixels, "healpix_nside": NSIDE}

        reference = association._healpix_probability(ra_deg, dec_deg, localization)
        result = healpix_common.pixel_probability(
            ra_deg, dec_deg, nside=NSIDE, sparse_pixels=pixels,
            precomputed_credible_levels=False)

        assert reference is not None
        assert result["pixel_probability"] == pytest.approx(reference["pixel_probability"])
        assert result["credible_level"] == pytest.approx(reference["credible_level"])

    def test_matches_association_for_an_unlisted_target_pixel(self):
        healpix, u = _healpix()
        rng = np.random.default_rng(22)
        pixels = self._sparse_localization(rng)
        listed = {item["index"] for item in pixels}
        unlisted = next(i for i in range(healpix.npix) if i not in listed)
        lon, lat = healpix.healpix_to_lonlat(unlisted)
        ra_deg, dec_deg = float(lon.to(u.deg).value), float(lat.to(u.deg).value)
        localization = {"pixels": pixels, "healpix_nside": NSIDE}

        reference = association._healpix_probability(ra_deg, dec_deg, localization)
        result = healpix_common.pixel_probability(
            ra_deg, dec_deg, nside=NSIDE, sparse_pixels=pixels,
            precomputed_credible_levels=False)

        assert reference["pixel_probability"] == pytest.approx(0.0)
        assert result["pixel_probability"] == pytest.approx(reference["pixel_probability"])
        assert result["credible_level"] == pytest.approx(reference["credible_level"])


class TestInputValidation:
    def test_requires_exactly_one_map_shape(self):
        with pytest.raises(ValueError):
            healpix_common.pixel_probability(180.0, 10.0, nside=NSIDE)
        with pytest.raises(ValueError):
            healpix_common.pixel_probability(
                180.0, 10.0, nside=NSIDE, probability_map=np.ones(12 * NSIDE * NSIDE),
                sparse_pixels=[{"index": 0, "probability": 1.0}])


class TestEffectivePointAndRadius:
    def test_dense_map_recovers_the_peak_pixel_as_the_point_estimate(self):
        healpix, u = _healpix()
        rng = np.random.default_rng(31)
        peak_pixel = 90
        probability = rng.random(healpix.npix) * 0.001
        probability[peak_pixel] = 10.0
        probability /= probability.sum()

        result = healpix_common.effective_point_and_radius(
            nside=NSIDE, probability_map=probability)

        lon, lat = healpix.healpix_to_lonlat(peak_pixel)
        assert result["ra_deg"] == pytest.approx(float(lon.to(u.deg).value), abs=1e-6)
        assert result["dec_deg"] == pytest.approx(float(lat.to(u.deg).value), abs=1e-6)
        assert result["radius_arcsec"] > 0
        assert result["n_pixels"] >= 1

    def test_a_more_concentrated_map_yields_a_smaller_radius(self):
        healpix, u = _healpix()
        npix = healpix.npix
        concentrated = np.full(npix, 1e-6)
        concentrated[100] = 100.0
        concentrated /= concentrated.sum()
        diffuse = np.full(npix, 1.0)
        diffuse /= diffuse.sum()

        tight = healpix_common.effective_point_and_radius(nside=NSIDE, probability_map=concentrated)
        wide = healpix_common.effective_point_and_radius(nside=NSIDE, probability_map=diffuse)

        assert tight["radius_arcsec"] < wide["radius_arcsec"]

    def test_sparse_precomputed_levels_shape(self):
        sparse = [{"index": 5, "confidence_level": 0.05},
                 {"index": 6, "confidence_level": 0.3},
                 {"index": 7, "confidence_level": 0.6}]
        result = healpix_common.effective_point_and_radius(
            nside=NSIDE, sparse_pixels=sparse, precomputed_credible_levels=True,
            credible_fraction=0.39)
        assert result is not None
        assert result["n_pixels"] == 2  # pixels 5 (CL 0.05) and 6 (CL 0.3) are <= 0.39

    def test_empty_inputs_return_none(self):
        assert healpix_common.effective_point_and_radius(
            nside=NSIDE, probability_map=np.array([])) is None
        assert healpix_common.effective_point_and_radius(
            nside=NSIDE, sparse_pixels=[]) is None
