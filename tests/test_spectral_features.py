from __future__ import annotations

import json

import numpy as np

from astra import spectral_features


def test_extract_reports_continuum_and_line_statistics():
    wave = np.linspace(5000.0, 5100.0, 1000)
    flux = np.ones_like(wave)
    flux[500] += 10.0
    error = np.full_like(wave, 0.1)
    payload = spectral_features.extract(wave, flux, error, frame="vacuum", units="adu")

    assert payload["schema_version"] == 1
    assert payload["features"]["points"] == 1000
    assert payload["features"]["max_positive_line_snr"] > 5
    assert payload["quality"]["error_fallback"] is False
    json.dumps(payload)


def test_from_fits_reads_ivar_and_persists_provenance(tmp_path):
    from astropy.io import fits

    columns = fits.ColDefs([
        fits.Column(name="wavelength", format="D", array=np.linspace(4000, 4010, 20)),
        fits.Column(name="flux", format="D", array=np.ones(20)),
        fits.Column(name="ivar", format="D", array=np.full(20, 100.0)),
    ])
    path = tmp_path / "spectrum.fits"
    fits.BinTableHDU.from_columns(columns).writeto(path)
    payload = spectral_features.from_fits(path)
    output = spectral_features.save(payload, tmp_path / "out")

    assert payload["source"]["sha256"]
    assert payload["quality"]["error_fallback"] is False
    assert output.exists()
