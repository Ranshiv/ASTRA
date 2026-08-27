"""Live, on-demand contract check: does `artifact_bank.extract_camera_ccd`
actually parse real `CAMERA`/`CCD` values from a real downloaded TPF?

Not run by pytest's default collection (see the `live` marker registered in
`tests/conftest.py`). Run explicitly with:

    pytest tests/test_artifact_bank_live.py -m live

`extract_camera_ccd`/`build_patch_bank` were written against SPOC TPF
primary-header documentation (`CAMERA`/`CCD` integer keywords), not a live
downloaded file -- this turns that documentation-only assumption into a
runnable check against a real target, the same discipline
`test_exoplanet_archive_live.py` established.
"""

from __future__ import annotations

import pytest

from astra import artifact_bank as ab
from astra import artifact_patches

pytestmark = pytest.mark.live

# A real, well-known bright target with confirmed TESS coverage (Pi Mensae,
# host of Pi Men c, observed in TESS's earliest sectors).
REAL_TARGET_RA_DEG = 84.2911
REAL_TARGET_DEC_DEG = -80.4689


def test_extract_camera_ccd_parses_real_values_from_a_real_downloaded_tpf(isolated_root):
    paths = artifact_patches.download_reference_tpfs(
        [(REAL_TARGET_RA_DEG, REAL_TARGET_DEC_DEG)], root=isolated_root.datasets)

    assert paths, (
        "download_reference_tpfs returned no real TPF for a known-covered target -- "
        "either MAST TESScut is down, or find_sectors/download_tpf has regressed; "
        "check both before assuming this is a flaky network failure")

    camera, ccd = ab.extract_camera_ccd(paths[0])
    assert camera is not None and 1 <= camera <= 4, (
        f"CAMERA header keyword did not parse to a real TESS camera number (1-4): {camera!r} "
        "-- the keyword name assumed in extract_camera_ccd may have drifted from the real "
        "SPOC TPF header")
    assert ccd is not None and 1 <= ccd <= 4, (
        f"CCD header keyword did not parse to a real TESS CCD number (1-4): {ccd!r} -- "
        "the keyword name assumed in extract_camera_ccd may have drifted from the real "
        "SPOC TPF header")

    # Confirmed live this session: Pi Mensae's real sector-1 QUALITY column
    # does carry real flagged cadences (10 "pointing", 5 "systematic"), but
    # every one is an isolated single-cadence run -- `min_run_length=1` is
    # used here (default is 3) specifically to exercise the metadata-
    # attachment path against this real file, not as the production default.
    records = ab.build_patch_bank(paths, min_run_length=1)
    assert records, (
        "build_patch_bank produced no patches even with min_run_length=1 -- "
        "either this target's real QUALITY column has changed, or the "
        "patch-extraction wiring has regressed")
    assert records[0].sector is not None and records[0].sector >= 1
    assert records[0].camera == camera and records[0].ccd == ccd
    assert records[0].night is not None and len(records[0].night) == 10
