"""Shared fixtures. Tests never touch the network or the real data root."""

from __future__ import annotations

import numpy as np
import pytest

from astra.surveys.base import ConeQuery, LightCurve, SourceRef


def pytest_configure(config: pytest.Config) -> None:
    # Registered here (not only in engine/pyproject.toml) because this
    # project's test suite is normally invoked as `pytest tests` from the
    # repo root, where pytest does not discover engine/pyproject.toml's
    # `[tool.pytest.ini_options]` -- see test_frb_live.py/
    # test_alerts_gcn_live.py's module docstrings for why a `live` marker
    # exists at all.
    config.addinivalue_line(
        "markers", "live: hits a real external service; skipped by default, run explicitly with -m live")


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Skip `live`-marked tests unless the caller explicitly selected them.

    `pytest tests -m live` sets `config.option.markexpr`, which this checks
    for and, when present, leaves every item's markers untouched so pytest's
    own `-m` filtering applies normally. Otherwise every `live`-marked item
    gets an explicit skip, so a default `pytest tests` run never touches a
    real external service.
    """
    if config.option.markexpr:
        return
    skip_live = pytest.mark.skip(reason="live test skipped by default; run with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def cone() -> ConeQuery:
    return ConeQuery(ra_deg=180.122, dec_deg=22.411, radius_arcsec=10.0)


@pytest.fixture
def source() -> SourceRef:
    return SourceRef(survey="ZTF", object_id="123456789",
                     ra_deg=180.122, dec_deg=22.411)


@pytest.fixture
def curve(source: SourceRef) -> LightCurve:
    # Realistic BJD magnitudes: the precision tests depend on values of this
    # magnitude, where float32 would visibly lose resolution.
    rng = np.random.default_rng(42)
    time = 2458000.123456 + np.arange(200, dtype=np.float64) * 0.5
    value = 18.0 + rng.normal(0.0, 0.05, size=200)
    err = np.full(200, 0.03)
    return LightCurve(source=source, release="dr24", band="g",
                      value_kind="mag", time=time, value=value,
                      value_err=err, time_system="HJD_UTC")


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    """Point every path at a temp directory for the duration of one test."""
    from astra import config

    paths = config.Paths(tmp_path)
    paths.ensure()
    monkeypatch.setattr(config, "PATHS", paths)
    return paths
