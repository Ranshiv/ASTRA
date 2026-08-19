"""Central path and cache configuration for the ASTRA science engine.

Every filesystem path used by the engine is resolved here. No other module
constructs a path itself. This exists because astroquery, astropy and
lightkurve each default to caching inside the user profile with no size
limit, which quietly consumes the system drive over a long research run.

Import this module before any astronomy library: `apply_cache_redirects()`
sets environment variables that astropy only reads at import time.
"""

from __future__ import annotations

import os
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Data layout follows section 11 of the project plan.
_DEFAULT_ROOT = Path.home() / "ASTRA"

# Caps are enforced by astra.cache, not by the libraries themselves.
DEFAULT_CACHE_CAP_GB = 15.0
DEFAULT_DATASET_CAP_GB = 45.0


@dataclass(frozen=True)
class Paths:
    """Resolved directory layout for one ASTRA installation."""

    root: Path
    projects: Path = field(init=False)
    datasets: Path = field(init=False)
    models: Path = field(init=False)
    cache: Path = field(init=False)
    logs: Path = field(init=False)
    config: Path = field(init=False)

    def __post_init__(self) -> None:
        for name in ("projects", "datasets", "models", "cache", "logs", "config"):
            object.__setattr__(self, name, self.root / name.capitalize())

    @property
    def astroquery_cache(self) -> Path:
        return self.cache / "astroquery"

    @property
    def lightkurve_cache(self) -> Path:
        return self.cache / "lightkurve"

    @property
    def astropy_cache(self) -> Path:
        return self.cache / "astropy"

    def survey_dir(self, survey: str) -> Path:
        """Canonical store for one survey's extracted Parquet data."""
        return self.datasets / survey.upper()

    def all_dirs(self) -> list[Path]:
        return [
            self.projects, self.datasets, self.models,
            self.cache, self.logs, self.config,
            self.astroquery_cache, self.lightkurve_cache, self.astropy_cache,
        ]

    def ensure(self) -> None:
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)


def resolve_root() -> Path:
    """ASTRA_ROOT wins, so the data directory can live on a larger drive."""
    override = os.environ.get("ASTRA_ROOT")
    return Path(override).expanduser() if override else _DEFAULT_ROOT


PATHS = Paths(resolve_root())


def cache_cap_gb() -> float:
    return float(os.environ.get("ASTRA_CACHE_CAP_GB", DEFAULT_CACHE_CAP_GB))


def dataset_cap_gb() -> float:
    return float(os.environ.get("ASTRA_DATASET_CAP_GB", DEFAULT_DATASET_CAP_GB))


def _cuda_toolkit_version_key(path: Path) -> tuple[int, ...]:
    """Sort key for a CUDA toolkit directory, by numeric version, not text.

    A plain string sort ranks "v9.0" above "v12.0" -- the character "9"
    sorts after "1" -- silently preferring an older toolkit whenever a
    single- and double-digit major version are installed side by side.
    Parsing the digits out and comparing them as a tuple of integers avoids
    that regardless of how many version components or digits each has.
    """
    numbers = tuple(int(part) for part in re.findall(r"\d+", path.name))
    return numbers or (-1,)


def ensure_cuda_path() -> str | None:
    """Point CUDA_PATH at an installed toolkit if the environment lacks it.

    CuPy JIT-compiles its kernels and therefore needs the CUDA headers at
    runtime, not just the driver. Without CUDA_PATH it fails with
    "Failed to find CUDA headers" on the first array operation — which looks
    like a broken GPU rather than a missing environment variable. Setting it
    here means the engine works on a machine where the toolkit is installed
    but never added to the environment.
    """
    # An existing value is only trusted if it actually points at headers.
    # This machine had CUDA_PATH set machine-wide to a v12.5 directory left
    # behind by an uninstall, so respecting it blindly would keep the engine
    # pointed at a path that does not exist.
    existing = os.environ.get("CUDA_PATH")
    if existing and (Path(existing) / "include").exists():
        return existing

    base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if not base.exists():
        return None

    # Highest version present, so a newer toolkit is preferred -- compared
    # numerically, see _cuda_toolkit_version_key.
    versions = sorted((p for p in base.iterdir()
                       if p.is_dir() and (p / "include").exists()),
                      key=_cuda_toolkit_version_key, reverse=True)
    if not versions:
        return None

    os.environ["CUDA_PATH"] = str(versions[0])
    return os.environ["CUDA_PATH"]


def apply_cache_redirects(paths: Paths | None = None) -> Paths:
    """Point every astronomy library's cache at ASTRA's managed cache directory.

    Must run before astropy/astroquery/lightkurve are imported. Returns the
    paths actually applied so callers can log them.
    """
    paths = paths or PATHS
    paths.ensure()
    ensure_cuda_path()

    # These libraries resolve cache locations at import time.  Do not use
    # setdefault: a parent process (including a packaged launcher) can leave a
    # stale path behind, which would put unbounded third-party cache writes
    # outside ASTRA's managed quota.  ASTRA_ROOT is the supported override.
    os.environ["XDG_CACHE_HOME"] = str(paths.cache)
    os.environ["ASTROPY_CACHE_DIR"] = str(paths.astropy_cache)
    os.environ["LIGHTKURVE_CACHE_DIR"] = str(paths.lightkurve_cache)
    return paths


def bind_library_caches(paths: Paths | None = None) -> dict[str, str]:
    """Set the in-process cache attributes the env vars do not cover.

    Both libraries have moved these settings between releases, so each bind is
    attempted independently and failures are reported rather than raised —
    a stale attribute name must not stop the engine from starting.
    """
    paths = paths or PATHS
    results: dict[str, str] = {}

    try:
        import lightkurve as lk

        try:
            lk.conf.cache_dir = str(paths.lightkurve_cache)
            results["lightkurve"] = str(paths.lightkurve_cache)
        except Exception as exc:  # noqa: BLE001 - packaged ConfigItem limitation
            # PyInstaller's resource-safe one-folder layout can make
            # Astropy's call-stack based ConfigItem setter unable to infer the
            # originating module.  The environment variable was applied
            # before importing Lightkurve and is the supported fallback for
            # this case; report it explicitly rather than claiming failure.
            if os.environ.get("LIGHTKURVE_CACHE_DIR") == str(paths.lightkurve_cache):
                results["lightkurve"] = f"environment:{paths.lightkurve_cache}"
            else:
                detail = (traceback.format_exc()
                          if os.environ.get("ASTRA_LIBRARY_BINDING_DEBUG") == "1"
                          else str(exc))
                results["lightkurve"] = f"unbound: {detail}"
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        detail = (traceback.format_exc() if os.environ.get("ASTRA_LIBRARY_BINDING_DEBUG") == "1"
                  else str(exc))
        results["lightkurve"] = f"unbound: {detail}"

    try:
        from astroquery.query import BaseQuery

        BaseQuery.cache_location = paths.astroquery_cache
        results["astroquery"] = str(paths.astroquery_cache)
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        detail = (traceback.format_exc() if os.environ.get("ASTRA_LIBRARY_BINDING_DEBUG") == "1"
                  else str(exc))
        results["astroquery"] = f"unbound: {detail}"

    return results
