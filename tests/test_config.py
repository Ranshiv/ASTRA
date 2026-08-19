"""Path resolution and cache-redirect behaviour."""

from __future__ import annotations

from pathlib import Path

from astra import config


def test_root_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_ROOT", str(tmp_path / "custom"))
    assert config.resolve_root() == tmp_path / "custom"


def test_root_defaults_to_home_when_unset(monkeypatch):
    monkeypatch.delenv("ASTRA_ROOT", raising=False)
    assert config.resolve_root() == Path.home() / "ASTRA"


def test_layout_matches_plan_section_11(tmp_path):
    paths = config.Paths(tmp_path)
    assert paths.projects == tmp_path / "Projects"
    assert paths.datasets == tmp_path / "Datasets"
    assert paths.models == tmp_path / "Models"
    assert paths.cache == tmp_path / "Cache"
    assert paths.logs == tmp_path / "Logs"
    assert paths.config == tmp_path / "Config"


def test_ensure_creates_every_directory(tmp_path):
    paths = config.Paths(tmp_path)
    paths.ensure()
    for directory in paths.all_dirs():
        assert directory.is_dir()


def test_survey_dir_is_upper_cased(tmp_path):
    paths = config.Paths(tmp_path)
    assert paths.survey_dir("ztf") == tmp_path / "Datasets" / "ZTF"


def test_caps_read_from_env(monkeypatch):
    monkeypatch.setenv("ASTRA_CACHE_CAP_GB", "3.5")
    monkeypatch.setenv("ASTRA_DATASET_CAP_GB", "9")
    assert config.cache_cap_gb() == 3.5
    assert config.dataset_cap_gb() == 9.0


def test_redirects_point_libraries_into_managed_cache(monkeypatch, tmp_path):
    for var in ("XDG_CACHE_HOME", "ASTROPY_CACHE_DIR", "LIGHTKURVE_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)

    paths = config.apply_cache_redirects(config.Paths(tmp_path))

    import os
    assert Path(os.environ["XDG_CACHE_HOME"]) == paths.cache
    assert Path(os.environ["ASTROPY_CACHE_DIR"]) == paths.astropy_cache
    assert paths.astroquery_cache.is_dir()


class TestCudaToolkitVersionKey:
    """Toolkit directories must sort by numeric version, not as plain text.

    A plain string sort ranks "v9.0" above "v12.0" (the character "9" sorts
    after "1"), which would silently prefer an older toolkit whenever a
    single- and double-digit major version are installed side by side.
    """

    def test_double_digit_major_beats_single_digit(self):
        versions = sorted([Path("v9.0"), Path("v12.0")],
                          key=config._cuda_toolkit_version_key, reverse=True)
        assert versions == [Path("v12.0"), Path("v9.0")]

    def test_minor_versions_compare_numerically_not_lexicographically(self):
        """"v12.10" must beat "v12.5" -- as text, "5" > "1" would say
        otherwise."""
        versions = sorted([Path("v12.5"), Path("v12.10")],
                          key=config._cuda_toolkit_version_key, reverse=True)
        assert versions == [Path("v12.10"), Path("v12.5")]

    def test_a_realistic_mixed_set_sorts_correctly(self):
        names = ["v9.0", "v12.0", "v12.5", "v12.10", "v13.0"]
        versions = sorted((Path(n) for n in names),
                          key=config._cuda_toolkit_version_key, reverse=True)
        assert [v.name for v in versions] == \
            ["v13.0", "v12.10", "v12.5", "v12.0", "v9.0"]

    def test_a_name_with_no_digits_sorts_last(self):
        versions = sorted([Path("v12.0"), Path("unknown")],
                          key=config._cuda_toolkit_version_key, reverse=True)
        assert versions[0] == Path("v12.0")


class TestEnsureCudaPath:
    def test_an_existing_valid_cuda_path_is_trusted(self, tmp_path, monkeypatch):
        (tmp_path / "include").mkdir()
        monkeypatch.setenv("CUDA_PATH", str(tmp_path))

        assert config.ensure_cuda_path() == str(tmp_path)

    def test_an_existing_path_missing_headers_is_not_trusted(
            self, tmp_path, monkeypatch):
        """The real, motivating case this function's own docstring names:
        CUDA_PATH set machine-wide to a directory an uninstall left behind,
        which must not be returned just because the variable is set."""
        stale = tmp_path / "stale-v12.5"
        stale.mkdir()  # no "include" subdirectory
        monkeypatch.setenv("CUDA_PATH", str(stale))

        result = config.ensure_cuda_path()

        assert result != str(stale)
