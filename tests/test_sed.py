"""Bounded broadband SED diagnostics."""

from astra import sed


def test_characterize_uses_colors_and_marks_extinction_provenance():
    result = sed.characterize(
        {"gaia_bp": 15.0, "gaia_g": 14.5, "gaia_rp": 14.0, "g": 14.7},
        extinction={"gaia_bp": 0.1, "gaia_rp": 0.05},
    )
    assert result["quality"] == "usable"
    assert "gaia_bp_rp" in result["colors"]
    assert result["extinction_applied"]["gaia_bp"] == 0.1
    assert result["temperature_k"] is not None


def test_characterize_does_not_invent_temperature_from_one_band():
    result = sed.characterize({"g": 14.7})
    assert result["quality"] == "insufficient"
    assert result["temperature_k"] is None
    assert any("three bands" in warning for warning in result["warnings"])
