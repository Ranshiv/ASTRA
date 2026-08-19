"""Offline readiness checks for externally gated ASTRA capabilities."""

from __future__ import annotations

import os
from datetime import date

from . import hardware, surveys

GAIA_DR4_EXPECTED = date(2026, 12, 2)
MULTIMODAL_MIN_FREE_VRAM_MB = 8192


def _configured(name: str) -> bool:
    value = os.environ.get(name, "").strip()
    return bool(value)


def status() -> dict:
    device = hardware.select_device().to_dict()
    free_vram = (device.get("gpu") or {}).get("free_vram_mb")
    return {
        "gaia_epoch": {
            "status": "awaiting_dr4_contract",
            "expected_release": GAIA_DR4_EXPECTED.isoformat(),
            "enabled": False,
            "reason": "DR3 is static context; live epoch ingestion stays disabled until the DR4 schema, access terms, and quota are verified.",
        },
        "multimodal": {
            "status": "ready_for_external_gpu" if isinstance(free_vram, (int, float)) and free_vram >= MULTIMODAL_MIN_FREE_VRAM_MB else "blocked_external_gpu",
            "free_vram_mb": free_vram,
            "required_min_free_vram_mb": MULTIMODAL_MIN_FREE_VRAM_MB,
            "enabled": False,
        },
        "release": {
            "status": "ready_for_credentials" if all(_configured(name) for name in (
                "ASTRA_SIGN_CERT", "ASTRA_TIMESTAMP_URL", "ASTRA_UPDATER_KEY")) else "blocked_missing_release_credentials",
            "authenticode_certificate": _configured("ASTRA_SIGN_CERT"),
            "timestamp_url": _configured("ASTRA_TIMESTAMP_URL"),
            "updater_key": _configured("ASTRA_UPDATER_KEY"),
            "publication_configured": _configured("ASTRA_RELEASE_CHANNEL"),
        },
        "connectors": surveys.describe_all(include_experimental=True),
    }
