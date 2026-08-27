"""Survey connector registry.

Plan section 7 requires that a new survey arrive as a connector rather than as
an edit to the pipeline. Registration is the whole extension point: anything
implementing `SurveyConnector` and registered here becomes available to the
acquisition stage, the UI and the manifest without further changes.
"""

from __future__ import annotations

from .base import (ConeQuery, LightCurve, SourceRef, SurveyConnector,
                   TimeSystem, ValueKind)
from .alerce import ALeRCEConnector
from .chandra import ChandraConnector
from .des import DESConnector
from .desi import DESIConnector
from .erosita import EROSITAConnector
from .gaia import GaiaConnector
from .galex import GALEXConnector
from .herschel import HerschelConnector
from .hubble import HubbleConnector
from .jwst import JWSTConnector
from .kepler import KeplerConnector
from .panstarrs import PanSTARRSConnector
from .rubin_tap import RubinTAPConnector
from .sdss import SDSSConnector
from .swift import SwiftConnector
from .tess import TESSConnector
from .twomass import TwoMASSConnector
from .wise import WISEConnector
from .xmm import XMMConnector
from .ogle import OGLEConnector
from .vlass import VLASSConnector
from .ztf import ZTFConnector

_REGISTRY: dict[str, type[SurveyConnector]] = {
    "ztf": ZTFConnector,
    "gaia": GaiaConnector,
    "tess": TESSConnector,
    "sdss": SDSSConnector,
    "panstarrs": PanSTARRSConnector,
    "chandra": ChandraConnector,
    "swift": SwiftConnector,
    "xmm": XMMConnector,
    "des": DESConnector,
    "desi": DESIConnector,
    "erosita": EROSITAConnector,
    "hubble": HubbleConnector,
    "jwst": JWSTConnector,
    "alerce": ALeRCEConnector,
    "rubin_tap": RubinTAPConnector,
    "ogle": OGLEConnector,
    "vlass": VLASSConnector,
    "galex": GALEXConnector,
    "twomass": TwoMASSConnector,
    "wise": WISEConnector,
    "herschel": HerschelConnector,
    # K2 shares this same connector; pass mission="K2" via
    # surveys.get("kepler", mission="K2") rather than a near-duplicate entry.
    "kepler": KeplerConnector,
}


def available(include_experimental: bool = False) -> list[str]:
    """Return registered survey keys, excluding opt-in connectors by default."""
    if include_experimental:
        return sorted(_REGISTRY)
    return sorted(name for name, connector in _REGISTRY.items()
                  if connector.enabled_by_default)


def register(name: str, connector: type[SurveyConnector]) -> None:
    """Add a survey.

    ALeRCE (`alerce.py`) already brokers real, credential-free LSST alerts
    and is registered above. `RubinTAPConnector` (`rubin_tap.py`, credential-
    required, `data.lsst.cloud/api/tap`) is also registered, but stays
    DORMANT until a real Rubin data-rights token exists: it is written and
    tested only against mocked TAP responses, and `enabled_by_default` is
    False -- see the ALeRCE entry in docs/DEFERRED.txt for the full history
    of why the direct-TAP path was deliberately deferred rather than built
    speculatively, and rubin_tap.py's own module docstring for what remains
    unvalidated.
    """
    if not issubclass(connector, SurveyConnector):
        raise TypeError(f"{connector!r} does not implement SurveyConnector")
    _REGISTRY[name.lower()] = connector


def get(name: str, **kwargs) -> SurveyConnector:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown survey {name!r}; available: {available()}")
    return _REGISTRY[key](**kwargs)


def describe_all(include_experimental: bool = True) -> list[dict]:
    """Connector metadata for the UI, without touching the network."""
    return [get(name).describe() for name in available(include_experimental)]


__all__ = [
    "ConeQuery", "LightCurve", "SourceRef", "SurveyConnector",
    "TimeSystem", "ValueKind",
    "GaiaConnector", "TESSConnector", "ZTFConnector",
    "SDSSConnector", "PanSTARRSConnector", "ChandraConnector",
    "SwiftConnector", "XMMConnector",
    "DESConnector", "DESIConnector", "HubbleConnector", "JWSTConnector",
    "ALeRCEConnector", "RubinTAPConnector", "VLASSConnector", "EROSITAConnector",
    "GALEXConnector", "TwoMASSConnector", "WISEConnector", "HerschelConnector",
    "KeplerConnector",
    "available", "register", "get", "describe_all",
]
