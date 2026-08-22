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
from .gaia import GaiaConnector
from .hubble import HubbleConnector
from .jwst import JWSTConnector
from .panstarrs import PanSTARRSConnector
from .sdss import SDSSConnector
from .swift import SwiftConnector
from .tess import TESSConnector
from .xmm import XMMConnector
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
    "hubble": HubbleConnector,
    "jwst": JWSTConnector,
    "alerce": ALeRCEConnector,
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
    and is registered above. A direct Rubin/LSST TAP connector (credential-
    required, `data.lsst.cloud/api/tap`) would land here too, once ASTRA has
    an actual data-rights token to validate it against -- see the ALeRCE
    entry in docs/DEFERRED.txt for why that is deliberately deferred rather
    than built speculatively.
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
    "DESConnector", "HubbleConnector", "JWSTConnector",
    "ALeRCEConnector",
    "available", "register", "get", "describe_all",
]
