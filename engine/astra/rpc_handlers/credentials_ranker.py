"""TNS/Rubin credential configuration and the supervised ranker (train/apply/
list saved models).

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

from typing import Any

from .common import Handler

from .. import credentials, ranker

def _handle_tns_credentials_configure(params: dict[str, Any]) -> dict[str, Any]:
    """Store the API key with Windows DPAPI; it is never echoed to the UI."""
    return credentials.save_tns_credentials(
        str(params["api_key"]), str(params.get("bot_id", "")),
        str(params.get("bot_name", "ASTRA")),
    )


def _handle_tns_credentials_clear(_params: dict[str, Any]) -> dict[str, Any]:
    return {"cleared": credentials.clear_tns_credentials()}


def _handle_rubin_credentials_configure(params: dict[str, Any]) -> dict[str, Any]:
    """Store a Rubin/LSST data-rights token with Windows DPAPI.

    RubinTAPConnector (surveys/rubin_tap.py) is dormant until this is called
    with a real token -- see that module's docstring. Never echoed back.
    """
    return credentials.save_credentials("rubin", {"token": str(params["token"])})


def _handle_rubin_credentials_status(_params: dict[str, Any]) -> dict[str, Any]:
    return credentials.credential_status("rubin")


def _handle_rubin_credentials_clear(_params: dict[str, Any]) -> dict[str, Any]:
    return {"cleared": credentials.clear_credentials("rubin")}


def _handle_ranker_train(params: dict[str, Any]) -> dict[str, Any]:
    return ranker.train(
        name=params.get("name", "default"),
        model_name=params.get("model_name", "calibrated-logistic"),
        seed=int(params.get("seed", 42)),
        bootstrap_samples=int(params.get("bootstrap_samples", ranker.BOOTSTRAP_SAMPLES)),
    )


def _handle_ranker_apply(params: dict[str, Any]) -> dict[str, Any]:
    return ranker.apply(name=params.get("name", "default"),
                        model_name=params.get("model_name", "calibrated-logistic"))


def _handle_ranker_list(_params: dict[str, Any]) -> list[dict]:
    return ranker.list_models()


HANDLERS: dict[str, Handler] = {
    "credentials.tns.configure": _handle_tns_credentials_configure,
    "credentials.tns.clear": _handle_tns_credentials_clear,
    "credentials.rubin.configure": _handle_rubin_credentials_configure,
    "credentials.rubin.status": _handle_rubin_credentials_status,
    "credentials.rubin.clear": _handle_rubin_credentials_clear,
    "ranker.train": _handle_ranker_train,
    "ranker.apply": _handle_ranker_apply,
    "ranker.list": _handle_ranker_list,
}
