"""Light-curve retrieval/folding/binning, FITS metadata/pixel access, image
and spectral feature extraction, and sidecar-file bookkeeping.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

from typing import Any

from .common import Handler, _workspace_root

from .. import (config, fitsio, image_features, modalitymatrix, security,
                spectral_features, viz)
from .. import project as project_mod

def _handle_curves_list(params: dict[str, Any]) -> list[dict]:
    return viz.list_curves(survey=params.get("survey"),
                           limit=int(params.get("limit", 500)),
                           root=config.PATHS.datasets)


def _handle_curves_get(params: dict[str, Any]) -> dict[str, Any]:
    return viz.curve_payload(
        security.authorized_path(params["path"]),
        max_points=int(params.get("max_points", viz.DEFAULT_MAX_POINTS)),
        frame=params.get("frame"),
    )


def _handle_curves_fold(params: dict[str, Any]) -> dict[str, Any]:
    return viz.fold(
        security.authorized_path(params["path"]),
        period_days=float(params["period_days"]),
        epoch=float(params["epoch"]) if params.get("epoch") is not None else None,
    )


def _handle_curves_bin(params: dict[str, Any]) -> dict[str, Any]:
    return viz.bin_curve(security.authorized_path(params["path"]),
                         bin_days=float(params["bin_days"]))


def _handle_fits_describe(params: dict[str, Any]) -> dict[str, Any]:
    return fitsio.describe(security.authorized_path(params["path"]))


def _handle_fits_header(params: dict[str, Any]) -> dict[str, Any]:
    return fitsio.read_header(security.authorized_path(params["path"]),
                              hdu=int(params.get("hdu", 0)))


def _handle_fits_image(params: dict[str, Any]) -> dict[str, Any]:
    return fitsio.image_payload(
        security.authorized_path(params["path"]),
        hdu=int(params["hdu"]) if params.get("hdu") is not None else None,
        contrast=float(params.get("contrast", 0.25)),
    )


def _handle_image_features(params: dict[str, Any]) -> dict[str, Any]:
    payload = image_features.extract(
        security.authorized_path(params["path"]),
        hdu=int(params["hdu"]) if params.get("hdu") is not None else None,
        target_xy=(float(params["target_x"]), float(params["target_y"]))
        if params.get("target_x") is not None and params.get("target_y") is not None
        else None,
    )
    if any(params.get(key) is not None for key in ("survey", "release", "object_id", "band")):
        payload["identity"] = {key: params.get(key, "unknown")
                                for key in ("survey", "release", "object_id", "band")}
    project_id = params.get("project_id")
    if project_id:
        output = image_features.save(payload,
                                     project_mod.project_dir(str(project_id)) / "results" / "image_features")
        payload["output_path"] = str(output)
    return payload


def _handle_spectral_features(params: dict[str, Any]) -> dict[str, Any]:
    path = security.authorized_path(params["path"])
    payload = spectral_features.from_fits(path)
    if any(params.get(key) is not None for key in ("survey", "release", "object_id", "band")):
        payload["identity"] = {key: params.get(key, "unknown")
                                for key in ("survey", "release", "object_id", "band")}
    project_id = params.get("project_id")
    if project_id:
        output = spectral_features.save(payload,
                                        project_mod.project_dir(str(project_id)) / "results" / "spectral_features")
        payload["output_path"] = str(output)
    return payload


def _handle_sidecars_list(params: dict[str, Any]) -> list[dict]:
    return modalitymatrix.list_sidecars(_workspace_root(params.get("project_id")))


def _handle_sidecar_save(params: dict[str, Any]) -> dict[str, Any]:
    kind = str(params["kind"])
    payloads = params.get("payloads") or []
    if not isinstance(payloads, list):
        raise ValueError("payloads must be a list")
    result = modalitymatrix.save_payloads(
        payloads, kind, name=str(params.get("name", "default")),
        root=_workspace_root(params.get("project_id")),
        identities=params.get("identities"),
    )
    return result.to_dict()


def _handle_sidecar_join(params: dict[str, Any]) -> dict[str, Any]:
    path = security.authorized_path(params["path"])
    identities = params.get("identities") or []
    if not isinstance(identities, list):
        raise ValueError("identities must be a list")
    rows, report = modalitymatrix.join_rows(
        identities, modalitymatrix.load(path), kind=str(params["kind"]),
    )
    return {"rows": rows, "report": report}


HANDLERS: dict[str, Handler] = {
    "curves.list": _handle_curves_list,
    "curves.get": _handle_curves_get,
    "curves.fold": _handle_curves_fold,
    "curves.bin": _handle_curves_bin,
    "fits.describe": _handle_fits_describe,
    "fits.header": _handle_fits_header,
    "fits.image": _handle_fits_image,
    "image.features": _handle_image_features,
    "spectrum.features": _handle_spectral_features,
    "sidecars.list": _handle_sidecars_list,
    "sidecars.save": _handle_sidecar_save,
    "sidecars.join": _handle_sidecar_join,
}
