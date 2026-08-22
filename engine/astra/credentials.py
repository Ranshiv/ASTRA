"""Protected local credentials for provider APIs.

A secret such as the TNS API key is deliberately never put in a project
manifest, candidate file, SQLite cache row, log record, or environment
variable.  On the Windows desktop release it is encrypted with the current
user's DPAPI key before it is written below ASTRA's configuration directory.
A copied credential file therefore cannot be decrypted by a different
Windows user or machine.

`save_credentials`/`load_credentials`/`credential_status`/`clear_credentials`
are the generic form, keyed by a provider `name` (e.g. `"tns"`, `"rubin"`) so
a second credentialed provider does not need a third bespoke DPAPI wrapper.
`save_tns_credentials` and friends are thin wrappers over the generic form,
kept byte-identical to their original behaviour (same file name, same DPAPI
entropy and description string) so an already-stored TNS credential file
keeps decrypting correctly.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import re
from ctypes import wintypes
from pathlib import Path

from . import config

_CREDENTIAL_FILE = "tns-credentials.dpapi.json"
_ENTROPY = b"ASTRA/TNS credential v1"
_DESCRIPTION = "ASTRA TNS API key"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


class CredentialError(RuntimeError):
    """A secret is unavailable, malformed, or cannot be protected."""


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def credential_path(paths: config.Paths | None = None) -> Path:
    """TNS's credential file path. Kept for backward compatibility."""
    return _credential_path_for(_CREDENTIAL_FILE, paths)


def _validate_name(name: str) -> str:
    if not _NAME_PATTERN.match(name):
        raise CredentialError(
            f"invalid credential name: {name!r}; expected lowercase "
            "letters/digits/underscore, starting with a letter")
    return name


def _credential_filename(name: str) -> str:
    return f"{_validate_name(name)}-credentials.dpapi.json"


def _credential_path_for(filename: str, paths: config.Paths | None = None) -> Path:
    paths = paths or config.PATHS
    paths.config.mkdir(parents=True, exist_ok=True)
    return paths.config / filename


def _default_entropy(name: str) -> bytes:
    return f"ASTRA/{name} credential v1".encode("ascii")


def _default_description(name: str) -> str:
    return f"ASTRA {name} credential"


def _blob(data: bytes) -> tuple[_DataBlob, object]:
    # Keep the backing buffer alive for the lifetime of the Win32 call.
    buffer = (ctypes.c_ubyte * max(1, len(data)))()
    if data:
        ctypes.memmove(buffer, data, len(data))
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _crypt32():
    if os.name != "nt":
        raise CredentialError("protected credentials require Windows DPAPI")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob), wintypes.LPCWSTR, ctypes.POINTER(_DataBlob),
        wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob), ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob), wintypes.LPVOID, wintypes.LPVOID,
        wintypes.DWORD, ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    return crypt32


def _raise_last_error(operation: str) -> None:
    code = ctypes.get_last_error()
    raise CredentialError(f"Windows DPAPI {operation} failed (Win32 error {code})")


def _protect(plaintext: bytes, entropy: bytes, description: str) -> bytes:
    crypt32 = _crypt32()
    source, source_buffer = _blob(plaintext)
    entropy_blob, entropy_buffer = _blob(entropy)
    output = _DataBlob()
    # Reference buffers explicitly so a future refactor cannot release them
    # before CryptProtectData has read them.
    _ = source_buffer, entropy_buffer
    if not crypt32.CryptProtectData(ctypes.byref(source), description,
                                    ctypes.byref(entropy_blob), None, None,
                                    _CRYPTPROTECT_UI_FORBIDDEN,
                                    ctypes.byref(output)):
        _raise_last_error("encryption")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.WinDLL("kernel32", use_last_error=True).LocalFree(output.pbData)


def _unprotect(ciphertext: bytes, entropy: bytes) -> bytes:
    crypt32 = _crypt32()
    source, source_buffer = _blob(ciphertext)
    entropy_blob, entropy_buffer = _blob(entropy)
    output = _DataBlob()
    _ = source_buffer, entropy_buffer
    if not crypt32.CryptUnprotectData(ctypes.byref(source), None,
                                      ctypes.byref(entropy_blob), None, None,
                                      _CRYPTPROTECT_UI_FORBIDDEN,
                                      ctypes.byref(output)):
        _raise_last_error("decryption")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.WinDLL("kernel32", use_last_error=True).LocalFree(output.pbData)


def save_credentials(name: str, payload: dict, *,
                     entropy: bytes | None = None, description: str | None = None,
                     paths: config.Paths | None = None) -> dict:
    """Encrypt and store an arbitrary secret payload for the current Windows user.

    `payload` must be a non-empty JSON-serialisable dict. The return value
    intentionally contains only non-secret configuration metadata, so it is
    safe for an RPC response or a UI status panel -- callers that want
    additional non-secret fields echoed back should merge them into the
    returned dict themselves.
    """
    _validate_name(name)
    if not isinstance(payload, dict) or not payload:
        raise CredentialError(f"{name} credential payload cannot be empty")
    entropy = entropy if entropy is not None else _default_entropy(name)
    description = description or _default_description(name)
    encrypted = _protect(json.dumps(payload, sort_keys=True).encode("utf-8"),
                         entropy, description)
    path = _credential_path_for(_credential_filename(name), paths)
    record = {
        "version": 1,
        "backend": "windows_dpapi",
        "ciphertext_b64": base64.b64encode(encrypted).decode("ascii"),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass  # DPAPI is the effective access control on Windows.
    temporary.replace(path)
    return {"configured": True, "backend": "windows_dpapi"}


def load_credentials(name: str, *, entropy: bytes | None = None,
                     paths: config.Paths | None = None) -> dict | None:
    """Return a decrypted credential payload only to the local provider client."""
    _validate_name(name)
    path = _credential_path_for(_credential_filename(name), paths)
    if not path.exists():
        return None
    entropy = entropy if entropy is not None else _default_entropy(name)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("version") != 1 or record.get("backend") != "windows_dpapi":
            raise CredentialError(f"unsupported {name} credential format")
        plaintext = _unprotect(base64.b64decode(record["ciphertext_b64"], validate=True), entropy)
        payload = json.loads(plaintext.decode("utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CredentialError(f"stored {name} credentials are unreadable") from exc
    if not isinstance(payload, dict) or not payload:
        raise CredentialError(f"stored {name} credentials are malformed")
    return payload


def credential_status(name: str, *, entropy: bytes | None = None,
                      paths: config.Paths | None = None) -> dict:
    """Report readiness without ever returning a secret."""
    _validate_name(name)
    path = _credential_path_for(_credential_filename(name), paths)
    if not path.exists():
        return {"configured": False, "backend": "windows_dpapi"}
    try:
        load_credentials(name, entropy=entropy, paths=paths)
    except CredentialError as exc:
        return {"configured": True, "usable": False,
                "backend": "windows_dpapi", "reason": str(exc)}
    return {"configured": True, "usable": True, "backend": "windows_dpapi"}


def clear_credentials(name: str, paths: config.Paths | None = None) -> bool:
    """Remove only ASTRA's encrypted record for `name`; the action is idempotent."""
    _validate_name(name)
    path = _credential_path_for(_credential_filename(name), paths)
    if not path.exists():
        return False
    path.unlink()
    return True


def save_tns_credentials(api_key: str, bot_id: str = "", bot_name: str = "ASTRA",
                         paths: config.Paths | None = None) -> dict:
    """Encrypt and store a TNS API key for the current Windows user."""
    api_key = api_key.strip()
    if not api_key:
        raise CredentialError("TNS API key cannot be empty")
    payload = {
        "version": 1,
        "api_key": api_key,
        "bot_id": str(bot_id).strip(),
        "bot_name": str(bot_name).strip() or "ASTRA",
    }
    save_credentials("tns", payload, entropy=_ENTROPY, description=_DESCRIPTION, paths=paths)
    return {"configured": True, "backend": "windows_dpapi",
            "bot_id": payload["bot_id"], "bot_name": payload["bot_name"]}


def load_tns_credentials(paths: config.Paths | None = None) -> dict | None:
    """Return decrypted TNS credentials only to the local provider client."""
    payload = load_credentials("tns", entropy=_ENTROPY, paths=paths)
    if payload is None:
        return None
    if not isinstance(payload.get("api_key"), str) or not payload["api_key"]:
        raise CredentialError("stored TNS credentials do not contain an API key")
    return {"api_key": payload["api_key"],
            "bot_id": str(payload.get("bot_id", "")),
            "bot_name": str(payload.get("bot_name", "ASTRA"))}


def tns_credential_status(paths: config.Paths | None = None) -> dict:
    """Report TNS readiness without ever returning a secret."""
    path = credential_path(paths)
    if not path.exists():
        return {"configured": False, "backend": "windows_dpapi"}
    try:
        payload = load_tns_credentials(paths)
    except CredentialError as exc:
        return {"configured": True, "usable": False,
                "backend": "windows_dpapi", "reason": str(exc)}
    return {"configured": True, "usable": True, "backend": "windows_dpapi",
            "bot_id": payload["bot_id"], "bot_name": payload["bot_name"]}


def clear_tns_credentials(paths: config.Paths | None = None) -> bool:
    """Remove only ASTRA's encrypted TNS record; the action is idempotent."""
    return clear_credentials("tns", paths)
