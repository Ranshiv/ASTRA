"""Protected local credentials for provider APIs.

The TNS API key is deliberately never put in a project manifest, candidate
file, SQLite cache row, log record, or environment variable.  On the Windows
desktop release it is encrypted with the current user's DPAPI key before it
is written below ASTRA's configuration directory.  A copied credential file
therefore cannot be decrypted by a different Windows user or machine.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path

from . import config

_CREDENTIAL_FILE = "tns-credentials.dpapi.json"
_ENTROPY = b"ASTRA/TNS credential v1"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class CredentialError(RuntimeError):
    """A secret is unavailable, malformed, or cannot be protected."""


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def credential_path(paths: config.Paths | None = None) -> Path:
    paths = paths or config.PATHS
    paths.config.mkdir(parents=True, exist_ok=True)
    return paths.config / _CREDENTIAL_FILE


def _blob(data: bytes) -> tuple[_DataBlob, object]:
    # Keep the backing buffer alive for the lifetime of the Win32 call.
    buffer = (ctypes.c_ubyte * max(1, len(data)))()
    if data:
        ctypes.memmove(buffer, data, len(data))
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _crypt32():
    if os.name != "nt":
        raise CredentialError("protected TNS credentials require Windows DPAPI")
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


def _protect(plaintext: bytes) -> bytes:
    crypt32 = _crypt32()
    source, source_buffer = _blob(plaintext)
    entropy, entropy_buffer = _blob(_ENTROPY)
    output = _DataBlob()
    # Reference buffers explicitly so a future refactor cannot release them
    # before CryptProtectData has read them.
    _ = source_buffer, entropy_buffer
    if not crypt32.CryptProtectData(ctypes.byref(source), "ASTRA TNS API key",
                                    ctypes.byref(entropy), None, None,
                                    _CRYPTPROTECT_UI_FORBIDDEN,
                                    ctypes.byref(output)):
        _raise_last_error("encryption")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.WinDLL("kernel32", use_last_error=True).LocalFree(output.pbData)


def _unprotect(ciphertext: bytes) -> bytes:
    crypt32 = _crypt32()
    source, source_buffer = _blob(ciphertext)
    entropy, entropy_buffer = _blob(_ENTROPY)
    output = _DataBlob()
    _ = source_buffer, entropy_buffer
    if not crypt32.CryptUnprotectData(ctypes.byref(source), None,
                                      ctypes.byref(entropy), None, None,
                                      _CRYPTPROTECT_UI_FORBIDDEN,
                                      ctypes.byref(output)):
        _raise_last_error("decryption")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.WinDLL("kernel32", use_last_error=True).LocalFree(output.pbData)


def save_tns_credentials(api_key: str, bot_id: str = "", bot_name: str = "ASTRA",
                         paths: config.Paths | None = None) -> dict:
    """Encrypt and store a TNS API key for the current Windows user.

    The return value intentionally contains only non-secret configuration
    metadata, so it is safe for an RPC response or a UI status panel.
    """
    api_key = api_key.strip()
    if not api_key:
        raise CredentialError("TNS API key cannot be empty")
    payload = {
        "version": 1,
        "api_key": api_key,
        "bot_id": str(bot_id).strip(),
        "bot_name": str(bot_name).strip() or "ASTRA",
    }
    encrypted = _protect(json.dumps(payload, sort_keys=True).encode("utf-8"))
    path = credential_path(paths)
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
    return {"configured": True, "backend": "windows_dpapi",
            "bot_id": payload["bot_id"], "bot_name": payload["bot_name"]}


def load_tns_credentials(paths: config.Paths | None = None) -> dict | None:
    """Return decrypted TNS credentials only to the local provider client."""
    path = credential_path(paths)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("version") != 1 or record.get("backend") != "windows_dpapi":
            raise CredentialError("unsupported TNS credential format")
        plaintext = _unprotect(base64.b64decode(record["ciphertext_b64"], validate=True))
        payload = json.loads(plaintext.decode("utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CredentialError("stored TNS credentials are unreadable") from exc
    if not isinstance(payload.get("api_key"), str) or not payload["api_key"]:
        raise CredentialError("stored TNS credentials do not contain an API key")
    return {"api_key": payload["api_key"],
            "bot_id": str(payload.get("bot_id", "")),
            "bot_name": str(payload.get("bot_name", "ASTRA"))}


def tns_credential_status(paths: config.Paths | None = None) -> dict:
    """Report readiness without ever returning a secret."""
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
    path = credential_path(paths)
    if not path.exists():
        return False
    path.unlink()
    return True
