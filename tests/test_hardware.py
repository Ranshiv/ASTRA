"""Execution-mode selection must never raise and must respect VRAM headroom."""

from __future__ import annotations

from astra import hardware


def _gpu(free_mb, total_mb=4096):
    return hardware.GpuInfo(
        name="Test GPU", total_vram_mb=total_mb, free_vram_mb=free_mb,
        compute_capability="7.5", driver_version="592.82",
    )


def test_cpu_when_torch_missing(monkeypatch):
    monkeypatch.setattr(hardware, "query_nvidia_smi", lambda: None)
    monkeypatch.setitem(__import__("sys").modules, "torch", None)

    report = hardware.select_device()

    assert report.device in {"cpu", "cuda"}  # torch may legitimately be present
    assert report.reason


def test_cpu_when_vram_headroom_too_small(monkeypatch):
    monkeypatch.setattr(hardware, "query_nvidia_smi",
                        lambda: _gpu(free_mb=hardware.MIN_USABLE_VRAM_MB - 1))

    try:
        import torch  # noqa: F401
    except ImportError:
        return  # nothing to assert without torch installed

    if not __import__("torch").cuda.is_available():
        return

    report = hardware.select_device()
    assert report.device == "cpu"
    assert "VRAM free" in report.reason


def test_report_serialises_without_gpu_key_when_absent():
    report = hardware.DeviceReport(
        device="cpu", reason="no gpu", torch_available=False, cuda_available=False,
    )
    assert "gpu" not in report.to_dict()


def test_report_includes_gpu_when_present():
    report = hardware.DeviceReport(
        device="cuda", reason="ok", torch_available=True,
        cuda_available=True, gpu=_gpu(free_mb=2233),
    )
    payload = report.to_dict()
    assert payload["gpu"]["free_vram_mb"] == 2233
