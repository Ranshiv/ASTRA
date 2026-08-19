"""Hardware detection and execution-mode selection (plan section 26).

The engine must run on machines with no usable GPU, so `select_device` never
raises: it degrades to CPU and records why. VRAM headroom is checked rather
than total VRAM, because on a display-attached GPU a large share is already
committed to the desktop compositor before training starts.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass

# Below this, batch sizes collapse to the point where GPU training is slower
# than CPU and far more likely to OOM mid-run.
MIN_USABLE_VRAM_MB = 1024


@dataclass
class GpuInfo:
    name: str
    total_vram_mb: int
    free_vram_mb: int
    compute_capability: str
    driver_version: str


@dataclass
class DeviceReport:
    device: str  # "cuda" or "cpu"
    reason: str
    torch_available: bool
    cuda_available: bool
    gpu: GpuInfo | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.gpu is None:
            payload.pop("gpu")
        return payload


def query_nvidia_smi() -> GpuInfo | None:
    """Read GPU state without importing torch, so the UI can show it instantly."""
    if shutil.which("nvidia-smi") is None:
        return None

    fields = "name,memory.total,memory.free,compute_cap,driver_version"
    try:
        raw = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None

    first = raw.splitlines()[0] if raw else ""
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 5:
        return None

    try:
        return GpuInfo(
            name=parts[0],
            total_vram_mb=int(float(parts[1])),
            free_vram_mb=int(float(parts[2])),
            compute_capability=parts[3],
            driver_version=parts[4],
        )
    except ValueError:
        return None


def select_device() -> DeviceReport:
    """Choose the execution mode, preferring CUDA only when it is actually usable."""
    gpu = query_nvidia_smi()

    try:
        import torch
    except ImportError:
        return DeviceReport(
            device="cpu",
            reason="PyTorch is not installed; running in CPU mode.",
            torch_available=False,
            cuda_available=False,
            gpu=gpu,
        )

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return DeviceReport(
            device="cpu",
            reason="PyTorch is present but its CUDA module is unavailable; running in CPU mode.",
            torch_available=False,
            cuda_available=False,
            gpu=gpu,
        )
    cuda_available = bool(cuda.is_available())
    if not cuda_available:
        return DeviceReport(
            device="cpu",
            reason="PyTorch reports no CUDA device; running in CPU mode.",
            torch_available=True,
            cuda_available=False,
            gpu=gpu,
        )

    if gpu is not None and gpu.free_vram_mb < MIN_USABLE_VRAM_MB:
        return DeviceReport(
            device="cpu",
            reason=(
                f"Only {gpu.free_vram_mb} MB VRAM free of {gpu.total_vram_mb} MB "
                f"(need {MIN_USABLE_VRAM_MB} MB); running in CPU mode."
            ),
            torch_available=True,
            cuda_available=True,
            gpu=gpu,
        )

    headroom = f"{gpu.free_vram_mb} MB free" if gpu else "unknown headroom"
    return DeviceReport(
        device="cuda",
        reason=f"CUDA device available ({headroom}).",
        torch_available=True,
        cuda_available=True,
        gpu=gpu,
    )
