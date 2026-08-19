"""Training loop for the deep models (plan phase 5, sections 19 and 26).

Everything here is shaped by running on a 4 GB display-attached GPU with about
2.2 GB genuinely free:

* Batch size is chosen from measured free VRAM, not assumed.
* Mixed precision is float16 with a GradScaler. This card is Turing (sm_75),
  which has no bfloat16 and no tensor cores, so AMP buys memory rather than
  much speed — worth taking for the memory alone.
* Gradient accumulation keeps the effective batch size independent of what
  fits, so results stay comparable between this machine and a rented GPU.
* Every run checkpoints each epoch, so a run can migrate machines part-way.
* Falling back to CPU is a normal outcome, not an error.

Runs record their seed, device, and configuration so plan section 37's
reproducibility requirement is satisfied for the deep models too.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config as config_mod, hardware, models
from .models import ModelConfig

# Reserve for CUDA context, fragmentation and the desktop growing mid-run.
VRAM_RESERVE_MB = 400
# Bytes per sequence element under float32 activations, measured empirically
# for these architectures with a safety factor for the backward pass.
BYTES_PER_ELEMENT = 512


@contextmanager
def _cuda_oom_guard():
    """Empty the allocator and add VRAM context when CUDA runs out of memory.

    `choose_batch_size()` sizes batches from measured free VRAM, but that is
    an estimate, not a guarantee -- another process (or this engine's own
    webview) can consume VRAM between the check and the actual allocation.
    On this project's target hardware (a 4 GB card with about 2.2 GB
    genuinely free) that is a real, expected failure mode, not an edge case.

    An uncaught OOM here is not a process crash: `rpc.dispatch()` and
    `jobs.py` both already catch any exception a handler raises and report it
    as a failed job. What they do NOT do is call `torch.cuda.empty_cache()`
    afterwards -- and this engine is a long-lived process serving the whole
    session, so a caught-but-unfreed OOM can leave the allocator fragmented
    for every training/inference call for the rest of the session, which
    looks like "the app keeps failing" even though nothing actually crashed.
    """
    import torch

    try:
        yield
    except RuntimeError as exc:
        message = str(exc)
        out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
        is_oom = isinstance(exc, out_of_memory_type) or "out of memory" in message.lower()
        if not is_oom or not torch.cuda.is_available():
            raise

        torch.cuda.empty_cache()
        try:
            free, total = torch.cuda.mem_get_info()
            vram = f"{free / 1024 ** 2:.0f} MB free of {total / 1024 ** 2:.0f} MB"
        except Exception:  # noqa: BLE001 - the diagnostic itself must not fail
            vram = "VRAM state unavailable"

        raise RuntimeError(
            f"CUDA ran out of memory ({vram}). The allocator has been "
            f"cleared, so a retry with a smaller batch size or a smaller "
            f"model is likely to succeed now. Original error: {message}"
        ) from exc


@dataclass
class TrainConfig:
    kind: str = "autoencoder"
    epochs: int = 30
    learning_rate: float = 1e-3
    batch_size: int | None = None        # None means choose from free VRAM
    effective_batch_size: int = 256      # held constant via accumulation
    kl_weight: float = 1.0
    seed: int = 42
    patience: int = 6
    amp: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["model"] = self.model.to_dict()
        return payload


@dataclass
class TrainReport:
    kind: str
    device: str
    device_reason: str
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    batch_size: int = 0
    accumulation_steps: int = 1
    parameters: int = 0
    amp_enabled: bool = False
    seconds: float = 0.0
    checkpoint: str | None = None
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def choose_batch_size(length: int, requested: int | None = None) -> int:
    """Pick a batch size that fits in the VRAM actually free right now.

    Free VRAM is read rather than total, because on a display-attached card a
    large share is already committed to the compositor before training starts.
    """
    if requested:
        return requested

    report = hardware.select_device()
    if report.device != "cuda" or report.gpu is None:
        return 64  # CPU: bounded by patience, not memory

    usable_mb = max(report.gpu.free_vram_mb - VRAM_RESERVE_MB, 128)
    per_row_bytes = length * BYTES_PER_ELEMENT
    fits = int((usable_mb * 1024 * 1024) / max(per_row_bytes, 1))

    # Clamped to powers-of-two-ish bounds that behave well with cuDNN.
    return int(max(8, min(256, 2 ** int(np.floor(np.log2(max(fits, 8)))))))


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(train_values: np.ndarray, val_values: np.ndarray,
          cfg: TrainConfig | None = None,
          checkpoint_dir: Path | None = None,
          name: str = "model") -> TrainReport:
    """Train one model and return a report with the full loss history."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg = cfg or TrainConfig()
    _set_seed(cfg.seed)

    device_report = hardware.select_device()
    device = torch.device(device_report.device)

    batch_size = choose_batch_size(cfg.model.length, cfg.batch_size)
    accumulation = max(1, cfg.effective_batch_size // max(batch_size, 1))

    model = models.make(cfg.kind, cfg.model).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # AMP only helps on CUDA; on CPU it is a slowdown with no memory benefit.
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_values)),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_tensor = torch.from_numpy(val_values).to(device)

    report = TrainReport(
        kind=cfg.kind, device=device_report.device,
        device_reason=device_report.reason, epochs_run=0, best_epoch=-1,
        best_val_loss=float("inf"), batch_size=batch_size,
        accumulation_steps=accumulation,
        parameters=models.count_parameters(model),
        amp_enabled=use_amp, seed=cfg.seed,
    )

    checkpoint_dir = checkpoint_dir or config_mod.PATHS.models
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_{cfg.kind}.pt"

    started = time.time()
    since_improvement = 0

    for epoch in range(cfg.epochs):
        model.train()
        running, batches = 0.0, 0
        optimiser.zero_grad(set_to_none=True)

        with _cuda_oom_guard():
            for step, (batch,) in enumerate(train_loader):
                batch = batch.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = _loss_for(model, batch, cfg)
                # Scaled so accumulated gradients average rather than sum.
                scaler.scale(loss / accumulation).backward()

                if (step + 1) % accumulation == 0:
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad(set_to_none=True)

                running += float(loss.detach())
                batches += 1

            # Flush any partial accumulation window so the last batches count.
            if batches % accumulation != 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

        train_loss = running / max(batches, 1)
        val_loss = evaluate_loss(model, val_tensor, cfg)

        report.train_losses.append(train_loss)
        report.val_losses.append(val_loss)
        report.epochs_run = epoch + 1

        if val_loss < report.best_val_loss - 1e-6:
            report.best_val_loss = val_loss
            report.best_epoch = epoch
            since_improvement = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg.to_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path)
            report.checkpoint = str(checkpoint_path)
        else:
            since_improvement += 1
            if since_improvement >= cfg.patience:
                break

    report.seconds = round(time.time() - started, 2)
    return report


def _loss_for(model, batch, cfg: TrainConfig):
    """Reconstruction loss, plus the KL term when the model is a VAE."""
    values, mask = batch[:, 0, :], batch[:, 1, :]
    prediction, mu, logvar = model(batch)
    loss = models.masked_reconstruction_loss(prediction, values, mask)
    if mu is not None and logvar is not None:
        loss = loss + cfg.kl_weight * models.kl_divergence(mu, logvar)
    return loss


def evaluate_loss(model, values, cfg: TrainConfig) -> float:
    import torch

    if values.shape[0] == 0:
        return float("nan")

    model.eval()
    with torch.no_grad():
        return float(_loss_for(model, values, cfg).detach())


def reconstruction_scores(model, values: np.ndarray,
                          batch_size: int = 64) -> np.ndarray:
    """Per-row masked reconstruction error — the deep anomaly score.

    Evaluated in float32 regardless of training precision: these numbers are
    ranked against each other, and float16 rounding would create ties.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    scores: list[np.ndarray] = []

    with torch.no_grad(), _cuda_oom_guard():
        for start in range(0, len(values), batch_size):
            chunk = torch.from_numpy(values[start:start + batch_size]).to(device)
            prediction, _, _ = model(chunk)
            per_row = models.masked_reconstruction_loss(
                prediction, chunk[:, 0, :], chunk[:, 1, :], reduction="none")
            scores.append(per_row.float().cpu().numpy())

    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def load_model(checkpoint_path: str | Path):
    """Restore a trained model together with the config it was trained under."""
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = payload["config"]
    model_config = ModelConfig(**{
        **saved["model"],
        "channels": tuple(saved["model"]["channels"]),
    })
    model = models.make(saved["kind"], model_config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, saved


def save_report(report: TrainReport, name: str,
                root: Path | None = None) -> Path:
    root = root or config_mod.PATHS.models
    path = root / f"{name}_{report.kind}_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
