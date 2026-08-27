"""Training loop and sampling for `diffusion.py`'s denoiser (backlog item 14).

Split out from `diffusion.py` purely to keep each file under this project's
500-line guideline, the same reason `pretrain.py`/`pretrain_probe.py` were
split earlier this session -- architecture (`diffusion.py`: `DiffusionConfig`,
`make_denoiser`, the schedule, `diffusion_loss`) and training/sampling are
genuinely separable concerns, and this module only ever consumes
`diffusion.py`'s public API, never its internals.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .diffusion import (
    DiffusionConfig, DiffusionReport, count_parameters, diffusion_constants,
    diffusion_loss, forward_diffusion_sample, make_denoiser,
)


def _step_loss(model, batch, constants_t, rng, cfg, device,
               artifact_labels=None, transient_labels=None):
    """One step's masked noise-prediction loss, for reuse in both the
    training loop (with grad) and validation (under no_grad).

    `artifact_labels`/`transient_labels`, when given, are `(batch,)`
    integer arrays already sliced to this batch's rows -- `None` for a
    disabled channel, or per-row `-1` for a row with an enabled channel but
    no known label (mapped to the model's own "unspecified" index).
    """
    import torch

    value = batch[:, 0, :]
    mask = batch[:, 1, :]
    n = value.shape[0]

    t = rng.integers(0, cfg.timesteps, size=n)
    noise = rng.normal(0.0, 1.0, size=value.shape).astype(np.float32)
    x_t_value = forward_diffusion_sample(value, t, noise, constants_t).astype(np.float32)

    model_input = torch.from_numpy(np.stack([x_t_value, mask], axis=1)).to(device)
    t_tensor = torch.from_numpy(t).to(device)

    def _label_tensor(labels, n_classes):
        if labels is None:
            return None
        resolved = np.where(labels < 0, n_classes, labels)
        return torch.from_numpy(resolved.astype(np.int64)).to(device)

    artifact_t = _label_tensor(artifact_labels, cfg.n_artifact_classes)
    transient_t = _label_tensor(transient_labels, cfg.n_transient_classes)
    predicted_noise = model(model_input, t_tensor, artifact_t, transient_t)

    noise_t = torch.from_numpy(noise).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    return diffusion_loss(predicted_noise, noise_t, mask_t)


def train_diffusion(train_patches: np.ndarray, val_patches: np.ndarray,
                    cfg: DiffusionConfig | None = None,
                    checkpoint_dir: Path | None = None,
                    name: str = "diffusion",
                    train_artifact_labels: np.ndarray | None = None,
                    val_artifact_labels: np.ndarray | None = None,
                    train_transient_labels: np.ndarray | None = None,
                    val_transient_labels: np.ndarray | None = None) -> DiffusionReport:
    """Train the denoiser on real patches (2, patch_length): value + gap mask.

    Deliberately its own loop, not `train.train()`: the noise-prediction
    objective needs a per-step random timestep and a freshly sampled noise
    tensor, which `train._loss_for`'s `(prediction, mu, logvar)` contract
    has no slot for. Reuses the genuinely kind-agnostic pieces `pretrain.py`/
    `multimodal_moco.py` already established as reusable: VRAM-aware batch
    sizing, the OOM guard, seeding, and the same AMP/accumulation/
    early-stopping/checkpoint-on-improvement shape `train.train()` itself
    uses.

    `*_labels` (optional) are `(n,)` integer arrays aligned with
    `train_patches`/`val_patches`; a `-1` entry means "this row is real but
    its label is unknown," mapped to the model's own "unspecified" index at
    each step, distinct from disabling the channel entirely
    (`cfg.n_*_classes == 0`).
    """
    import torch
    from . import config as config_mod, hardware
    from . import train as train_mod

    cfg = cfg or DiffusionConfig()
    train_mod._set_seed(cfg.seed)
    data_rng = np.random.default_rng(cfg.seed)
    noise_rng = np.random.default_rng(cfg.seed + 1)

    device_report = hardware.select_device()
    device = torch.device(device_report.device)

    batch_size = train_mod.choose_batch_size(cfg.patch_length, cfg.batch_size)
    accumulation = max(1, cfg.effective_batch_size // max(batch_size, 1))

    model = make_denoiser(cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    constants = diffusion_constants(cfg.timesteps)
    constants_t = {k: v for k, v in constants.items()}

    report = DiffusionReport(
        device=device_report.device, device_reason=device_report.reason,
        epochs_run=0, best_epoch=-1, best_val_loss=float("inf"),
        batch_size=batch_size, accumulation_steps=accumulation,
        parameters=count_parameters(model), amp_enabled=use_amp, seed=cfg.seed,
    )

    checkpoint_dir = checkpoint_dir or config_mod.PATHS.models
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_diffusion.pt"

    started = time.time()
    since_improvement = 0
    n = len(train_patches)

    for epoch in range(cfg.epochs):
        model.train()
        order = data_rng.permutation(n) if n else np.empty(0, dtype=int)
        running, steps = 0.0, 0

        with train_mod._cuda_oom_guard():
            optimiser.zero_grad(set_to_none=True)
            for step, start in enumerate(range(0, n, max(batch_size, 1))):
                idx = order[start:start + batch_size]
                batch = train_patches[idx]
                batch_artifact = (train_artifact_labels[idx]
                                  if train_artifact_labels is not None else None)
                batch_transient = (train_transient_labels[idx]
                                   if train_transient_labels is not None else None)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = _step_loss(model, batch, constants_t, noise_rng, cfg, device,
                                      batch_artifact, batch_transient)
                scaler.scale(loss / accumulation).backward()

                if (step + 1) % accumulation == 0:
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad(set_to_none=True)

                running += float(loss.detach())
                steps += 1

            if steps and steps % accumulation != 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

        report.train_losses.append(running / max(steps, 1))

        model.eval()
        with torch.no_grad():
            val_loss = (float(_step_loss(model, val_patches, constants_t, noise_rng,
                                         cfg, device, val_artifact_labels,
                                         val_transient_labels).detach())
                       if len(val_patches) else float("nan"))
        report.val_losses.append(val_loss)
        report.epochs_run = epoch + 1

        if val_loss < report.best_val_loss - 1e-6:
            report.best_val_loss = val_loss
            report.best_epoch = epoch
            since_improvement = 0
            torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                       "epoch": epoch, "val_loss": val_loss}, checkpoint_path)
            report.checkpoint = str(checkpoint_path)
        else:
            since_improvement += 1
            if since_improvement >= cfg.patience:
                break

    report.seconds = round(time.time() - started, 2)
    return report


def sample(model, n: int, cfg: DiffusionConfig, device=None,
          mask: np.ndarray | None = None, seed: int = 0,
          artifact_class: int | None = None,
          transient_class: int | None = None) -> np.ndarray:
    """Standard DDPM ancestral reverse sampling: start from Gaussian noise
    and iteratively denoise `cfg.timesteps` steps back to a sample.

    `mask` (n, patch_length) is the conditioning gap-validity channel fed
    to the denoiser at every step -- defaults to all-ones (every position
    "real") when not supplied, since a caller generating a fresh patch to
    splice into a real curve normally wants a fully-valid patch.

    `artifact_class`/`transient_class`: a single integer index (e.g.
    `artifact_patches.CATEGORY_NAMES.index("cosmic_ray")`), broadcast to
    all `n` samples, or `None` to sample the model's "unspecified" index
    for that channel. Ignored (with a clear error) if the loaded model
    does not have that channel enabled.
    """
    import torch

    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()

    if artifact_class is not None and cfg.n_artifact_classes == 0:
        raise ValueError("this model has no artifact conditioning channel enabled")
    if transient_class is not None and cfg.n_transient_classes == 0:
        raise ValueError("this model has no transient conditioning channel enabled")

    constants = diffusion_constants(cfg.timesteps)
    betas = torch.from_numpy(constants["betas"]).to(device)
    alphas = torch.from_numpy(constants["alphas"]).to(device)
    alphas_cumprod = torch.from_numpy(constants["alphas_cumprod"]).to(device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, cfg.patch_length, generator=generator).to(device)
    mask_array = np.ones((n, cfg.patch_length), dtype=np.float32) if mask is None else mask
    mask_t = torch.from_numpy(mask_array.astype(np.float32)).to(device)

    artifact_t = (torch.full((n,), int(artifact_class), dtype=torch.long, device=device)
                 if artifact_class is not None else None)
    transient_t = (torch.full((n,), int(transient_class), dtype=torch.long, device=device)
                  if transient_class is not None else None)

    with torch.no_grad():
        for step in reversed(range(cfg.timesteps)):
            t = torch.full((n,), step, dtype=torch.long, device=device)
            model_input = torch.stack([x, mask_t], dim=1)
            predicted_noise = model(model_input, t, artifact_t, transient_t).squeeze(1)

            alpha_t = alphas[step]
            alpha_bar_t = alphas_cumprod[step]
            beta_t = betas[step]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)
            if step > 0:
                noise = torch.randn(n, cfg.patch_length, generator=generator).to(device)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

    return x.cpu().numpy()


def load_diffusion_model(checkpoint_path: str | Path):
    """Restore a trained denoiser together with the config it was trained under."""
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = payload["config"]
    cfg = DiffusionConfig(**{**saved, "channels": tuple(saved["channels"])})
    model = make_denoiser(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, saved
