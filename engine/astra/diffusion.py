"""Minimal 1-D denoising diffusion model (backlog item 14).

Generates short, spliceable "morphology patches" for the open-world
injection simulator in `open_world_injection.py`. Unlike `models.py`'s
plain convolutional autoencoder/VAE (a single encoder->decoder pass with no
skip connections), this is a genuine U-Net: each down-sampling stage's
activation is kept and concatenated back into the matching up-sampling
stage, and a sinusoidal timestep embedding is broadcast into every block --
the standard DDPM (Ho et al. 2020, "Denoising Diffusion Probabilistic
Models") architecture, at the smallest scale that still has a real skip
path, since this project's own house rule (`docs/LIMITATIONS.md`'s repeated
"don't assume newer/bigger is automatically better, measure it") argues
against building more capacity than the data/hardware need.

Only the VALUE channel is noised and denoised. The gap-validity MASK
channel (same convention `tensors.py` uses everywhere else in this
codebase) is treated as conditioning information -- which positions in the
patch are real observations -- not a quantity to generate, and
`diffusion_loss` only scores the noise prediction at those real positions,
mirroring `models.masked_reconstruction_loss`'s exact masking discipline.

`timesteps` defaults to 100, not the 1000+ common in image-diffusion
literature: sampling cost (an iterative reverse process, one forward pass
per step) is the genuine expense of a diffusion model on this hardware, not
parameter count or per-step VRAM (both negligible at this scale, confirmed
before writing this module) -- so T is chosen for affordable sampling, not
maximum sample fidelity.

The training loop and sampler live in `diffusion_train.py`, split out
purely to keep each file under this project's 500-line guideline (the same
reason `pretrain.py`/`pretrain_probe.py` were split earlier this session).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_PATCH_LENGTH = 32


@dataclass
class DiffusionConfig:
    patch_length: int = DEFAULT_PATCH_LENGTH
    channels: tuple[int, ...] = (16, 32, 64)
    timesteps: int = 100
    kernel_size: int = 5
    dropout: float = 0.1
    time_embed_dim: int = 32
    # 0 disables a channel entirely -- with both at 0 the model is
    # structurally identical to the unconditional version (same parameter
    # names, same shapes), so every existing unconditional caller/test
    # keeps working unchanged. Two INDEPENDENT channels, not one combined
    # label space: a patch can carry an artifact category
    # (`artifact_patches.CATEGORY_NAMES`), a real transient class, both, or
    # neither.
    n_artifact_classes: int = 0
    n_transient_classes: int = 0
    epochs: int = 30
    learning_rate: float = 1e-3
    batch_size: int | None = None
    effective_batch_size: int = 256
    seed: int = 42
    patience: int = 6
    amp: bool = True

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["channels"] = list(self.channels)
        return payload


@dataclass
class DiffusionReport:
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


def linear_beta_schedule(timesteps: int) -> np.ndarray:
    """Standard DDPM linear variance schedule, beta_1=1e-4 to beta_T=0.02."""
    return np.linspace(1e-4, 0.02, timesteps, dtype=np.float64)


def diffusion_constants(timesteps: int) -> dict:
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": np.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - alphas_cumprod),
    }


def forward_diffusion_sample(x0_value: np.ndarray, t: np.ndarray, noise: np.ndarray,
                             constants: dict) -> np.ndarray:
    """x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise, per-row t.

    `x0_value`/`noise` are (batch, length); `t` is (batch,) integer
    timesteps. Only the value channel is noised -- see module docstring.
    """
    sqrt_alpha_bar = constants["sqrt_alphas_cumprod"][t][:, None]
    sqrt_one_minus_alpha_bar = constants["sqrt_one_minus_alphas_cumprod"][t][:, None]
    return sqrt_alpha_bar * x0_value + sqrt_one_minus_alpha_bar * noise


def _sinusoidal_embedding(t, dim: int):
    """Standard transformer-style sinusoidal timestep embedding."""
    import torch

    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def make_denoiser(config: DiffusionConfig | None = None):
    """A small 1-D U-Net predicting the noise added to a patch's value
    channel, conditioned on the timestep and the patch's own gap mask.
    """
    import torch
    from torch import nn

    config = config or DiffusionConfig()
    channels = config.channels
    kernel = config.kernel_size
    time_dim = config.time_embed_dim

    class DownBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel,
                                  stride=2, padding=kernel // 2)
            self.norm = nn.BatchNorm1d(out_channels)
            self.time_proj = nn.Linear(time_dim, out_channels)
            self.act = nn.GELU()
            self.drop = nn.Dropout(config.dropout)

        def forward(self, x, time_embed):
            h = self.act(self.norm(self.conv(x)))
            h = h + self.time_proj(time_embed).unsqueeze(-1)
            return self.drop(h)

    class UpBlock(nn.Module):
        """`skip=None` for the final stage: by then the up-path is back at
        the input's own resolution, and the input (2 raw channels) is not a
        down-path feature map in the U-Net sense -- there is nothing
        meaningful left to concatenate, so the final stage is a plain
        upsampling convolution."""

        def __init__(self, in_channels: int, skip_channels: int | None,
                    out_channels: int) -> None:
            super().__init__()
            self.has_skip = skip_channels is not None
            # Upsampling (ConvTranspose1d, changes LENGTH) and channel
            # mixing (Conv1d, changes CHANNELS) are kept as two separate
            # steps: `x` and `skip` only have matching length AFTER
            # upsampling, so they cannot be concatenated before it.
            self.upsample = nn.ConvTranspose1d(in_channels, in_channels, kernel,
                                               stride=2, padding=kernel // 2,
                                               output_padding=1)
            conv_in = in_channels + (skip_channels if self.has_skip else 0)
            self.conv = nn.Conv1d(conv_in, out_channels, kernel, padding=kernel // 2)
            self.final = not self.has_skip
            if not self.final:
                self.norm = nn.BatchNorm1d(out_channels)
                self.time_proj = nn.Linear(time_dim, out_channels)
                self.act = nn.GELU()

        def forward(self, x, skip, time_embed):
            upsampled = self.upsample(x)
            if self.has_skip:
                upsampled = upsampled[..., :skip.shape[-1]]
                h = self.conv(torch.cat([upsampled, skip], dim=1))
            else:
                h = self.conv(upsampled)
            if self.final:
                return h
            h = self.act(self.norm(h))
            return h + self.time_proj(time_embed).unsqueeze(-1)

    class UNetDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim * 2), nn.GELU(),
                nn.Linear(time_dim * 2, time_dim),
            )
            # Two INDEPENDENT conditioning channels, each with one extra
            # "unspecified" index (the class count itself) -- a patch can
            # carry an artifact category, a transient class, both, or
            # neither. Fused by ADDITION into the shared timestep
            # embedding before it's broadcast into every block, the same
            # "fuse a real condition into a shared embedding" pattern
            # `multimodal_encoders.encode_and_fuse` already established
            # this session. With both counts at 0 (the default), neither
            # embedding table is created and the model is structurally
            # identical to the unconditional version.
            self.artifact_embed = (
                nn.Embedding(config.n_artifact_classes + 1, time_dim)
                if config.n_artifact_classes > 0 else None)
            self.transient_embed = (
                nn.Embedding(config.n_transient_classes + 1, time_dim)
                if config.n_transient_classes > 0 else None)

            self.down_blocks = nn.ModuleList()
            in_channels = 2  # noisy value + gap mask (conditioning)
            for out_channels in channels:
                self.down_blocks.append(DownBlock(in_channels, out_channels))
                in_channels = out_channels

            # channels=(16,32,64) -> reversed_channels=[64,32,16]. Each
            # up-block upsamples from reversed_channels[i] and concatenates
            # the down-path activation at the matching resolution, which
            # always has reversed_channels[i+1] channels by construction of
            # this doubling scheme.
            reversed_channels = list(reversed(channels))
            self.up_blocks = nn.ModuleList()
            for index in range(len(reversed_channels) - 1):
                out_channels = reversed_channels[index + 1]
                self.up_blocks.append(
                    UpBlock(reversed_channels[index], out_channels, out_channels))
            self.up_blocks.append(UpBlock(reversed_channels[-1], None, 1))

        def forward(self, x, t, artifact_class=None, transient_class=None):
            time_embed = self.time_mlp(_sinusoidal_embedding(t, time_dim))

            if self.artifact_embed is not None:
                if artifact_class is None:
                    artifact_class = torch.full(
                        (x.shape[0],), config.n_artifact_classes,
                        dtype=torch.long, device=x.device)
                time_embed = time_embed + self.artifact_embed(artifact_class)

            if self.transient_embed is not None:
                if transient_class is None:
                    transient_class = torch.full(
                        (x.shape[0],), config.n_transient_classes,
                        dtype=torch.long, device=x.device)
                time_embed = time_embed + self.transient_embed(transient_class)

            skips = []
            h = x
            for block in self.down_blocks:
                h = block(h, time_embed)
                skips.append(h)

            bottleneck, shallow_skips = skips[-1], skips[:-1][::-1]
            h = bottleneck
            for block, skip in zip(self.up_blocks[:-1], shallow_skips):
                h = block(h, skip, time_embed)
            h = self.up_blocks[-1](h, None, time_embed)
            return h[..., :x.shape[-1]]

    return UNetDenoiser()


def diffusion_loss(predicted_noise, true_noise, mask, reduction: str = "mean"):
    """Masked MSE between predicted and true noise, restricted to real
    (non-interpolated) points -- same masking discipline as
    `models.masked_reconstruction_loss`, applied to a noise target instead
    of a value target."""
    error = (predicted_noise.squeeze(1) - true_noise) ** 2 * mask
    denominator = mask.sum(dim=-1).clamp(min=1.0)
    per_row = error.sum(dim=-1) / denominator
    if reduction == "none":
        return per_row
    return per_row.mean()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

