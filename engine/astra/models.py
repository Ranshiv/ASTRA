"""Convolutional autoencoder and VAE for light-curve sequences (plan phase 5).

Sizing is driven by the hardware this runs on: roughly 2.2 GB of usable VRAM
on a 4 GB card whose desktop already holds the rest. The defaults here total a
few hundred thousand parameters, which trains comfortably in that budget at
batch 256. They are deliberately small — plan section 13 warns against
assuming the newest, largest model is automatically better, and the honest
comparison against the Phase 4 baselines only means something if the deep
model is trained properly rather than starved.

Both models consume the 2-channel (value, mask) representation from
astra.tensors and reconstruct only the value channel, weighted by the mask, so
points invented across observing gaps never contribute to the loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# torch is imported lazily inside factories so the engine still starts, and
# every non-deep code path still works, on a machine with no PyTorch.


@dataclass
class ModelConfig:
    """Architecture and capacity, kept small enough for a 4 GB card."""

    length: int = 256
    channels: tuple[int, ...] = (16, 32, 64)
    latent_dim: int = 16
    kernel_size: int = 5
    dropout: float = 0.1
    patch_size: int = 16
    transformer_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2

    def to_dict(self) -> dict:
        return {
            "length": self.length,
            "channels": list(self.channels),
            "latent_dim": self.latent_dim,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "patch_size": self.patch_size,
            "transformer_dim": self.transformer_dim,
            "transformer_heads": self.transformer_heads,
            "transformer_layers": self.transformer_layers,
        }


def _build_encoder(torch, nn, config: ModelConfig):
    """Strided convolutions: each block halves the sequence length."""
    layers = []
    in_channels = 2  # value + validity mask
    for out_channels in config.channels:
        layers += [
            nn.Conv1d(in_channels, out_channels, config.kernel_size,
                      stride=2, padding=config.kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
        ]
        in_channels = out_channels
    return nn.Sequential(*layers)


def _build_decoder(torch, nn, config: ModelConfig):
    """Mirror of the encoder, ending at a single reconstructed channel."""
    layers = []
    reversed_channels = list(reversed(config.channels))
    for index, out_channels in enumerate(reversed_channels[1:]):
        layers += [
            nn.ConvTranspose1d(reversed_channels[index], out_channels,
                               config.kernel_size, stride=2,
                               padding=config.kernel_size // 2,
                               output_padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        ]
    layers += [
        nn.ConvTranspose1d(reversed_channels[-1], 1, config.kernel_size,
                           stride=2, padding=config.kernel_size // 2,
                           output_padding=1),
    ]
    return nn.Sequential(*layers)


def encoded_length(config: ModelConfig) -> int:
    length = config.length
    for _ in config.channels:
        length = (length + 1) // 2
    return length


def make_autoencoder(config: ModelConfig | None = None):
    """Plain convolutional autoencoder; reconstruction error is the score."""
    import torch
    from torch import nn

    config = config or ModelConfig()

    class ConvAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.encoder = _build_encoder(torch, nn, config)
            flat = config.channels[-1] * encoded_length(config)
            self.to_latent = nn.Linear(flat, config.latent_dim)
            self.from_latent = nn.Linear(config.latent_dim, flat)
            self.decoder = _build_decoder(torch, nn, config)

        def encode(self, x):
            hidden = self.encoder(x)
            return self.to_latent(hidden.flatten(1))

        def decode(self, latent):
            hidden = self.from_latent(latent)
            hidden = hidden.view(-1, self.config.channels[-1],
                                 encoded_length(self.config))
            # Convolution arithmetic can overshoot by a step; trim to match.
            return self.decoder(hidden)[..., :self.config.length]

        def forward(self, x):
            return self.decode(self.encode(x)), None, None

    return ConvAutoencoder()


def make_vae(config: ModelConfig | None = None):
    """Variational autoencoder.

    The reparameterisation trick is done in float32 even under mixed
    precision: sampling and the KL term involve exponentials of the log
    variance, which overflow readily in float16. This card is Turing and has
    no bfloat16, so float16 is the only mixed-precision option available and
    this guard is what keeps VAE training from collapsing to NaN.
    """
    import torch
    from torch import nn

    config = config or ModelConfig()

    class ConvVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.encoder = _build_encoder(torch, nn, config)
            flat = config.channels[-1] * encoded_length(config)
            self.to_mu = nn.Linear(flat, config.latent_dim)
            self.to_logvar = nn.Linear(flat, config.latent_dim)
            self.from_latent = nn.Linear(config.latent_dim, flat)
            self.decoder = _build_decoder(torch, nn, config)

        def encode(self, x):
            hidden = self.encoder(x).flatten(1)
            # Clamped so exp(logvar) cannot overflow before the float32 cast.
            return self.to_mu(hidden), self.to_logvar(hidden).clamp(-10.0, 10.0)

        def reparameterise(self, mu, logvar):
            mu32, logvar32 = mu.float(), logvar.float()
            std = torch.exp(0.5 * logvar32)
            return mu32 + std * torch.randn_like(std)

        def decode(self, latent):
            hidden = self.from_latent(latent)
            hidden = hidden.view(-1, self.config.channels[-1],
                                 encoded_length(self.config))
            return self.decoder(hidden)[..., :self.config.length]

        def forward(self, x):
            mu, logvar = self.encode(x)
            latent = self.reparameterise(mu, logvar)
            return self.decode(latent.to(mu.dtype)), mu, logvar

    return ConvVAE()


def make_transformer(config: ModelConfig | None = None):
    """Patch-based transformer autoencoder for bounded long sequences.

    Attention operates on ``ceil(length / patch_size)`` tokens rather than
    individual cadences, changing memory from O(L²) to O((L/P)²).  The model
    still reconstructs the observed channel and uses the same masked loss as
    the convolutional baselines.  It is intentionally small and remains a
    development-build option until Stage-B comparison proves its value.
    """
    import torch
    from torch import nn

    config = config or ModelConfig()
    patch = max(2, int(config.patch_size))
    dimension = max(8, int(config.transformer_dim))
    heads = max(1, int(config.transformer_heads))
    if dimension % heads:
        raise ValueError("transformer_dim must be divisible by transformer_heads")
    layers = max(1, int(config.transformer_layers))
    token_count = (config.length + patch - 1) // patch

    class PatchTransformerAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.patch = patch
            self.token_count = token_count
            self.embed = nn.Conv1d(2, dimension, kernel_size=patch, stride=patch)
            self.position = nn.Parameter(torch.zeros(1, token_count, dimension))
            layer = nn.TransformerEncoderLayer(
                d_model=dimension, nhead=heads,
                dim_feedforward=dimension * 4,
                dropout=config.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.decode = nn.ConvTranspose1d(
                dimension, 1, kernel_size=patch, stride=patch,
            )

        def forward(self, x):
            tokens = self.embed(x)
            tokens = tokens.transpose(1, 2) + self.position[:, :tokens.shape[2], :]
            tokens = self.encoder(tokens)
            output = self.decode(tokens.transpose(1, 2))[..., :self.config.length]
            return output, None, None

    return PatchTransformerAutoencoder()


def masked_reconstruction_loss(prediction, target, mask, reduction: str = "mean"):
    """Mean squared error over observed points only.

    Without the mask the model would be rewarded for reproducing the zeros
    that fill observing gaps, which is not a fact about the source.
    """
    import torch

    error = (prediction.squeeze(1) - target) ** 2 * mask
    denominator = mask.sum(dim=-1).clamp(min=1.0)
    per_row = error.sum(dim=-1) / denominator
    if reduction == "none":
        return per_row
    return per_row.mean()


def kl_divergence(mu, logvar):
    """KL to a unit Gaussian, computed in float32 for numerical safety."""
    import torch

    mu32, logvar32 = mu.float(), logvar.float()
    return -0.5 * torch.mean(
        torch.sum(1 + logvar32 - mu32.pow(2) - logvar32.exp(), dim=1)
    )


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODEL_FACTORIES = {
    "autoencoder": make_autoencoder,
    "vae": make_vae,
    "transformer": make_transformer,
}


def make(kind: str, config: ModelConfig | None = None):
    if kind not in MODEL_FACTORIES:
        raise KeyError(f"unknown model {kind!r}; available: {sorted(MODEL_FACTORIES)}")
    return MODEL_FACTORIES[kind](config)
