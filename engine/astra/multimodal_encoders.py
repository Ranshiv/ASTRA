"""Four modality encoders plus the physical-scale-token fusion mechanism
(backlog item 11).

Each branch (image, spectrum, light curve, catalog metadata) produces a
pooled embedding of the SAME width (`embedding_dim`), then fuses in a real
per-object absolute-scale scalar before the shared contrastive projection
head lives in `multimodal_moco.py`. Fusing scale INTO the embedding is what
makes "brightness-preservation error" well-defined: `tensors.py`'s
MAD-normalised light-curve representation (and the analogous
background-subtracted image/spectrum inputs here) discards absolute
brightness by design (`docs/DEFERRED.txt`'s `[KNOWN] Sequence representation
discards absolute brightness` entry) -- "a model that needs it must consume
both." A frozen probe trained on the fused embedding (`multimodal_eval.py`)
is how that claim gets checked, not assumed.

The image branch is the first Conv2d/2-D model in this codebase; everything
else here (spectrum, light curve) uses 1-D convolutions over sequences, the
same shape family `models.py`/`pretrain.py` already use. The light-curve
branch is not reimplemented at all -- it delegates to `pretrain.make_encoder`
(backlog item 13), reused verbatim.
"""

from __future__ import annotations

import numpy as np

DEFAULT_EMBEDDING_DIM = 64
CATALOG_FEATURE_COUNT = 40  # features.FEATURE_NAMES (29) + GAIA_JOIN_COLUMNS (11)


def signed_log_scale(x: np.ndarray | float) -> np.ndarray | float:
    """sign(x) * log1p(|x|) -- one transform shared across all four
    modalities' otherwise differently-signed/ranged raw scale scalars
    (flux vs. magnitude vs. continuum level), so the scale-token MLP sees
    comparable input statistics regardless of source. Invertible via
    `inverse_signed_log_scale`."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_signed_log_scale(y: np.ndarray | float) -> np.ndarray | float:
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * np.expm1(np.abs(y))


def resample_spectrum(wavelength: np.ndarray, flux: np.ndarray, error: np.ndarray,
                      length: int = 256) -> np.ndarray:
    """A real (wavelength, flux, error) triple onto a fixed-length grid.

    Returns (3, length): log10(wavelength), flux, error -- log-wavelength
    because a spectrum's real wavelength range spans a much narrower
    relative span than a light curve's time axis, and log spacing is the
    standard way to represent it without one end of the grid being far
    denser in real wavelength than the other.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    mask = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(error) & (wavelength > 0)
    wavelength, flux, error = wavelength[mask], flux[mask], error[mask]
    if len(wavelength) < 2:
        raise ValueError("resample_spectrum needs at least two finite points")

    order = np.argsort(wavelength)
    wavelength, flux, error = wavelength[order], flux[order], error[order]
    log_wave = np.log10(wavelength)
    grid = np.linspace(log_wave[0], log_wave[-1], length)

    flux_grid = np.interp(grid, log_wave, flux).astype(np.float32)
    error_grid = np.interp(grid, log_wave, error).astype(np.float32)
    return np.stack([grid.astype(np.float32), flux_grid, error_grid], axis=0)


def make_image_encoder(embedding_dim: int = DEFAULT_EMBEDDING_DIM):
    """Conv2d(1,16,3)->ReLU->MaxPool2->Conv2d(16,32,3)->ReLU->MaxPool2->
    Conv2d(32,64,3)->ReLU->AdaptiveAvgPool2d(1)->Linear(64,embedding_dim).

    Single-band grayscale input: neither a ZTF cutout nor a TESS TPF
    carries multi-band pixel data in one file, so a multi-channel input
    would be fabricated. ~27K parameters -- negligible against the ~2.2 GB
    VRAM budget; the queue (multimodal_moco.py), not model capacity, is
    what this backlog item's blocker was ever about.
    """
    import torch
    from torch import nn

    class ImageEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.project = nn.Linear(64, embedding_dim)

        def pooled(self, x):
            # x: (batch, 1, H, W)
            features = self.net(x).flatten(1)
            return self.project(features)

        def forward(self, x):
            return self.pooled(x)

    return ImageEncoder()


def make_spectrum_encoder(embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                          length: int = 256, patch_size: int = 16,
                          transformer_heads: int = 4, transformer_layers: int = 2,
                          dropout: float = 0.1):
    """Mirrors `pretrain.make_encoder`'s patch-embedding shape exactly, with
    3 input channels (log-wavelength, flux, error) instead of 2
    (value, gap-mask)."""
    import torch
    from torch import nn

    patch = max(2, int(patch_size))
    dimension = max(8, int(embedding_dim))
    heads = max(1, int(transformer_heads))
    if dimension % heads:
        raise ValueError("embedding_dim must be divisible by transformer_heads")
    layers = max(1, int(transformer_layers))
    token_count = (length + patch - 1) // patch

    class SpectrumEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.length = length
            self.patch = patch
            self.embed = nn.Conv1d(3, dimension, kernel_size=patch, stride=patch)
            self.position = nn.Parameter(torch.zeros(1, token_count, dimension))
            layer = nn.TransformerEncoderLayer(
                d_model=dimension, nhead=heads, dim_feedforward=dimension * 4,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

        def forward(self, x):
            tokens = self.embed(x).transpose(1, 2)
            tokens = tokens + self.position[:, :tokens.shape[1], :]
            return self.encoder(tokens)

        def pooled(self, x):
            return self.forward(x).mean(dim=1)

    return SpectrumEncoder()


def make_catalog_encoder(embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                         n_features: int = CATALOG_FEATURE_COUNT,
                         hidden: int = 128):
    """Linear(n_features, hidden) -> GELU -> Linear(hidden, embedding_dim)
    over a standardised row of `features.FEATURE_NAMES` +
    `featurematrix.GAIA_JOIN_COLUMNS` -- NOT
    `GAIA_EXTINCTION_IDENTITY_KEYS`, which stay per-row identity metadata,
    never a value column (see `featurematrix.py`'s own documented reason)."""
    from torch import nn

    class CatalogEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden), nn.GELU(),
                nn.Linear(hidden, embedding_dim),
            )

        def pooled(self, x):
            return self.net(x)

        def forward(self, x):
            return self.pooled(x)

    return CatalogEncoder()


def make_lightcurve_encoder(embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                            length: int = 256, patch_size: int = 16,
                            transformer_heads: int = 4, transformer_layers: int = 2,
                            dropout: float = 0.1, checkpoint: str | None = None):
    """Delegates to `pretrain.make_encoder`/`pretrain.load_pretrained_encoder`
    (backlog item 13) -- zero reimplementation of the light-curve branch."""
    from . import pretrain

    if checkpoint is not None:
        encoder, _ = pretrain.load_pretrained_encoder(checkpoint)
        return encoder

    config = pretrain.PretrainConfig(
        length=length, patch_size=patch_size, transformer_dim=embedding_dim,
        transformer_heads=transformer_heads, transformer_layers=transformer_layers,
        dropout=dropout,
    )
    return pretrain.make_encoder(config)


def make_scale_token(embedding_dim: int = DEFAULT_EMBEDDING_DIM, hidden: int = 16):
    """Linear(1, hidden) -> GELU -> Linear(hidden, embedding_dim): embeds one
    real, signed-log-scaled absolute-scale scalar."""
    from torch import nn

    return nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, embedding_dim))


def make_scale_fusion(embedding_dim: int = DEFAULT_EMBEDDING_DIM):
    """Linear(2*embedding_dim, embedding_dim) over concat(pooled, scale) --
    the FUSED embedding that enters the shared contrastive projection head,
    so the loss pulls together shape AND scale jointly, not shape alone."""
    from torch import nn

    return nn.Linear(embedding_dim * 2, embedding_dim)


def make_projection_head(embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                         projection_dim: int = 32):
    """Linear(D,D)->GELU->Linear(D,P): the contrastive projection head every
    modality shares the SHAPE of (each modality gets its own instance).
    L2-normalisation happens in `multimodal_moco.py`'s forward pass, not
    here, so this stays a plain linear projection reusable by both the
    online and momentum copies."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim), nn.GELU(),
        nn.Linear(embedding_dim, projection_dim),
    )


def encode_and_fuse(modules, x, raw_scale):
    """Shared forward path: pooled embedding -> scale-token fusion.

    `modules` is any object exposing `.encoder` (with `.pooled(x)`),
    `.scale_token`, `.fusion` -- an `nn.ModuleDict` with those three keys
    satisfies this, which is exactly how `multimodal_moco.py` assembles
    each modality's online/momentum pair. `raw_scale` is `(batch,)` or
    `(batch, 1)`, already `signed_log_scale`-transformed. Returns the fused
    embedding, `(batch, embedding_dim)`.
    """
    import torch

    pooled = modules["encoder"].pooled(x)
    if raw_scale.dim() == 1:
        raw_scale = raw_scale.unsqueeze(-1)
    scale_embedding = modules["scale_token"](raw_scale)
    return modules["fusion"](torch.cat([pooled, scale_embedding], dim=-1))
