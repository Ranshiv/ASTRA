"""Masked self-supervised light-curve pretraining (plan phase 5, backlog item 13).

Two per-sample pretext tasks, trained jointly through one shared patch-
embedding transformer encoder, on UNLABELED 2-channel (value, gap-mask)
sequences from `tensors.py`:

1. Span-masking reconstruction -- additional spans of currently-VALID points
   are hidden and the model must reconstruct just those points.
2. Time-order corruption detection -- non-adjacent patches are swapped and a
   head trained to detect whether a sequence's patch order was corrupted.

Deliberately NOT the contrastive objective `docs/DEFERRED.txt` marks
``[BLOCKED] Multimodal encoder (section 14) and contrastive learning``: that
approach needs large in-batch negative counts (1024-4096) that do not fit in
this machine's ~2.2 GB usable VRAM. Both objectives here are per-sample --
no in-batch negatives -- so that VRAM constraint does not block this module.

The pretraining mask this module builds (`apply_span_mask`'s `pretrain_mask`)
is a THIRD, separate concept from `tensors.py`'s gap-validity mask (channel
1: "not interpolated across an observing gap"). The two are never conflated:
`apply_span_mask` only ever samples spans from points the gap mask already
marks valid, and never writes into channel 1.

No real human-labelled data exists yet (Phase 7's human-in-the-loop
labelling system is not built). `probe_transfer`'s "1%/10%/100% labels"
metric is therefore measured against `evaluate.build_injected()`'s
true-by-construction injection labels, the same honest stand-in
`evaluate.py` already uses everywhere else in this codebase, with the same
stated caveat: it measures transfer to the anomaly types actually injected,
not to whatever a human would eventually flag.

Like every other opt-in research module in this codebase (`multiband_hier.py`,
`moving_objects.py`, `tess_psf.py`), this is deliberately NOT wired into
`rpc.py`, `scoring.WEIGHTS`, or `evidence.py`: new evidence about a training
strategy, not a production score change.

The label-fraction transfer evaluation itself (`probe_transfer`) lives in
`pretrain_probe.py`, a sibling module, purely to keep each file under this
project's 500-line guideline -- pretraining and transfer-evaluation are two
genuinely separable concerns (produce an encoder vs. measure what it's worth)
and `pretrain_probe.py` only ever consumes this module's public
`load_pretrained_encoder` output, never its internals.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# A row with fewer real, valid points than this cannot support a meaningful
# span mask; span-masking it is skipped rather than degraded, matching
# tensors.resample()'s MIN_POINTS skip convention.
MIN_VALID_FOR_MASK = 8

# Bounds a span-masking attempt loop that stops once enough previously-valid
# points have been hidden; caps runaway looping on a sparsely-valid row.
SPAN_MASK_MAX_ATTEMPTS = 200
SPAN_MASK_MAX_LENGTH_FACTOR = 4

# A row needs at least this many patches to swap two non-adjacent ones.
MIN_TOKENS_FOR_ORDER_TASK = 4
ORDER_TASK_MAX_ATTEMPTS = 50


@dataclass
class PretrainConfig:
    """Encoder shape mirrors ModelConfig's naming; the rest is pretext-specific."""

    length: int = 256
    patch_size: int = 16
    transformer_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.1
    span_mask_ratio: float = 0.15
    span_mask_mean_length: int = 8
    order_shuffle_fraction: float = 0.5
    order_num_swaps: int = 2
    reconstruction_weight: float = 1.0
    order_weight: float = 1.0
    epochs: int = 30
    learning_rate: float = 1e-3
    batch_size: int | None = None
    effective_batch_size: int = 256
    seed: int = 42
    patience: int = 6
    amp: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PretrainReport:
    device: str
    device_reason: str
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    train_losses: list[float] = field(default_factory=list)
    reconstruction_losses: list[float] = field(default_factory=list)
    order_losses: list[float] = field(default_factory=list)
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


def make_encoder(config: PretrainConfig | None = None):
    """Patch-embedding transformer encoder, reused by pretraining and probing.

    Mirrors `models.make_transformer`'s patch-embed shape (Conv1d(2, dim,
    kernel=patch, stride=patch) -> learned position -> TransformerEncoder) so
    results stay comparable, but returns TOKEN embeddings (batch, tokens,
    dim), not a decoded (batch, 1, length) reconstruction: this encoder is
    shared by two different heads plus a pooling method for the linear
    probe, none of which fit the (prediction, mu, logvar) contract every
    `models.MODEL_FACTORIES` entry shares -- that is why it lives here
    rather than joining that dict.
    """
    import torch
    from torch import nn

    config = config or PretrainConfig()
    patch = max(2, int(config.patch_size))
    dimension = max(8, int(config.transformer_dim))
    heads = max(1, int(config.transformer_heads))
    if dimension % heads:
        raise ValueError("transformer_dim must be divisible by transformer_heads")
    layers = max(1, int(config.transformer_layers))
    token_count = (config.length + patch - 1) // patch

    class PatchEncoder(nn.Module):
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

        def forward(self, x):
            tokens = self.embed(x).transpose(1, 2)
            tokens = tokens + self.position[:, :tokens.shape[1], :]
            return self.encoder(tokens)

        def pooled(self, x):
            """Mean pool over tokens -- the embedding the linear probe consumes."""
            return self.forward(x).mean(dim=1)

    return PatchEncoder()


def make_pretrain_model(config: PretrainConfig | None = None):
    """Shared encoder plus a per-patch reconstruction head and an order head."""
    import torch
    from torch import nn

    config = config or PretrainConfig()

    class MaskedOrderPretrainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.encoder = make_encoder(config)
            dim = self.encoder.embed.out_channels
            self.reconstruction_head = nn.Linear(dim, self.encoder.patch)
            self.order_head = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1),
            )

        def forward(self, x):
            tokens = self.encoder(x)
            per_patch = self.reconstruction_head(tokens)
            reconstruction = per_patch.reshape(per_patch.shape[0], -1)[:, :self.config.length]
            order_logit = self.order_head(tokens.mean(dim=1)).squeeze(-1)
            return reconstruction, order_logit

    return MaskedOrderPretrainer()


def apply_span_mask(values: np.ndarray, rng: np.random.Generator,
                    mask_ratio: float = 0.15,
                    mean_span_length: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Hide additional spans among currently-valid points for reconstruction.

    `values` is (batch, 2, length): channel 0 the normalised value, channel 1
    the gap-validity mask `tensors.py` already computed. Spans are sampled
    ONLY from points where that mask is 1 -- a point already zeroed as
    "interpolated across an observing gap" carries no real value to
    reconstruct, and masking one would train the model to predict a
    fabricated interpolated point rather than a real observation.

    Returns `(masked_input, pretrain_mask)`. `pretrain_mask` (batch, length)
    is 1 wherever THIS function hid a point -- a separate concept from the
    gap mask, never written into channel 1. The gap-mask channel of
    `masked_input` is left untouched, so the model still sees "this was a
    real observation", just with its value withheld.
    """
    batch = values.shape[0]
    length = values.shape[-1]
    masked = values.copy()
    pretrain_mask = np.zeros((batch, length), dtype=np.float32)
    max_span = max(1, int(mean_span_length * SPAN_MASK_MAX_LENGTH_FACTOR))

    for row in range(batch):
        gap_mask = values[row, 1]
        n_valid = int(gap_mask.sum())
        if n_valid < MIN_VALID_FOR_MASK:
            continue

        target = max(1, int(round(mask_ratio * n_valid)))
        hidden = 0
        attempts = 0
        while hidden < target and attempts < SPAN_MASK_MAX_ATTEMPTS:
            attempts += 1
            start = int(rng.integers(0, length))
            span = int(np.clip(rng.geometric(1.0 / mean_span_length), 1, max_span))
            end = min(length, start + span)

            newly_hidden = (gap_mask[start:end] > 0) & (pretrain_mask[row, start:end] == 0)
            if not newly_hidden.any():
                continue

            positions = np.arange(start, end)[newly_hidden]
            masked[row, 0, positions] = 0.0
            pretrain_mask[row, positions] = 1.0
            hidden += int(newly_hidden.sum())

    return masked, pretrain_mask


def span_reconstruction_loss(prediction, original_values, pretrain_mask, gap_mask):
    """Masked MSE restricted to points that are BOTH a pretraining target and
    a real observation -- a point the gap mask already marked invalid is
    never credited either way, even if `apply_span_mask` never targets it."""
    target_mask = pretrain_mask * gap_mask
    error = (prediction - original_values) ** 2 * target_mask
    denominator = target_mask.sum(dim=-1).clamp(min=1.0)
    return (error.sum(dim=-1) / denominator).mean()


def make_order_task(values: np.ndarray, rng: np.random.Generator, patch: int,
                    shuffle_fraction: float = 0.5,
                    num_swaps: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Swap non-adjacent patches in a subset of rows; label 1 = corrupted.

    Value and gap-mask channels are swapped TOGETHER, so a shuffled row's
    gap-mask stays consistent with its own value at every position. Only
    full patches are eligible (a ragged final partial patch is left alone).
    Adjacent patches are never swapped -- too weak a corruption to give the
    order head a clean label boundary. A row with too few full patches to
    swap, or where two disjoint non-adjacent pairs cannot be found, is left
    unshuffled and labelled 0 rather than forced into a degenerate swap.
    """
    batch, channels, length = values.shape
    out = values.copy()
    labels = np.zeros(batch, dtype=np.float32)

    token_count = length // patch
    if token_count < MIN_TOKENS_FOR_ORDER_TASK:
        return out, labels

    n_shuffle = int(round(shuffle_fraction * batch))
    shuffle_rows = rng.choice(batch, size=min(n_shuffle, batch), replace=False)

    for row in shuffle_rows:
        used: set[int] = set()
        pairs: list[tuple[int, int]] = []
        attempts = 0
        while len(pairs) < num_swaps and attempts < ORDER_TASK_MAX_ATTEMPTS:
            attempts += 1
            i, j = rng.choice(token_count, size=2, replace=False)
            i, j = int(i), int(j)
            if abs(i - j) < 2 or i in used or j in used:
                continue
            pairs.append((i, j))
            used.add(i)
            used.add(j)

        if not pairs:
            continue

        for i, j in pairs:
            left = slice(i * patch, (i + 1) * patch)
            right = slice(j * patch, (j + 1) * patch)
            out[row, :, left], out[row, :, right] = (
                values[row, :, right].copy(), values[row, :, left].copy())
        labels[row] = 1.0

    return out, labels


def _pretrain_losses(model, batch: np.ndarray, rng: np.random.Generator,
                     cfg: PretrainConfig, device):
    """One step's combined loss, plus its two components, for reuse in both
    the training loop (with grad) and validation (under no_grad)."""
    import torch
    from torch.nn import functional as F

    masked_input, pretrain_mask = apply_span_mask(
        batch, rng, cfg.span_mask_ratio, cfg.span_mask_mean_length)
    shuffled_input, order_labels = make_order_task(
        batch, rng, model.encoder.patch, cfg.order_shuffle_fraction, cfg.order_num_swaps)

    reconstruction, _ = model(torch.from_numpy(masked_input).to(device))
    _, order_logit = model(torch.from_numpy(shuffled_input).to(device))

    original = torch.from_numpy(batch[:, 0, :]).to(device)
    gap_mask = torch.from_numpy(batch[:, 1, :]).to(device)
    pretrain_mask_t = torch.from_numpy(pretrain_mask).to(device)
    order_labels_t = torch.from_numpy(order_labels).to(device)

    recon_loss = span_reconstruction_loss(reconstruction, original, pretrain_mask_t, gap_mask)
    order_loss = F.binary_cross_entropy_with_logits(order_logit, order_labels_t)
    total = cfg.reconstruction_weight * recon_loss + cfg.order_weight * order_loss
    return total, recon_loss, order_loss


def pretrain(unlabeled_values: np.ndarray, val_values: np.ndarray,
            cfg: PretrainConfig | None = None,
            checkpoint_dir: Path | None = None,
            name: str = "pretrain") -> PretrainReport:
    """Train the encoder jointly on span-reconstruction + order-classification.

    Deliberately its own loop, not `train.train()`: `train._loss_for`
    unpacks `(prediction, mu, logvar)` against a single masked-MSE-plus-
    optional-KL contract with no slot for a second head/second loss or for
    two independently-corrupted views of the same batch per step. This loop
    instead reuses the genuinely kind-agnostic pieces of `train.py` it does
    not need to reinvent: VRAM-aware batch sizing, the OOM guard, seeding,
    and the same AMP/accumulation/early-stopping/checkpoint-on-improvement
    shape `train.train()` already established.
    """
    import torch
    from . import config as config_mod, hardware
    from . import train as train_mod

    cfg = cfg or PretrainConfig()
    train_mod._set_seed(cfg.seed)
    data_rng = np.random.default_rng(cfg.seed)

    device_report = hardware.select_device()
    device = torch.device(device_report.device)

    batch_size = train_mod.choose_batch_size(cfg.length, cfg.batch_size)
    accumulation = max(1, cfg.effective_batch_size // max(batch_size, 1))

    model = make_pretrain_model(cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    report = PretrainReport(
        device=device_report.device, device_reason=device_report.reason,
        epochs_run=0, best_epoch=-1, best_val_loss=float("inf"),
        batch_size=batch_size, accumulation_steps=accumulation,
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        amp_enabled=use_amp, seed=cfg.seed,
    )

    checkpoint_dir = checkpoint_dir or config_mod.PATHS.models
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_pretrain.pt"

    started = time.time()
    since_improvement = 0
    n = len(unlabeled_values)

    for epoch in range(cfg.epochs):
        model.train()
        order = data_rng.permutation(n) if n else np.empty(0, dtype=int)
        running_total = running_recon = running_order = 0.0
        steps = 0

        with train_mod._cuda_oom_guard():
            optimiser.zero_grad(set_to_none=True)
            for step, start in enumerate(range(0, n, max(batch_size, 1))):
                idx = order[start:start + batch_size]
                batch = unlabeled_values[idx]
                with torch.amp.autocast("cuda", enabled=use_amp):
                    total_loss, recon_loss, order_loss = _pretrain_losses(
                        model, batch, data_rng, cfg, device)
                scaler.scale(total_loss / accumulation).backward()

                if (step + 1) % accumulation == 0:
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad(set_to_none=True)

                running_total += float(total_loss.detach())
                running_recon += float(recon_loss.detach())
                running_order += float(order_loss.detach())
                steps += 1

            if steps and steps % accumulation != 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

        report.train_losses.append(running_total / max(steps, 1))
        report.reconstruction_losses.append(running_recon / max(steps, 1))
        report.order_losses.append(running_order / max(steps, 1))

        val_loss = _evaluate_pretrain_loss(model, val_values, cfg, device)
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


def _evaluate_pretrain_loss(model, values: np.ndarray, cfg: PretrainConfig, device) -> float:
    import torch

    if len(values) == 0:
        return float("nan")

    model.eval()
    eval_rng = np.random.default_rng(cfg.seed + 1)
    with torch.no_grad():
        total_loss, _, _ = _pretrain_losses(model, values, eval_rng, cfg, device)
    return float(total_loss.detach())


def load_pretrained_encoder(checkpoint_path: str | Path):
    """Restore a pretrained encoder together with the config it was trained under."""
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = PretrainConfig(**payload["config"])
    model = make_pretrain_model(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model.encoder, payload["config"]
