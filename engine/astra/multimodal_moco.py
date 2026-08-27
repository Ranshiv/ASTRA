"""MoCo-style momentum contrast across four modalities (backlog item 11).

Resolves `docs/DEFERRED.txt`'s `[BLOCKED] Multimodal encoder ... and
contrastive learning` entry: standard InfoNCE needs large in-batch negative
counts (1024-4096) that don't fit this machine's ~2.2 GB usable VRAM, and
gradient accumulation doesn't fix that (more gradient steps, not more
negatives). MoCo (He et al. 2019, "Momentum Contrast for Unsupervised
Visual Representation Learning") decouples negative count from batch size
entirely: a momentum-updated (EMA) copy of each encoder produces "key"
embeddings that get pushed onto a FIFO queue, so the queue can hold
thousands of negatives while the actual forward/backward batch stays tiny.

Topology: light curve is the contrastive ANCHOR. Three symmetric pairs
(lightcurve<->image, lightcurve<->spectrum, lightcurve<->catalog), NOT full
pairwise C(4,2)=6 -- an object needing two NON-anchor modalities
simultaneously (e.g. both a real image AND a real spectrum) is a rarer real
intersection than "has a light curve plus one other modality," and
light-curve ingestion is this codebase's base survey product. Each pair is
symmetric (two InfoNCE directions), because every modality needs its own
independently deployable encoder, not just three encoders' EMA shadows of
light curves.

Own Config/Report/training loop, not `train.TrainConfig`/`TrainReport`/
`train.train()` -- same reasoning `pretrain.py` already documents:
`train._loss_for`'s `(prediction, mu, logvar)` contract has no slot for 4
encoders, 6 loss directions, EMA updates, or queue bookkeeping. Reuses only
the genuinely kind-agnostic pieces `pretrain.py` already established as
reusable: `hardware.select_device`, `train.choose_batch_size`,
`train._cuda_oom_guard`, `train._set_seed`.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

KINDS: tuple[str, ...] = ("lightcurve", "image", "spectrum", "catalog")
PAIRS: tuple[tuple[str, str], ...] = (
    ("lightcurve", "image"), ("lightcurve", "spectrum"), ("lightcurve", "catalog"),
)


@dataclass
class MultimodalMoCoConfig:
    embedding_dim: int = 64
    projection_dim: int = 32
    queue_size: int = 2048
    momentum: float = 0.999
    temperature: float = 0.07
    image_input_size: int = 32
    spectrum_length: int = 256
    spectrum_patch_size: int = 16
    catalog_hidden: int = 128
    catalog_features: int = 40
    lc_length: int = 256
    lc_patch_size: int = 16
    lc_checkpoint: str | None = None
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
class MultimodalMoCoReport:
    device: str
    device_reason: str
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    train_losses: list[float] = field(default_factory=list)
    pair_losses: dict[str, list[float]] = field(default_factory=dict)
    val_losses: list[float] = field(default_factory=list)
    queue_fill: dict[str, int] = field(default_factory=dict)
    batch_size: int = 0
    accumulation_steps: int = 1
    parameters: dict[str, int] = field(default_factory=dict)
    amp_enabled: bool = False
    seconds: float = 0.0
    checkpoint: str | None = None
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def _make_kind_modules(kind: str, cfg: MultimodalMoCoConfig):
    from torch import nn

    from . import multimodal_encoders as enc

    if kind == "lightcurve":
        encoder = enc.make_lightcurve_encoder(
            embedding_dim=cfg.embedding_dim, length=cfg.lc_length,
            patch_size=cfg.lc_patch_size, checkpoint=cfg.lc_checkpoint)
    elif kind == "image":
        encoder = enc.make_image_encoder(embedding_dim=cfg.embedding_dim)
    elif kind == "spectrum":
        encoder = enc.make_spectrum_encoder(
            embedding_dim=cfg.embedding_dim, length=cfg.spectrum_length,
            patch_size=cfg.spectrum_patch_size)
    elif kind == "catalog":
        encoder = enc.make_catalog_encoder(
            embedding_dim=cfg.embedding_dim, n_features=cfg.catalog_features,
            hidden=cfg.catalog_hidden)
    else:
        raise ValueError(f"unknown modality kind: {kind!r}")

    return nn.ModuleDict({
        "encoder": encoder,
        "scale_token": enc.make_scale_token(cfg.embedding_dim),
        "fusion": enc.make_scale_fusion(cfg.embedding_dim),
        "projection": enc.make_projection_head(cfg.embedding_dim, cfg.projection_dim),
    })


def make_model(cfg: MultimodalMoCoConfig | None = None):
    """Four online (gradient-trained) modality modules plus their momentum
    (EMA, no-grad) copies, all under one `nn.Module` for checkpointing."""
    import torch
    from torch import nn

    cfg = cfg or MultimodalMoCoConfig()

    class MultimodalMoCoModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = cfg
            self.online = nn.ModuleDict({kind: _make_kind_modules(kind, cfg) for kind in KINDS})
            self.momentum = copy.deepcopy(self.online)
            for parameter in self.momentum.parameters():
                parameter.requires_grad_(False)

        def encode_online(self, kind: str, x, raw_scale):
            from . import multimodal_encoders as enc

            fused = enc.encode_and_fuse(self.online[kind], x, raw_scale)
            projected = self.online[kind]["projection"](fused)
            return fused, torch.nn.functional.normalize(projected, dim=-1)

        def encode_momentum(self, kind: str, x, raw_scale):
            from . import multimodal_encoders as enc

            with torch.no_grad():
                fused = enc.encode_and_fuse(self.momentum[kind], x, raw_scale)
                projected = self.momentum[kind]["projection"](fused)
                return fused, torch.nn.functional.normalize(projected, dim=-1)

    return MultimodalMoCoModel()


class EmbeddingQueue:
    """FIFO ring buffer of momentum-encoder keys, used as InfoNCE negatives.

    This is the mechanism that decouples negative count from batch size:
    `queue_size` (default 2048) can be raised freely later without touching
    the training loop's batch size or memory profile at all -- unlike a
    standard in-batch-negatives InfoNCE loss, where more negatives means a
    literally larger forward/backward batch.
    """

    def __init__(self, dim: int, size: int, device=None) -> None:
        import torch

        self.size = size
        self.dim = dim
        self.buffer = torch.zeros(size, dim, device=device)
        self.pointer = 0
        self.filled = 0

    def enqueue(self, keys) -> None:
        keys = keys.detach()
        n = keys.shape[0]
        if n == 0:
            return
        if n >= self.size:
            self.buffer = keys[-self.size:].clone()
            self.pointer = 0
            self.filled = self.size
            return
        end = self.pointer + n
        if end <= self.size:
            self.buffer[self.pointer:end] = keys
        else:
            first = self.size - self.pointer
            self.buffer[self.pointer:] = keys[:first]
            self.buffer[:end - self.size] = keys[first:]
        self.pointer = end % self.size
        self.filled = min(self.size, self.filled + n)

    def negatives(self):
        # A CLONE, not a view: `self.buffer` is mutated in place by later
        # `enqueue()` calls within the same training step (every pair reads
        # one queue's negatives, then immediately enqueues into it before
        # the next pair runs) -- a view would corrupt the autograd graph's
        # saved tensor between forward and backward, raising "modified by
        # an inplace operation" once a later enqueue bumped the buffer's
        # version counter.
        return self.buffer[:self.filled].clone()


def update_momentum(online, momentum, m: float) -> None:
    """In-place EMA update: `momentum <- m*momentum + (1-m)*online`."""
    import torch

    with torch.no_grad():
        for p_online, p_momentum in zip(online.parameters(), momentum.parameters()):
            p_momentum.mul_(m).add_(p_online, alpha=1.0 - m)


def info_nce_loss(query, positive_key, queue_negatives, temperature: float = 0.07):
    """Standard He et al. 2019 InfoNCE: logits = [q.k+, q.k-_1 ... q.k-_K] /
    temperature, cross-entropy against label 0 (the positive is always
    placed first).

    Computed in float32 regardless of the caller's autocast state -- same
    "numerically sensitive step forced to float32 under AMP" discipline
    `models.ConvVAE.reparameterise` already applies for the same reason
    (this card is Turing, sm_75, with no bfloat16; float16 is the only
    mixed-precision option and overflows readily). Measured directly: with
    this cast, `dividing by a 0.07 temperature then exponentiating inside
    cross_entropy` produced NaN losses within the first training epoch on
    this machine's GPU; float32 here removes it without giving up AMP's
    memory benefit on the heavier conv/transformer forward passes.
    """
    import torch
    from torch.nn import functional as F

    query = query.float()
    positive_key = positive_key.float()
    queue_negatives = queue_negatives.float()

    positive_logit = (query * positive_key).sum(dim=-1, keepdim=True)
    if queue_negatives.shape[0] > 0:
        negative_logits = query @ queue_negatives.t()
        logits = torch.cat([positive_logit, negative_logits], dim=1)
    else:
        logits = positive_logit
    logits = logits / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def kind_batch(batch, kind: str):
    """(values_array, scale_array) for one modality from a
    `multimodal_synthetic.SyntheticMultimodalBatch`-shaped object (duck
    typed: any object with `{kind}_values`/`{kind}_scale`,
    `{kind}_arrays`/`{kind}_scale`, or `{kind}_features`/`{kind}_scale`
    attributes)."""
    for values_attr in (f"{kind}_values", f"{kind}_arrays", f"{kind}_features"):
        if hasattr(batch, values_attr):
            return getattr(batch, values_attr), getattr(batch, f"{kind}_scale")
    raise AttributeError(f"batch has no data for modality {kind!r}")


def train_moco(train_batch, val_batch, cfg: MultimodalMoCoConfig | None = None,
               checkpoint_dir: Path | None = None,
               name: str = "multimodal_moco") -> MultimodalMoCoReport:
    """Train all four branches jointly against the light-curve anchor.

    `train.choose_batch_size`'s per-row-byte model is calibrated for a
    single 1-D sequence and is a poor fit for a 4-branch batch with very
    different per-modality memory profiles -- called once with
    `cfg.lc_length` as a representative length to get a starting candidate,
    then clamped to at most 256. This whole model is on the order of 250K
    parameters and well under 50 MB of activation memory at batch 64 (see
    `docs/DEFERRED.txt`'s entry for this backlog item), two orders of
    magnitude below the VRAM budget regardless -- the clamp is a
    conservative floor, not a load-bearing calculation.
    """
    import torch
    from . import config as config_mod, hardware
    from . import train as train_mod

    cfg = cfg or MultimodalMoCoConfig()
    train_mod._set_seed(cfg.seed)
    data_rng = np.random.default_rng(cfg.seed)

    device_report = hardware.select_device()
    device = torch.device(device_report.device)

    batch_size = min(train_mod.choose_batch_size(cfg.lc_length, cfg.batch_size), 256)
    accumulation = max(1, cfg.effective_batch_size // max(batch_size, 1))

    model = make_model(cfg).to(device)
    optimiser = torch.optim.Adam(
        (p for p in model.online.parameters() if p.requires_grad), lr=cfg.learning_rate)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    queues = {kind: EmbeddingQueue(cfg.projection_dim, cfg.queue_size, device=device)
             for kind in KINDS}

    from . import models as models_mod

    report = MultimodalMoCoReport(
        device=device_report.device, device_reason=device_report.reason,
        epochs_run=0, best_epoch=-1, best_val_loss=float("inf"),
        pair_losses={f"{a}_{b}": [] for a, b in PAIRS},
        batch_size=batch_size, accumulation_steps=accumulation,
        parameters={kind: models_mod.count_parameters(model.online[kind]) for kind in KINDS},
        amp_enabled=use_amp, seed=cfg.seed,
    )

    checkpoint_dir = checkpoint_dir or config_mod.PATHS.models
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_multimodal_moco.pt"

    started = time.time()
    since_improvement = 0
    n = len(train_batch.object_ids)

    def _to_device(values, scale):
        # `scale` is the RAW physical-unit scalar (flux/magnitude); every
        # scale-token module expects signed_log_scale-transformed input
        # (see multimodal_encoders.encode_and_fuse's docstring).
        from . import multimodal_encoders as enc

        transformed_scale = enc.signed_log_scale(np.asarray(scale, dtype=np.float64))
        return (torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device),
               torch.from_numpy(transformed_scale.astype(np.float32)).to(device))

    def _step_losses(batch, idx):
        total = torch.zeros((), device=device)
        epoch_pair_losses: dict[str, float] = {}
        lc_values, lc_scale = _to_device(
            *[arr[idx] for arr in kind_batch(batch, "lightcurve")])
        for anchor_kind, other_kind in PAIRS:
            other_values, other_scale = _to_device(
                *[arr[idx] for arr in kind_batch(batch, other_kind)])

            _, lc_query = model.encode_online("lightcurve", lc_values, lc_scale)
            _, lc_key = model.encode_momentum("lightcurve", lc_values, lc_scale)
            _, other_query = model.encode_online(other_kind, other_values, other_scale)
            _, other_key = model.encode_momentum(other_kind, other_values, other_scale)

            forward_loss = info_nce_loss(
                lc_query, other_key, queues[other_kind].negatives(), cfg.temperature)
            backward_loss = info_nce_loss(
                other_query, lc_key, queues["lightcurve"].negatives(), cfg.temperature)
            pair_loss = forward_loss + backward_loss
            total = total + pair_loss
            epoch_pair_losses[f"{anchor_kind}_{other_kind}"] = float(pair_loss.detach())

            queues[other_kind].enqueue(other_key)
            queues["lightcurve"].enqueue(lc_key)

        return total, epoch_pair_losses

    for epoch in range(cfg.epochs):
        model.train()
        order = data_rng.permutation(n) if n else np.empty(0, dtype=int)
        running_total = 0.0
        running_pairs = {f"{a}_{b}": 0.0 for a, b in PAIRS}
        steps = 0

        with train_mod._cuda_oom_guard():
            optimiser.zero_grad(set_to_none=True)
            for step, start in enumerate(range(0, n, max(batch_size, 1))):
                idx = order[start:start + batch_size]
                with torch.amp.autocast("cuda", enabled=use_amp):
                    total_loss, pair_losses = _step_losses(train_batch, idx)
                scaler.scale(total_loss / accumulation).backward()

                if (step + 1) % accumulation == 0:
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad(set_to_none=True)
                    for kind in KINDS:
                        update_momentum(model.online[kind], model.momentum[kind], cfg.momentum)

                running_total += float(total_loss.detach())
                for key, value in pair_losses.items():
                    running_pairs[key] += value
                steps += 1

            if steps and steps % accumulation != 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                for kind in KINDS:
                    update_momentum(model.online[kind], model.momentum[kind], cfg.momentum)

        report.train_losses.append(running_total / max(steps, 1))
        for key in running_pairs:
            report.pair_losses[key].append(running_pairs[key] / max(steps, 1))
        report.queue_fill = {kind: queues[kind].filled for kind in KINDS}

        model.eval()
        with torch.no_grad():
            val_idx = np.arange(len(val_batch.object_ids))
            val_loss, _ = _step_losses(val_batch, val_idx) if len(val_idx) else (
                torch.tensor(float("nan")), {})
        val_loss_value = float(val_loss.detach())
        report.val_losses.append(val_loss_value)
        report.epochs_run = epoch + 1

        if val_loss_value < report.best_val_loss - 1e-6:
            report.best_val_loss = val_loss_value
            report.best_epoch = epoch
            since_improvement = 0
            torch.save({"online_state_dict": model.online.state_dict(),
                       "momentum_state_dict": model.momentum.state_dict(),
                       "config": cfg.to_dict(), "epoch": epoch,
                       "val_loss": val_loss_value}, checkpoint_path)
            report.checkpoint = str(checkpoint_path)
        else:
            since_improvement += 1
            if since_improvement >= cfg.patience:
                break

    report.seconds = round(time.time() - started, 2)
    return report


def load_multimodal_moco(checkpoint_path: str | Path):
    """Restore a trained model (online + momentum weights) and its config."""
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = MultimodalMoCoConfig(**payload["config"])
    model = make_model(cfg)
    model.online.load_state_dict(payload["online_state_dict"])
    model.momentum.load_state_dict(payload["momentum_state_dict"])
    model.eval()
    return model, payload["config"]
