"""Shared helpers used by handlers across more than one domain module.

`rpc.py` grew to 156 top-level functions before this split; most private
helpers turned out to be domain-local (moved alongside the handlers that use
them), but `_workspace_root` is called from nearly every handler regardless
of domain, and `_require_torch`/`DEEP_UNAVAILABLE` are shared by the two
domains that can attempt a PyTorch-backed operation (digital-twin transfer
scoring and the deep-model handlers). Keeping this common module tiny and
having every domain module depend only on it (never on each other) avoids
recreating the tangle this split was meant to remove.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

# `astra.crossmatch` and `astra.surveys` (via surveys/des.py, which imports
# `angular_separation_arcsec` back from crossmatch) form a genuine import
# cycle. It is silent as long as something fully imports `astra.surveys`
# before anything reaches `astra.crossmatch` -- rpc.py's old one-line
# `from . import (ablation, acquire, ...)` happened to do that because
# `acquire` (which imports `surveys` directly) preceded `crossmatch`/
# `candidates` in that tuple. Splitting rpc.py into independent domain
# modules removed that accidental ordering, so it's pinned explicitly here
# instead: every domain module imports `common` first, so priming `surveys`
# in this shared module before any domain module's own imports keeps the
# cycle from ever being hit first by `crossmatch`.
from .. import surveys as _surveys  # noqa: F401
from .. import project as project_mod

Handler = Callable[[dict[str, Any]], Any]

PROTOCOL_VERSION = 1


def _workspace_root(project_id: object) -> Path | None:
    """Resolve an optional project workspace without weakening path checks."""
    if project_id is None or str(project_id).strip() == "":
        return None
    return project_mod.project_dir(str(project_id))


DEEP_UNAVAILABLE = (
    "PyTorch is not available in this build, so deep models cannot run. "
    "Released ASTRA installers ship a CPU-only engine that deliberately "
    "excludes PyTorch and CUDA — they would add roughly 3.5 GB to the "
    "installer for a capability most sessions never use. Everything else "
    "(acquisition, features, baseline anomaly detection, cross-survey "
    "matching, ranking and export) works normally. To train deep models, "
    "run the engine from a development checkout with the 'gpu' extra "
    "installed: uv pip install -e engine[gpu]"
)


def _require_torch() -> None:
    """Fail with an explanation rather than a bare ModuleNotFoundError.

    A packaged build genuinely cannot do this, so the message has to say why
    and what to do instead. "No module named 'torch'" reads like a broken
    installation; this is a deliberate build choice.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(DEEP_UNAVAILABLE) from exc
