"""Adapt ``binded.Loader`` batches to the CellFlow flow-matching objective's expected keys.

``binded`` yields ``{source, target, condition, source_reps, target_reps}`` (see
:class:`binded.Loader`). :class:`~sc_flow.backends.torch.jaxbridge._objective.CellFlowFMObjective`
(and :class:`~sc_flow.backends.torch.jaxbridge._cellflow.CellFlowJaxModule`) instead read
``{time, source, target, encoder_noise, conditions}``. The gap is purely mechanical and lives here so
neither the loader nor the objective has to know about the other:

* rename ``condition`` (binded) → ``conditions`` (objective);
* draw the per-step ``time`` ~ U[0,1) and ``encoder_noise`` ~ N(0, I) the FM loss samples each step
  (binded is data-only and emits neither);
* hand ``source`` / ``target`` as ``float32`` torch tensors (the objective mirrors them into JAX via
  the DLPack bridge); condition arrays stay as-is (the objective forwards non-torch values to JAX).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import torch

__all__ = ["adapt_fm_batch", "iter_fm_batches"]


def _to_torch(x: Any) -> torch.Tensor:
    """A ``float32`` torch view of a binded rep (torch tensor, numpy, or jax array)."""
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def adapt_fm_batch(
    batch: dict[str, Any],
    *,
    condition_embedding_dim: int,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Map one ``binded`` batch to a CellFlow-FM objective batch (adds ``time`` + ``encoder_noise``)."""
    source = _to_torch(batch["source"])
    target = _to_torch(batch["target"])
    n = target.shape[0]
    out: dict[str, Any] = {
        "source": source,
        "target": target,
        "time": torch.rand(n, 1, generator=generator),
        "encoder_noise": torch.randn(n, condition_embedding_dim, generator=generator),
    }
    condition = batch.get("condition")
    if condition is not None:
        # kept as float32 arrays; the objective forwards non-torch values straight into the JAX loss.
        out["conditions"] = {k: np.asarray(v, dtype=np.float32) for k, v in condition.items()}
    return out


def iter_fm_batches(
    loader: Iterable[dict[str, Any]],
    *,
    condition_embedding_dim: int,
    seed: int = 0,
) -> Iterator[dict[str, Any]]:
    """Wrap a ``binded.Loader`` so each yielded batch is FM-objective-shaped (see :func:`adapt_fm_batch`)."""
    generator = torch.Generator().manual_seed(seed)
    for batch in loader:
        yield adapt_fm_batch(batch, condition_embedding_dim=condition_embedding_dim, generator=generator)
