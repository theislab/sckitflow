"""Device-native minibatch OT coupling: torch reps ↔ JAX via zero-copy DLPack, ott on the same device.

The batch lives where the model lives (GPU under CUDA). This keeps the reps **and** the transport plan
on that device — torch tensor → JAX array via DLPack (zero-copy, same device) → ott sinkhorn / GW →
``sample_joint`` → indices → back to torch via DLPack. Only the tiny integer index arrays are produced;
no cell data or coupling matrix is copied to host. Keeps the proven ott kernel while operating on the
GPU batch (needs a CUDA ``jaxlib`` for the JAX side to be on the GPU too).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

# torch and jax share the GPU here (torch owns the model/optimizer; jax only does the small OT coupling).
# By default jax preallocates ~75% of VRAM on backend init, which would starve a large torch model — cap it
# to on-demand allocation. Set at import (this module loads just before jax's first GPU op) via setdefault
# so a user can still override. Harmless on CPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

__all__ = ["couple_device", "torch_to_jax", "jax_to_torch"]


def torch_to_jax(t: torch.Tensor) -> Any:
    """Zero-copy view of a torch tensor as a JAX array on the **same** device (DLPack)."""
    import jax

    return jax.dlpack.from_dlpack(t.detach().contiguous())


def jax_to_torch(a: Any) -> torch.Tensor:
    """Zero-copy view of a JAX array as a torch tensor on the **same** device (DLPack)."""
    return torch.from_dlpack(a)


def _linear_plan(src: Any, tgt: Any, *, epsilon: float = 1.0, scale_cost: Any = "mean",
                 tau_a: float = 1.0, tau_b: float = 1.0, threshold: float | None = None, **kwargs: Any) -> Any:
    from ott.geometry import costs, pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    if threshold is None:
        threshold = 1e-3 if (tau_a == 1.0 and tau_b == 1.0) else 1e-2
    geom = pointcloud.PointCloud(src, tgt, cost_fn=costs.SqEuclidean(), epsilon=epsilon, scale_cost=scale_cost)
    problem = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
    return sinkhorn.Sinkhorn(threshold=threshold, **kwargs)(problem).matrix


def _quadratic_plan(src_quad: Any, tgt_quad: Any, src_lin: Any = None, tgt_lin: Any = None,
                    *, scale_cost: Any = "mean", cost_fn: Any = None, **kwargs: Any) -> Any:
    from ott.solvers.utils import match_quadratic

    return match_quadratic(xx=src_quad, yy=tgt_quad, x=src_lin, y=tgt_lin, scale_cost=scale_cost, cost_fn=cost_fn)


def couple_device(
    src_rep: torch.Tensor,
    tgt_rep: torch.Tensor,
    *,
    key: Any,
    quad: bool = False,
    src_lin: torch.Tensor | None = None,
    tgt_lin: torch.Tensor | None = None,
    match_kwargs: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OT plan on torch reps (on any device) → ``(src_ixs, tgt_ixs)`` torch long tensors on the same device.

    Linear sinkhorn by default; quadratic/fused GW when ``quad`` (matching on ``src_rep``/``tgt_rep`` as the
    quadratic terms, fused with ``src_lin``/``tgt_lin`` when given). ``key`` is a JAX ``PRNGKey`` for the
    plan-sampling (deterministic given the seed).
    """
    from ott.solvers.utils import sample_joint

    mk = dict(match_kwargs or {})
    src_j, tgt_j = torch_to_jax(src_rep), torch_to_jax(tgt_rep)
    if quad:
        sl = torch_to_jax(src_lin) if src_lin is not None else None
        tl = torch_to_jax(tgt_lin) if tgt_lin is not None else None
        tmat = _quadratic_plan(src_j, tgt_j, sl, tl, **mk)
    else:
        tmat = _linear_plan(src_j, tgt_j, **mk)
    src_ixs_j, tgt_ixs_j = sample_joint(key, tmat)
    return jax_to_torch(src_ixs_j).long(), jax_to_torch(tgt_ixs_j).long()
