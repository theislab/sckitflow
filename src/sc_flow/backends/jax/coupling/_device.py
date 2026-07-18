"""Device-native minibatch OT coupling: torch reps ↔ JAX via zero-copy DLPack, ott on the same device.

The batch lives where the model lives (GPU under CUDA). This keeps the reps **and** the transport plan
on that device — torch tensor → JAX array via DLPack (zero-copy, same device) → ott sinkhorn / GW →
``sample_joint`` → indices → back to torch via DLPack. Only the tiny integer index arrays are produced;
no cell data or coupling matrix is copied to host. Keeps the proven ott kernel while operating on the
GPU batch (needs a CUDA ``jaxlib`` for the JAX side to be on the GPU too).
"""

from __future__ import annotations

import functools
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


# The whole solve (sinkhorn/GW) + sample_joint runs under jax.jit. Un-jitted, ott dispatches every
# fixed-point iteration op-by-op in eager jax — MEASURED ~390ms/step (H100, 1024²), 99% of the training
# step. Jitting compiles the solve into one XLA program: ~1.6ms, a ~245x coupling speedup. cellflow does
# the same (`self.match_fn = jax.jit(match_fn)`). We memoize one jitted callable per *static* config
# (epsilon/threshold/…) so jax.jit's shape-cache is reused across steps; source & target batch sizes are
# fixed, so every step is a cache hit (a new batch shape simply triggers one recompile for that shape).


@functools.lru_cache(maxsize=None)
def _linear_coupler(epsilon: float, scale_cost: Any, tau_a: float, tau_b: float, threshold: float):
    """Return a jitted ``(src, tgt, key) -> (src_ixs, tgt_ixs)`` linear-sinkhorn coupler for this config."""
    import jax

    from ott.geometry import costs, pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn
    from ott.solvers.utils import sample_joint

    @jax.jit
    def _fn(src: Any, tgt: Any, key: Any):
        geom = pointcloud.PointCloud(src, tgt, cost_fn=costs.SqEuclidean(), epsilon=epsilon, scale_cost=scale_cost)
        problem = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
        tmat = sinkhorn.Sinkhorn(threshold=threshold)(problem).matrix
        return sample_joint(key, tmat)

    return _fn


@functools.lru_cache(maxsize=None)
def _quadratic_coupler(scale_cost: Any, fused: bool):
    """Return a jitted quadratic/GW coupler; ``fused`` selects whether linear reps also condition the plan."""
    import jax

    from ott.solvers.utils import match_quadratic, sample_joint

    if fused:
        @jax.jit
        def _fn(src_quad: Any, tgt_quad: Any, src_lin: Any, tgt_lin: Any, key: Any):
            tmat = match_quadratic(xx=src_quad, yy=tgt_quad, x=src_lin, y=tgt_lin, scale_cost=scale_cost)
            return sample_joint(key, tmat)
    else:
        @jax.jit
        def _fn(src_quad: Any, tgt_quad: Any, key: Any):
            tmat = match_quadratic(xx=src_quad, yy=tgt_quad, scale_cost=scale_cost)
            return sample_joint(key, tmat)

    return _fn


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
    mk = dict(match_kwargs or {})
    src_j, tgt_j = torch_to_jax(src_rep), torch_to_jax(tgt_rep)
    if quad:
        scale_cost = mk.get("scale_cost", "mean")
        fused = src_lin is not None and tgt_lin is not None
        coupler = _quadratic_coupler(scale_cost, fused)
        if fused:
            src_ixs_j, tgt_ixs_j = coupler(src_j, tgt_j, torch_to_jax(src_lin), torch_to_jax(tgt_lin), key)
        else:
            src_ixs_j, tgt_ixs_j = coupler(src_j, tgt_j, key)
    else:
        # Resolve the same defaults the eager path used; unbalanced (tau<1) loosens the convergence threshold.
        tau_a, tau_b = float(mk.get("tau_a", 1.0)), float(mk.get("tau_b", 1.0))
        threshold = mk.get("threshold")
        if threshold is None:
            threshold = 1e-3 if (tau_a == 1.0 and tau_b == 1.0) else 1e-2
        coupler = _linear_coupler(
            float(mk.get("epsilon", 1.0)), mk.get("scale_cost", "mean"), tau_a, tau_b, float(threshold)
        )
        src_ixs_j, tgt_ixs_j = coupler(src_j, tgt_j, key)
    return jax_to_torch(src_ixs_j).long(), jax_to_torch(tgt_ixs_j).long()
