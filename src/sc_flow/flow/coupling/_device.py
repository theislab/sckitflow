
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
    import jax

    return jax.dlpack.from_dlpack(t.detach().contiguous())


def jax_to_torch(a: Any) -> torch.Tensor:
    return torch.from_dlpack(a)


# The whole solve (sinkhorn/GW) + sample_joint runs under jax.jit. Un-jitted, ott dispatches every
# fixed-point iteration op-by-op in eager jax — MEASURED ~390ms/step (H100, 1024²), 99% of the training
# step. Jitting compiles the solve into one XLA program: ~1.6ms, a ~245x coupling speedup. cellflow does
# the same (`self.match_fn = jax.jit(match_fn)`). We memoize one jitted callable per *static* config
# (epsilon/threshold/…) so jax.jit's shape-cache is reused across steps; source & target batch sizes are
# fixed, so every step is a cache hit (a new batch shape simply triggers one recompile for that shape).


@functools.cache
def _linear_coupler(epsilon: float, scale_cost: Any, tau_a: float, tau_b: float, threshold: float, extra: tuple):
    import jax
    from ott.geometry import costs, pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn
    from ott.solvers.utils import sample_joint

    extra_kwargs = dict(extra)

    @jax.jit
    def _fn(src: Any, tgt: Any, key: Any):
        geom = pointcloud.PointCloud(src, tgt, cost_fn=costs.SqEuclidean(), epsilon=epsilon, scale_cost=scale_cost)
        problem = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
        tmat = sinkhorn.Sinkhorn(threshold=threshold, **extra_kwargs)(problem).matrix
        return sample_joint(key, tmat)

    return _fn


@functools.cache
def _quadratic_coupler(scale_cost: Any, cost_fn: Any, fused: bool):
    import jax
    from ott.solvers.utils import match_quadratic, sample_joint

    if fused:
        @jax.jit
        def _fn(src_quad: Any, tgt_quad: Any, src_lin: Any, tgt_lin: Any, key: Any):
            tmat = match_quadratic(xx=src_quad, yy=tgt_quad, x=src_lin, y=tgt_lin,
                                   scale_cost=scale_cost, cost_fn=cost_fn)
            return sample_joint(key, tmat)
    else:
        @jax.jit
        def _fn(src_quad: Any, tgt_quad: Any, key: Any):
            tmat = match_quadratic(xx=src_quad, yy=tgt_quad, scale_cost=scale_cost, cost_fn=cost_fn)
            return sample_joint(key, tmat)

    return _fn


def _make_hashable(val: Any) -> Any:
    if isinstance(val, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in val.items()))
    if isinstance(val, (list, set)):
        return tuple(_make_hashable(x) for x in val)
    return val


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
    mk = dict(match_kwargs or {})
    src_j, tgt_j = torch_to_jax(src_rep), torch_to_jax(tgt_rep)
    if quad:
        scale_cost = mk.pop("scale_cost", "mean")
        cost_fn = mk.pop("cost_fn", None)
        fused = src_lin is not None and tgt_lin is not None
        coupler = _quadratic_coupler(scale_cost, cost_fn, fused)
        if fused:
            src_ixs_j, tgt_ixs_j = coupler(src_j, tgt_j, torch_to_jax(src_lin), torch_to_jax(tgt_lin), key)
        else:
            src_ixs_j, tgt_ixs_j = coupler(src_j, tgt_j, key)
    else:
        # Resolve the same defaults the eager path used; unbalanced (tau<1) loosens the convergence threshold.
        tau_a, tau_b = float(mk.pop("tau_a", 1.0)), float(mk.pop("tau_b", 1.0))
        epsilon = float(mk.pop("epsilon", 1.0))
        scale_cost = mk.pop("scale_cost", "mean")
        threshold = mk.pop("threshold", None)
        if threshold is None:
            threshold = 1e-3 if (tau_a == 1.0 and tau_b == 1.0) else 1e-2
        # Any remaining match_kwargs are further Sinkhorn solver args (max_iterations, lse_mode, …): forward
        # them + key the cache on them, so they are honored — not silently dropped (must be hashable).
        extra = tuple(sorted((k, _make_hashable(v)) for k, v in mk.items()))
        coupler = _linear_coupler(epsilon, scale_cost, tau_a, tau_b, float(threshold), extra)
        src_ixs_j, tgt_ixs_j = coupler(src_j, tgt_j, key)
    return jax_to_torch(src_ixs_j).long(), jax_to_torch(tgt_ixs_j).long()
