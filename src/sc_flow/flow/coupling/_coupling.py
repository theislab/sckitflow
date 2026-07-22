import logging
from typing import Literal, overload

import numpy as np
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.utils import match_quadratic

from sc_flow.flow.coupling._types import (
    ArrayLike,
    LinCouplingMethod,
    NumpyArray,
    OTFn,
    QuadCouplingMethod,
    ScaleMethod,
)
from sc_flow.flow.coupling._utils import to_jax_array

logger = logging.getLogger(__name__)


def _select_indices(
    coupling_matrix: NumpyArray,
    size: int,
    rng: np.random.Generator | None = None,
) -> tuple[NumpyArray, NumpyArray]:
    # checking for numerical errors in the coupling matrix
    coupling_matrix = _sanitize_coupling_matrix(coupling_matrix=coupling_matrix)

    # retrieving coupling probabilities
    coupling_probs = coupling_matrix.flatten()
    coupling_probs = coupling_probs / coupling_probs.sum()
    # sampling indices (seeded Generator when provided, else global state)
    chooser = np.random if rng is None else rng
    choices = chooser.choice(
        coupling_matrix.shape[0] * coupling_matrix.shape[1],
        p=coupling_probs,
        size=size,
        replace=False,
    )
    source_idxs, target_idxs = np.divmod(choices, coupling_matrix.shape[1])
    return source_idxs, target_idxs


def _sanitize_coupling_matrix(
    coupling_matrix: NumpyArray,
    eps: float = 1e-8,
) -> NumpyArray:
    # Non-finite check (use NumPy for robustness with logging)
    if not np.all(np.isfinite(coupling_matrix)):
        msg = f"Non-finite values found in `coupling_matrix` {coupling_matrix=}\n"
        logger.warning(msg)

    # Degenerate sum → fallback to uniform
    if np.abs(coupling_matrix.sum()) < eps:
        logger.warning("Sum of `coupling_matrix` is numerically zero; replacing with a uniform coupling.")
        coupling_matrix = np.ones_like(coupling_matrix) / coupling_matrix.size

    return coupling_matrix


def independent_coupling(
    source: ArrayLike,
    target: ArrayLike,
    rng: np.random.Generator | None = None,
) -> tuple[NumpyArray, NumpyArray]:
    # randomly permuting the tensors (seeded Generator when provided, else global state)
    chooser = np.random if rng is None else rng
    src_random_perm_idx = chooser.choice(source.shape[0], size=source.shape[0], replace=False)
    tgt_random_perm_idx = chooser.choice(target.shape[0], size=source.shape[0], replace=False)
    min_shape = min(src_random_perm_idx.shape[0], tgt_random_perm_idx.shape[0])

    return src_random_perm_idx[:min_shape], tgt_random_perm_idx[:min_shape]


@overload
def ot_linear_coupling(
    source: ArrayLike,
    target: ArrayLike,
    return_matrix: Literal[False],
    cost_fn: costs.CostFn | None = ...,
    scale_cost: ScaleMethod = ...,
    method: LinCouplingMethod = ...,
    ot_fn: OTFn | None = ...,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray]: ...


@overload
def ot_linear_coupling(
    source: ArrayLike,
    target: ArrayLike,
    return_matrix: Literal[True],
    cost_fn: costs.CostFn | None = ...,
    scale_cost: ScaleMethod = ...,
    method: LinCouplingMethod = ...,
    ot_fn: OTFn | None = ...,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray, NumpyArray]: ...


def ot_linear_coupling(
    source: ArrayLike,
    target: ArrayLike,
    return_matrix: bool = False,
    cost_fn: costs.CostFn | None = None,
    scale_cost: ScaleMethod = "mean",
    method: LinCouplingMethod = "sinkhorn",
    ot_fn: OTFn | None = None,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray] | tuple[NumpyArray, NumpyArray, NumpyArray]:
    source = to_jax_array(source)
    target = to_jax_array(target)

    cost_fn = cost_fn or costs.SqEuclidean()

    scale_cost = scale_cost or "mean"

    geom = pointcloud.PointCloud(
        source,
        target,
        cost_fn=cost_fn,
        epsilon=1.0,
        scale_cost=scale_cost,
    )

    threshold = kwargs.get("threshold", 1e-3)

    is_unbalanced = method == "unbalanced"
    tau_a = kwargs.get("tau_a", 0.9 if is_unbalanced else 1.0)
    tau_b = kwargs.get("tau_b", 0.9 if is_unbalanced else 1.0)

    if method in ["sinkhorn", "unbalanced"]:
        ot_fn = sinkhorn.Sinkhorn(threshold=threshold)
    elif (method == "partial") | (method == "exact"):
        msg = f"{method=} has to equivalent in `ott-ajax`"
        raise ValueError(msg)
    elif method in ["exact", "sinkhorn", "partial", "unbalanced"] and ot_fn is None:
        msg = f"{method=} is not found, please specify a custom `method` in `ot_fn`"
        raise ValueError(msg)

    problem = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
    # computing coupling matrix
    coupling_matrix = np.asarray(ot_fn(problem).matrix)

    source_idxs, target_idxs = _select_indices(coupling_matrix=coupling_matrix, size=source.shape[0], rng=rng)
    if return_matrix:
        return source_idxs, target_idxs, coupling_matrix
    return source_idxs, target_idxs


@overload
def ot_quadratic_coupling(
    source_quad: ArrayLike,
    target_quad: ArrayLike,
    return_matrix: Literal[False],
    source_lin: ArrayLike | None = ...,
    target_lin: ArrayLike | None = ...,
    cost_fn: costs.CostFn | None = None,
    scale_cost: ScaleMethod = ...,
    method: QuadCouplingMethod = ...,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray]: ...


@overload
def ot_quadratic_coupling(
    source_quad: ArrayLike,
    target_quad: ArrayLike,
    return_matrix: Literal[True],
    source_lin: ArrayLike | None = ...,
    target_lin: ArrayLike | None = ...,
    cost_fn: costs.CostFn | None = None,
    scale_cost: ScaleMethod = ...,
    method: QuadCouplingMethod = ...,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray, NumpyArray]: ...


def ot_quadratic_coupling(
    source_quad: ArrayLike,
    target_quad: ArrayLike,
    source_lin: ArrayLike | None = None,
    target_lin: ArrayLike | None = None,
    return_matrix: bool = False,
    cost_fn: costs.CostFn | None = None,
    scale_cost: ScaleMethod = 1.0,
    method: QuadCouplingMethod = "entropic_gromov_wasserstein",
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray] | tuple[NumpyArray, NumpyArray, NumpyArray]:
    scale_cost = scale_cost or "mean"

    # moving arrays to jax arrays
    source_quad = to_jax_array(source_quad)
    target_quad = to_jax_array(target_quad)

    source_lin = to_jax_array(source_lin)
    target_lin = to_jax_array(target_lin)

    if method not in ["entropic_gromov_wasserstein", "entropic_fused_gromov_wasserstein"]:
        msg = f"{method=} is not found, please specify a custom `method` in `ot_fn`"
        raise ValueError(msg)
    elif method == "entropic_fused_gromov_wasserstein" and source_lin is None:
        msg = f"{method=} requires fused terms"
        raise ValueError(msg)

    # computing coupling matrix
    coupling_matrix = np.asarray(
        match_quadratic(
            xx=source_quad,
            yy=target_quad,
            x=source_lin,
            y=target_lin,
            scale_cost=scale_cost,
            cost_fn=cost_fn,
        )
    )

    source_idxs, target_idxs = _select_indices(coupling_matrix=coupling_matrix, size=source_quad.shape[0], rng=rng)
    if return_matrix:
        return source_idxs, target_idxs, coupling_matrix
    return source_idxs, target_idxs
