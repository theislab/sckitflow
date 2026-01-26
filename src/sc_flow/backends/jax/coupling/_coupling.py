import logging
from collections.abc import Callable
from typing import Literal, Protocol, overload

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.utils import match_quadratic

logger = logging.getLogger(__name__)

CostFn = Callable[[jax.Array, jax.Array], jax.Array]


class OTResult(Protocol):
    matrix: jax.Array


class OTFn(Protocol):
    def __call__(self, problem: "linear_problem.LinearProblem") -> OTResult: ...


def ensure_jax_array(x: jax.Array | np.ndarray) -> jax.Array:
    """Convert a NumPy array to a JAX array if needed."""
    if isinstance(x, np.ndarray):
        return jnp.array(x)
    return x


def sanitize_coupling_matrix(
    coupling_matrix: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Checks a coupling matrix for numerical issues and applies safe fallbacks.

    :param coupling_matrix: Coupling matrix to be checked for numerical stability.
    :type coupling_matrix: class:`jax.numpy.ndarray`

    :param eps: Threshold below which the sum of the coupling matrix is considered
        numerically zero.
    :type eps: float

    Returns
    -------
    jax.numpy.ndarray
        A numerically stable coupling matrix.
    """
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
    source: jax.Array,
    target: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`torch.Tensor`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`torch.Tensor`

    Returns
    -------
    ## TODO
    """
    # randomy permuting the tensors
    src_random_perm_idx = np.random.choice(np.arange(source.shape[0]), size=source.shape[0], replace=False)
    tgt_random_perm_idx = np.random.choice(np.arange(target.shape[0]), size=source.shape[0], replace=False)
    min_shape = min(src_random_perm_idx.shape[0], tgt_random_perm_idx.shape[0])

    return src_random_perm_idx[:min_shape], tgt_random_perm_idx[:min_shape]


@overload
def ot_linear_coupling(
    source: jax.Array,
    target: jax.Array,
    cost_fn: CostFn | None = ...,
    scale_cost: float | Literal["mean", "max_cost", "median"] = ...,
    method: Literal["exact", "sinkhorn", "partial", "unbalanced"] = ...,
    ot_fn: OTFn | None = ...,
    reg: float = ...,
    reg_m: float = ...,
    return_matrix: bool = False,
    **kwargs,
) -> "tuple[np.ndarray, np.ndarray]": ...


@overload
def ot_linear_coupling(
    source: jax.Array,
    target: jax.Array,
    cost_fn: CostFn | None = ...,
    scale_cost: float | Literal["mean", "max_cost", "median"] = ...,
    method: Literal["exact", "sinkhorn", "partial", "unbalanced"] = ...,
    ot_fn: OTFn | None = ...,
    reg: float = ...,
    reg_m: float = ...,
    return_matrix: bool = True,
    **kwargs,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]": ...


def ot_linear_coupling(
    source: jax.Array,
    target: jax.Array,
    cost_fn: CostFn | None = None,
    scale_cost: float | Literal["mean", "max_cost", "median"] = "mean",
    method: Literal["exact", "sinkhorn", "partial", "unbalanced"] = "sinkhorn",
    ot_fn: OTFn | None = None,
    reg: float = 5e-1,
    reg_m: float = 1.0,
    return_matrix: bool = False,
    **kwargs,
) -> "tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`torch.Tensor`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`torch.Tensor`

    #TODO
    """
    source = jnp.array(source)
    target = jnp.array(target)

    if cost_fn is None:
        cost_fn = costs.SqEuclidean()

    if scale_cost is None:
        scale_cost = "mean"

    geom = pointcloud.PointCloud(
        source,
        target,
        cost_fn=cost_fn,
        epsilon=1.0,
        scale_cost=scale_cost,
    )

    if "threshold" not in kwargs.keys():
        threshold = 1e-3
    else:
        threshold = kwargs["threshold"]
    if "tau_a" not in kwargs.keys() and method != "unbalanced":
        tau_a = 1.0
    elif "tau_a" not in kwargs.keys() and method == "unbalanced":
        tau_a = 0.9
    else:
        tau_a = kwargs["tau_a"]
    if "tau_b" not in kwargs.keys() and method != "unbalanced":
        tau_b = 1.0
    elif "tau_b" not in kwargs.keys() and method == "unbalanced":
        tau_b = 0.9
    else:
        tau_b = kwargs["tau_b"]

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

    # checking for numerical errors in the coupling matrix
    if not jnp.all(jnp.isfinite(coupling_matrix)):
        msg = f"Non finite values found in `coupling_matrix` \n {coupling_matrix=} \n {source=} \n {target=}"
        logger.warning(msg)
    if jnp.abs(coupling_matrix.sum()) < 1e-8:
        msg = ""
        logger.warning(msg)
        coupling_matrix = jnp.ones_like(coupling_matrix) / coupling_matrix.size

    coupling_matrix = sanitize_coupling_matrix(coupling_matrix=coupling_matrix)
    # retrieving coupling probabilities
    coupling_probs = coupling_matrix.flatten()
    coupling_probs = coupling_probs / coupling_probs.sum()

    choices = np.random.choice(
        coupling_matrix.shape[0] * coupling_matrix.shape[1],
        p=coupling_probs,
        size=source.shape[0],
        replace=False,
    )
    source_idxs, target_idxs = np.divmod(choices, coupling_matrix.shape[1])
    if return_matrix:
        return source_idxs, target_idxs, coupling_matrix
    return source_idxs, target_idxs


@overload
def ot_quadratic_coupling(
    src_xx_cell_coupling: jax.Array,
    tgt_yy_cell_coupling: jax.Array,
    src_xy_cell_coupling: jax.Array | None = ...,
    tgt_xy_cell_coupling: jax.Array | None = ...,
    cost_fn: Callable | None = None,
    scale_cost: float | Literal["mean", "max_cost", "median"] = ...,
    method: Literal[
        "entropic_gromov_wasserstein",
        "entropic_fused_gromov_wasserstein",
    ] = ...,
    ot_fn: Callable | None = ...,
    reg: float = ...,
    reg_m: float = ...,
    return_matrix: bool = False,
    **kwargs,
) -> "tuple[np.ndarray, np.ndarray]": ...


@overload
def ot_quadratic_coupling(
    src_xx_cell_coupling: jax.Array,
    tgt_yy_cell_coupling: jax.Array,
    src_xy_cell_coupling: jax.Array | None = ...,
    tgt_xy_cell_coupling: jax.Array | None = ...,
    cost_fn: Callable | None = None,
    scale_cost: float | Literal["mean", "max_cost", "median"] = ...,
    method: Literal[
        "entropic_gromov_wasserstein",
        "entropic_fused_gromov_wasserstein",
    ] = ...,
    ot_fn: Callable | None = ...,
    reg: float = ...,
    reg_m: float = ...,
    return_matrix: bool = True,
    **kwargs,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]": ...


def ot_quadratic_coupling(
    src_xx_cell_coupling: jax.Array,
    tgt_yy_cell_coupling: jax.Array,
    src_xy_cell_coupling: jax.Array | None = None,
    tgt_xy_cell_coupling: jax.Array | None = None,
    cost_fn: Callable | None = None,
    scale_cost: float | Literal["mean", "max_cost", "median"] = 1.0,
    method: Literal[
        "entropic_gromov_wasserstein",
        "entropic_fused_gromov_wasserstein",
    ] = "entropic_gromov_wasserstein",
    ot_fn: Callable | None = None,
    reg: float = 5e-1,
    reg_m: float = 1.0,
    return_matrix: bool = False,
    **kwargs,
) -> "tuple[np.ndarray, np.ndarray]" | "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`torch.Tensor`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`torch.Tensor`

    #TODO
    """
    if cost_fn is None:
        cost_fn = lambda source, target: sp.spatial.distance.cdist(source, target) ** 2

    if scale_cost is None:
        scale_cost = "mean"

    # moving arrays to jax arrays
    src_xx_cell_coupling = ensure_jax_array(src_xx_cell_coupling)
    tgt_yy_cell_coupling = ensure_jax_array(tgt_yy_cell_coupling)
    src_xy_cell_coupling = ensure_jax_array(src_xy_cell_coupling)
    tgt_xy_cell_coupling = ensure_jax_array(tgt_xy_cell_coupling)

    if method not in ["entropic_gromov_wasserstein", "entropic_fused_gromov_wasserstein"]:
        msg = f"{method=} is not found, please specify a custom `method` in `ot_fn`"
        raise ValueError(msg)
    elif method == "entropic_fused_gromov_wasserstein" and src_xy_cell_coupling is None:
        msg = f"{method=} requires fused terms"
        raise ValueError(msg)

    # computing coupling matrix
    coupling_matrix = np.asarray(
        match_quadratic(
            xx=src_xx_cell_coupling,
            yy=tgt_yy_cell_coupling,
            x=src_xy_cell_coupling,
            y=tgt_xy_cell_coupling,
            scale_cost=scale_cost,
        )
    )

    # checking for numerical errors in the coupling matrix
    coupling_matrix = sanitize_coupling_matrix(coupling_matrix=coupling_matrix)

    # retrieving coupling probabilities
    coupling_probs = coupling_matrix.flatten()
    coupling_probs = coupling_probs / coupling_probs.sum()
    # sampling indices
    choices = np.random.choice(
        coupling_matrix.shape[0] * coupling_matrix.shape[1],
        p=coupling_probs,
        size=src_xx_cell_coupling.shape[0],
        replace=False,
    )
    source_idxs, target_idxs = np.divmod(choices, coupling_matrix.shape[1])
    if return_matrix:
        return source_idxs, target_idxs, coupling_matrix
    return source_idxs, target_idxs
