import logging
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.utils import match_quadratic

logger = logging.getLogger(__name__)


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


def ot_linear_coupling(
    source: jax.Array,
    target: jax.Array,
    cost_fn: Callable | None = None,
    scale_cost: float | Literal["mean", "max_cost", "median"] = "mean",
    method: Literal["exact", "sinkhorn", "partial", "unbalanced"] = "sinkhorn",
    of_fn: Callable | None = None,
    reg: float = 5e-1,
    reg_m: float = 1.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, jax.Array]:
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
    if "return_matrix" in kwargs and kwargs["return_matrix"]:
        return source_idxs, target_idxs, coupling_matrix
    return source_idxs, target_idxs


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
    of_fn: Callable | None = None,
    reg: float = 5e-1,
    reg_m: float = 1.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, jax.Array]:
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
    if isinstance(src_xx_cell_coupling, np.ndarray):
        src_xx_cell_coupling = jnp.array(src_xx_cell_coupling)
    if isinstance(tgt_yy_cell_coupling, np.ndarray):
        tgt_yy_cell_coupling = jnp.array(tgt_yy_cell_coupling)
    if isinstance(src_xy_cell_coupling, np.ndarray):
        src_xy_cell_coupling = jnp.array(src_xy_cell_coupling)
    if isinstance(tgt_xy_cell_coupling, np.ndarray):
        tgt_xy_cell_coupling = jnp.array(tgt_xy_cell_coupling)

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
    if not jnp.all(np.isfinite(coupling_matrix)):
        msg = f"Non finite values found in `coupling_matrix` \n {coupling_matrix=} \n {src_xx_cell_coupling=} \n {tgt_yy_cell_coupling=}"
        logger.warning(msg)
    if jnp.abs(coupling_matrix.sum()) < 1e-8:
        msg = ""
        logger.warning(msg)
        coupling_matrix = jnp.ones_like(coupling_matrix) / coupling_matrix.size
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
    if "return_matrix" in kwargs and kwargs["return_matrix"]:
        return source_idxs, target_idxs, coupling_matrix
    return source_idxs, target_idxs
