import logging
from typing import Literal, overload

import numpy as np
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.utils import match_quadratic

from sc_flow.backends.jax._types import (
    ArrayLike,
    LinCouplingMethod,
    NumpyArray,
    OTFn,
    QuadCouplingMethod,
    ScaleMethod,
)
from sc_flow.backends.jax._utils import to_jax_array

logger = logging.getLogger(__name__)


def _select_indices(coupling_matrix: NumpyArray, size: int) -> tuple[NumpyArray, NumpyArray]:
    """Samples matching indices from a coupling matrix.

    Draws index pairs from the provided coupling matrix according to the
    normalized coupling probabilities and returns the corresponding source
    and target indices.

    :param coupling_matrix: Optimal transport coupling matrix encoding
        matching probabilities between source and target samples.
    :type coupling_matrix: class:`numpy.ndarray`

    :param size: Number of index pairs to sample from the coupling matrix.
    :type size: int

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing:

        - An array of sampled source indices.
        - An array of sampled target indices.

        The two arrays have identical length equal to :param:`size`.
    """
    # checking for numerical errors in the coupling matrix
    coupling_matrix = _sanitize_coupling_matrix(coupling_matrix=coupling_matrix)

    # retrieving coupling probabilities
    coupling_probs = coupling_matrix.flatten()
    coupling_probs = coupling_probs / coupling_probs.sum()
    # sampling indices
    choices = np.random.choice(
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
    source: ArrayLike,
    target: ArrayLike,
) -> tuple[NumpyArray, NumpyArray]:
    """Randomly matches source and target samples independently.

    Generates a random permutation of source and target indices independently
    and returns paired indices without using any notion of distance or
    optimal transport. This serves as a baseline or control coupling.

    :param source: Array containing samples from the source distribution.
    :type source: class:`ArrayLike`

    :param target: Array containing samples from the target distribution.
    :type target: class:`ArrayLike`

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing:

        - Randomly permuted source indices.
        - Randomly permuted target indices.

        The length of both arrays is equal to
        ``min(len(source), len(target))``.
    """
    # randomy permuting the tensors
    src_random_perm_idx = np.random.choice(source.shape[0], size=source.shape[0], replace=False)
    tgt_random_perm_idx = np.random.choice(target.shape[0], size=source.shape[0], replace=False)
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
    **kwargs,
) -> tuple[NumpyArray, NumpyArray] | tuple[NumpyArray, NumpyArray, NumpyArray]:
    """Computes a linear optimal transport coupling between source and target samples.

    Solves a linear optimal transport problem between the source and target
    distributions using the specified method (e.g. Sinkhorn or unbalanced
    Sinkhorn), then samples index pairs from the resulting coupling matrix.

    :param source: Array containing samples from the source distribution.
    :type source: class:`ArrayLike`

    :param target: Array containing samples from the target distribution.
    :type target: class:`ArrayLike`

    :param return_matrix: Whether to return the full coupling matrix in
        addition to the sampled indices.
    :type return_matrix: bool

    :param cost_fn: Cost function used to compute pairwise distances between
        source and target samples. If ``None``, squared Euclidean distance is used.
    :type cost_fn: costs.CostFn | None

    :param scale_cost: Method used to rescale the cost matrix (e.g. ``"mean"``).
    :type scale_cost: ScaleMethod

    :param method: Optimal transport solver to use. Supported values include
        ``"sinkhorn"`` and ``"unbalanced"``.
    :type method: LinCouplingMethod

    :param ot_fn: Custom optimal transport solver. If provided, overrides
        the solver implied by :param:`method`.
    :type ot_fn: OTFn | None

    :param kwargs: Additional keyword arguments forwarded to the OT solver.
    :type kwargs: dict

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray] or tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        If ``return_matrix=False``, returns:

        - Source indices
        - Target indices

        If ``return_matrix=True``, additionally returns the coupling matrix.
    """
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

    source_idxs, target_idxs = _select_indices(coupling_matrix=coupling_matrix, size=source.shape[0])
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
    **kwargs,
) -> tuple[NumpyArray, NumpyArray] | tuple[NumpyArray, NumpyArray, NumpyArray]:
    """Computes a quadratic (Gromov-type) optimal transport coupling.

    Solves a quadratic optimal transport problem based on pairwise relational
    structures within the source and target domains. Optionally supports
    fused Gromov-Wasserstein when cross-domain costs are provided.

    :param source_quad: Source intra-domain structure (XX cost matrix
        or feature representation).
    :type source_quad: class:`ArrayLike`

    :param target_quad: Target intra-domain structure (YY cost matrix
        or feature representation).
    :type target_quad: class:`ArrayLike`

    :param return_matrix: Whether to return the full coupling matrix.
    :type return_matrix: bool

     :param source_lin: Optional source cross-domain features used
        for fused Gromov-Wasserstein. Default `None`
    :type source_lin: class:`ArrayLike`

    :param target_lin: Optional target cross-domain features used
        for fused Gromov-Wasserstein. Default `None`
    :type target_lin: class:`ArrayLike`

    :param cost_fn: Cost function used to compute pairwise distances.
        If ``None``, squared Euclidean distance is used.
    :type cost_fn: costs.CostFn | None

    :param scale_cost: Method or factor used to rescale the cost matrices.
    :type scale_cost: ScaleMethod

    :param method: Quadratic OT method to use. Supported values include
        ``"entropic_gromov_wasserstein"`` and
        ``"entropic_fused_gromov_wasserstein"``.
    :type method: QuadCouplingMethod

    :param kwargs: Additional keyword arguments forwarded to the solver.
    :type kwargs: dict


    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray] or tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        If ``return_matrix=False``, returns:

        - Source indices
        - Target indices

        If ``return_matrix=True``, additionally returns the quadratic
        coupling matrix.
    """
    scale_cost = scale_cost or "mean"

    # moving arrays to jax arrays
    source_quad = to_jax_array(source_quad)
    target_quad = to_jax_array(target_quad)

    source_lin = to_jax_array(source_lin)
    target_lin = to_jax_array(target_lin)
    print(source_lin)

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

    source_idxs, target_idxs = _select_indices(coupling_matrix=coupling_matrix, size=source_quad.shape[0])
    if return_matrix:
        return source_idxs, target_idxs, coupling_matrix
    return source_idxs, target_idxs
