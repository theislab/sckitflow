import logging
from functools import partial
from typing import Literal, overload

import numpy as np
import ot as pot
import torch

from sc_flow.backends.torch._types import (
    CostFN,
    LinCouplingMethod,
    NumpyArray,
    QuadCouplingMethod,
    ScaleMethod,
    TensorLike,
)
from sc_flow.backends.torch._utils import to_torch_tensor

logger = logging.getLogger(__name__)


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


def _scale_distance_matrix(
    distance_matrix: TensorLike,
    scale_method: ScaleMethod,
) -> TensorLike:
    """
    Scale a distance matrix according to a given method.

    Parameters
    ----------
    distance_matrix : TensorLike
        The distance matrix to be scaled.
    scale_method : {"mean", "max", "median"} or float
        Scaling strategy. If a float is provided, the matrix is divided
        by this value directly.

    Returns
    -------
    TensorLike
        The scaled distance matrix.
    """
    if isinstance(scale_method, float):
        if scale_method == 0.0:
            msg = "`scale_method` m`ust be non-zero when given as a float."
            raise ValueError(msg)
        return distance_matrix / scale_method

    if scale_method == "mean":
        denom = distance_matrix.mean()
    elif scale_method == "max":
        denom = distance_matrix.max()
    elif scale_method == "median":
        denom = distance_matrix.median()
    else:
        # Should be unreachable due to typing
        msg = f"Unknown scale_method: {scale_method}"
        raise ValueError(msg)

    if denom == 0:
        msg = "Scaling denominator is zero; cannot scale distance matrix."
        raise ValueError(msg)

    return distance_matrix / denom


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


def independent_coupling(
    source: TensorLike,
    target: TensorLike,
) -> tuple[NumpyArray, NumpyArray]:
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`TensorLike`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`TensorLike`

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing:

        - An array of randomly permuted indices into the source tensor.
        - An array of randomly permuted indices into the target tensor.

        The two arrays have identical length, equal to the minimum number of
        available samples in the source and target tensors.
    """
    # randomy permuting the tensors
    src_random_perm_idx = np.random.choice(source.shape[0], size=source.shape[0], replace=False)
    tgt_random_perm_idx = np.random.choice(target.shape[0], size=source.shape[0], replace=False)

    min_shape = min(src_random_perm_idx.shape[0], tgt_random_perm_idx.shape[0])

    return src_random_perm_idx[:min_shape], tgt_random_perm_idx[:min_shape]


@overload
def ot_linear_coupling(
    source: TensorLike,
    target: TensorLike,
    return_matrix: Literal[False],
    cost_fn: CostFN | None = ...,
    scale_cost: ScaleMethod = ...,
    method: LinCouplingMethod = ...,
    reg: float = ...,
    reg_m: float = ...,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray]: ...


@overload
def ot_linear_coupling(
    source: TensorLike,
    target: TensorLike,
    return_matrix: Literal[True],
    cost_fn: CostFN | None = ...,
    scale_cost: ScaleMethod = ...,
    method: LinCouplingMethod = ...,
    reg: float = ...,
    reg_m: float = ...,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray, NumpyArray]: ...


def ot_linear_coupling(
    source: TensorLike,
    target: TensorLike,
    return_matrix: bool = False,
    cost_fn: CostFN | None = None,
    scale_cost: ScaleMethod = "mean",
    method: LinCouplingMethod = None,
    reg: float = 5e-1,
    reg_m: float = 1.0,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray] | tuple[NumpyArray, NumpyArray, NumpyArray]:
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices
    using an optimal transport-based linear coupling.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`TensorLike`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`TensorLike`

    :param cost_fn: A callable used to compute the pairwise cost matrix between source and
        target samples. If ``None``, the squared Euclidean distance is used.
    :type cost_fn: class:`CostFN`, optional

    :param scale_cost: Method used to scale the cost matrix. Can be a scalar value or one of
        ``"mean"``, ``"max"``, or ``"median"``.
    :type scale_cost: class:`ScaleMethod`

    :param method: Optimal transport solver to use. If ``None``, defaults to ``"sinkhorn"``.
        Supported methods are ``"exact"``, ``"sinkhorn"``, ``"partial"``, and ``"unbalanced"``.
    :type method: class:`LinCouplingMethod`, optional

    :param reg: Entropic regularization parameter used by Sinkhorn-based solvers.
    :type reg: float

    :param reg_m: Marginal relaxation parameter used by unbalanced optimal transport solvers.
    :type reg_m: float

    :param return_matrix: If ``True``, also return the optimal transport coupling matrix.
    :type return_matrix: bool

    :param kwargs: Additional keyword arguments forwarded to the selected optimal transport solver.
    :type kwargs: dict

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the indices of matched samples in the source and target tensors.

    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        If ``return_matrix`` is ``True``, also returns the optimal transport coupling matrix
        computed between the source and target distributions.
    """  # noqa
    method = method or "sinkhorn"

    cost_fn = cost_fn or (lambda source, target: torch.cdist(source, target) ** 2)

    # computing weights
    src_weights = pot.unif(source.shape[0])
    tgt_weights = pot.unif(target.shape[0])
    # moving arrays to torch tensors
    source = to_torch_tensor(source)
    target = to_torch_tensor(target)

    # flattening tensors
    source = torch.flatten(source, start_dim=1)
    target = torch.flatten(target, start_dim=1)
    # computing cost matrix
    distance_matrix = cost_fn(source, target)
    # normalization of cost
    distance_matrix = _scale_distance_matrix(distance_matrix, scale_method=scale_cost)

    if method == "exact":
        ot_fn = pot.emd
    elif method == "sinkhorn":
        if reg is None:
            msg = f"{method=} requires `reg` to be a `float`, `None` found"
            raise ValueError(msg)
        ot_fn = partial(pot.sinkhorn, reg=reg, **kwargs)
    elif method == "partial":
        if reg is None:
            msg = f"{method=} requires `reg` to be a `float`, `None` found"
            raise ValueError(msg)
        ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg, **kwargs)
    elif method == "unbalanced":
        if reg is None:
            msg = f"{method=} requires `reg` to be a `float`, `None` found"
            raise ValueError(msg)
        if reg_m is None:
            msg = f"{method=} requires `reg_m` to be a `float`, `None` found"
            raise ValueError(msg)
        ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m, **kwargs)
    else:
        msg = f"{method=} is not found, please specify a custom `method` in `ot_fn`"
        raise ValueError(msg)

    # computing coupling matrix
    coupling_matrix = ot_fn(src_weights, tgt_weights, distance_matrix.detach().cpu().numpy())
    source_idxs, target_idxs = _select_indices(coupling_matrix=coupling_matrix, size=source.shape[0])
    if return_matrix:
        return source_idxs, target_idxs, coupling_matrix
    return source_idxs, target_idxs


@overload
def ot_quadratic_coupling(
    source_quad: TensorLike,
    target_quad: TensorLike,
    return_matrix: Literal[False],
    source_lin: TensorLike | None = ...,
    target_lin: TensorLike | None = ...,
    cost_fn: CostFN | None = ...,
    scale_cost: ScaleMethod = ...,
    method: QuadCouplingMethod = ...,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray]: ...


@overload
def ot_quadratic_coupling(
    source_quad: TensorLike,
    target_quad: TensorLike,
    return_matrix: Literal[True],
    source_lin: TensorLike | None = ...,
    target_lin: TensorLike | None = ...,
    cost_fn: CostFN | None = ...,
    scale_cost: ScaleMethod = ...,
    method: QuadCouplingMethod = ...,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray, NumpyArray]: ...


def ot_quadratic_coupling(
    source_quad: TensorLike,
    target_quad: TensorLike,
    return_matrix: bool = False,
    source_lin: TensorLike | None = None,
    target_lin: TensorLike | None = None,
    cost_fn: CostFN | None = None,
    scale_cost: ScaleMethod = 1.0,
    method: QuadCouplingMethod = None,
    **kwargs,
) -> tuple[NumpyArray, NumpyArray] | tuple[NumpyArray, NumpyArray, NumpyArray]:
    """Matches source and target groups using an optimal transport-based quadratic coupling
    (Gromov-Wasserstein or fused Gromov-Wasserstein) and returns the respective indices.

    :param source_quad: A tensor containing pairwise relationships within the source
        distribution (source-source structure).
    :type source_quad: class:`TensorLike`

    :param target_quad: A tensor containing pairwise relationships within the target
        distribution (target-target structure).
    :type target_quad: class:`TensorLike`

    :param return_matrix: If ``True``, also return the quadratic optimal transport coupling matrix.
    :type return_matrix: bool

    :param source_lin: Optional source cross-domain features used
    for fused Gromov-Wasserstein.
    :type source_lin: class:`TensorLike`

    :param target_lin: Optional target cross-domain features used
    for fused Gromov-Wasserstein.
    :type target_lin: class:`TensorLike`

    :param cost_fn: A callable used to compute pairwise cost matrices. If ``None``, the squared
        Euclidean distance is used.
    :type cost_fn: class:`CostFN`, optional

    :param scale_cost: Method used to scale the cost matrices. Can be a scalar value or one of
        ``"mean"``, ``"max_cost"``, or ``"median"``.
    :type scale_cost: class:`ScaleMethod`

    :param method: Quadratic optimal transport solver to use. If ``None``, defaults to
        ``"entropic_gromov_wasserstein"``. Supported methods are
        ``"entropic_gromov_wasserstein"`` and
        ``"entropic_fused_gromov_wasserstein"``.
    :type method: class:`QuadCouplingMethod`, optional

    :param kwargs: Additional keyword arguments forwarded to the solver.
    :type kwargs: dict

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the indices of matched samples in the source and target distributions.

    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        If ``return_matrix`` is ``True``, also returns the quadratic optimal transport coupling
        matrix computed between the source and target distributions.
    """  # noqa
    cost_fn = cost_fn or (lambda source, target: torch.cdist(source, target) ** 2)
    method = method or "entropic_gromov_wasserstein"

    # moving arrays to torch tensors
    source_quad = to_torch_tensor(source_quad)
    target_quad = to_torch_tensor(target_quad)

    source_lin = to_torch_tensor(source_lin)
    target_lin = to_torch_tensor(target_lin)

    if source_lin is not None and target_lin is not None:
        distance_matrix_lin = cost_fn(source_lin, target_lin)
        distance_matrix_lin = _scale_distance_matrix(distance_matrix=distance_matrix_lin, scale_method=scale_cost)
    else:
        distance_matrix_lin = None

    # computing cost matrix
    distance_matrix_source_quad = cost_fn(source_quad, source_quad)
    distance_matrix_target_quad = cost_fn(target_quad, target_quad)
    distance_matrix_source_quad = _scale_distance_matrix(
        distance_matrix=distance_matrix_source_quad, scale_method=scale_cost
    )
    distance_matrix_target_quad = _scale_distance_matrix(
        distance_matrix=distance_matrix_target_quad, scale_method=scale_cost
    )

    if method == "entropic_gromov_wasserstein":
        ot_fn = partial(pot.gromov.entropic_gromov_wasserstein, epsilon=1.0, alpha=1.0)
        # computing coupling matrix
        coupling_matrix = ot_fn(
            C1=distance_matrix_source_quad,
            C2=distance_matrix_target_quad,
        )
    elif method == "entropic_fused_gromov_wasserstein" and distance_matrix_lin is not None:
        ot_fn = partial(pot.gromov.entropic_fused_gromov_wasserstein, epsilon=1.0, alpha=0.5)
        # computing coupling matrix
        coupling_matrix = ot_fn(
            C1=distance_matrix_source_quad,
            C2=distance_matrix_target_quad,
            M=distance_matrix_lin,
        )
    else:
        msg = f"{method=} is not found, please specify a custom `method` in `ot_fn`"
        raise ValueError(msg)
    print(type(coupling_matrix))
    source_idxs, target_idxs = _select_indices(coupling_matrix=coupling_matrix.numpy(), size=source_quad.shape[0])
    if return_matrix:
        return source_idxs, target_idxs, coupling_matrix.numpy()
    return source_idxs, target_idxs
