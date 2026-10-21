import logging
import warnings
from functools import partial
from typing import Literal, overload

import numpy as np
import ot as pot
import torch

from sckitflow.core._types import (
    CostFN,
    LinCouplingMethod,
    NumpyArray,
    QuadCouplingMethod,
    ScaleMethod,
    TensorLike,
)
from sckitflow.core._utils import to_torch_tensor

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

    # Degenerate sum → fallback to uniform. It keeps a long run going, but the batch is then matched at random,
    # so it must not pass as a log line.
    if np.abs(coupling_matrix.sum()) < eps:
        warnings.warn(
            "The optimal transport solver returned an all-zero plan, so this batch is matched uniformly at random. "
            "Scale the cost (`scale_cost='mean'`) or raise `epsilon`.",
            RuntimeWarning,
            stacklevel=2,
        )
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


def _select_indices(
    coupling_matrix: NumpyArray, size: int, *, rng: np.random.Generator
) -> tuple[NumpyArray, NumpyArray]:
    """Samples matching indices from a coupling matrix.

    Draws index pairs from the provided coupling matrix according to the
    normalized coupling probabilities and returns the corresponding source
    and target indices.

    :param coupling_matrix: Optimal transport coupling matrix encoding
        matching probabilities between source and target samples.
    :type coupling_matrix: class:`numpy.ndarray`

    :param size: Number of index pairs to sample from the coupling matrix.
    :type size: int

    :param rng: Generator the matched indices are drawn from.
    :type rng: class:`numpy.random.Generator`

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
    choices = rng.choice(
        coupling_matrix.shape[0] * coupling_matrix.shape[1],
        p=coupling_probs,
        size=size,
        replace=False,
    )
    source_idxs, target_idxs = np.divmod(choices, coupling_matrix.shape[1])
    return source_idxs, target_idxs


def independent_coupling(
    *,
    source_lin: TensorLike | None,
    target_lin: TensorLike | None,
    source_quad: TensorLike | None = None,
    target_quad: TensorLike | None = None,
    rng: np.random.Generator,
) -> tuple[NumpyArray, NumpyArray]:
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`TensorLike`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`TensorLike`

    :param rng: Generator the permutations are drawn from.
    :type rng: class:`numpy.random.Generator`

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing:

        - An array of randomly permuted indices into the source tensor.
        - An array of randomly permuted indices into the target tensor.

        The two arrays have identical length, equal to the minimum number of
        available samples in the source and target tensors.
    """
    # only the row counts matter, so either rep will do
    source = source_lin if source_lin is not None else source_quad
    target = target_lin if target_lin is not None else target_quad
    if source is None or target is None:
        raise ValueError("Independent coupling requires a source and a target representation, None found.")
    n = min(source.shape[0], target.shape[0])
    return rng.permutation(source.shape[0])[:n], rng.permutation(target.shape[0])[:n]


@overload
def ot_linear_coupling(
    *,
    source_lin: TensorLike | None,
    target_lin: TensorLike | None,
    source_quad: TensorLike | None = None,
    target_quad: TensorLike | None = None,
    return_matrix: Literal[False] = False,
    cost_fn: CostFN | None = ...,
    method: LinCouplingMethod,
    scale_cost: ScaleMethod,
    num_iter_max: int | None,
    reg: float | None = None,
    reg_m: float | None = None,
    rng: np.random.Generator,
    m: float | None = None,
    stop_thr: float | None = None,
) -> tuple[NumpyArray, NumpyArray]: ...


@overload
def ot_linear_coupling(
    *,
    source_lin: TensorLike | None,
    target_lin: TensorLike | None,
    source_quad: TensorLike | None = None,
    target_quad: TensorLike | None = None,
    return_matrix: Literal[True] = True,
    cost_fn: CostFN | None = ...,
    method: LinCouplingMethod,
    scale_cost: ScaleMethod,
    num_iter_max: int | None,
    reg: float | None = None,
    reg_m: float | None = None,
    rng: np.random.Generator,
    m: float | None = None,
    stop_thr: float | None = None,
) -> tuple[NumpyArray, NumpyArray, NumpyArray]: ...


def ot_linear_coupling(
    *,
    source_lin: TensorLike | None,
    target_lin: TensorLike | None,
    source_quad: TensorLike | None = None,
    target_quad: TensorLike | None = None,
    return_matrix: bool = False,
    cost_fn: CostFN | None = None,
    method: LinCouplingMethod,
    scale_cost: ScaleMethod,
    num_iter_max: int | None,
    reg: float | None = None,
    reg_m: float | None = None,
    rng: np.random.Generator,
    m: float | None = None,
    stop_thr: float | None = None,
) -> tuple[NumpyArray, NumpyArray] | tuple[NumpyArray, NumpyArray, NumpyArray]:
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices
    using an optimal transport-based linear coupling.

    The solve is delegated to POT, so the solvers' behaviour and defaults follow the installed POT version.

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

    :param method: Optimal transport solver to use: ``"exact"``, ``"sinkhorn"``, ``"partial"`` or ``"unbalanced"``.
    :type method: class:`LinCouplingMethod`

    :param reg: Entropic regularization, required by every method but ``"exact"``.
    :type reg: float, optional

    :param reg_m: Marginal relaxation, required by ``"unbalanced"``.
    :type reg_m: float, optional

    :param return_matrix: If ``True``, also return the optimal transport coupling matrix.
    :type return_matrix: bool

    :param rng: Generator the matched indices are drawn from.
    :type rng: class:`numpy.random.Generator`

    :param m: Transported mass, used only by ``"partial"``. ``None`` is the solver's default.
    :type m: float, optional

    :param num_iter_max: POT's ``numItermax``. ``None`` is the solver's default.
    :type num_iter_max: int, optional

    :param stop_thr: POT's ``stopThr``, not taken by ``"exact"``. ``None`` is the solver's default.
    :type stop_thr: float, optional

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the indices of matched samples in the source and target tensors.

    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        If ``return_matrix`` is ``True``, also returns the optimal transport coupling matrix
        computed between the source and target distributions.
    """  # noqa
    if source_lin is None or target_lin is None:
        raise ValueError("Linear coupling requires linear source and target representations, None found.")
    cost_fn = cost_fn or (lambda source, target: torch.cdist(source, target) ** 2)

    # computing weights
    src_weights = pot.unif(source_lin.shape[0])
    tgt_weights = pot.unif(target_lin.shape[0])
    # moving arrays to torch tensors
    source_lin = to_torch_tensor(source_lin)
    target_lin = to_torch_tensor(target_lin)

    # flattening tensors
    source_lin = torch.flatten(source_lin, start_dim=1)
    target_lin = torch.flatten(target_lin, start_dim=1)
    # computing cost matrix
    distance_matrix = cost_fn(source_lin, target_lin)
    # normalization of cost
    distance_matrix = _scale_distance_matrix(distance_matrix, scale_method=scale_cost)

    if method != "exact" and reg is None:
        msg = f"{method=} requires `reg` to be a `float`, `None` found"
        raise ValueError(msg)
    opts = {k: v for k, v in {"numItermax": num_iter_max, "stopThr": stop_thr}.items() if v is not None}
    if method == "exact":
        ot_fn = partial(pot.emd, **opts)
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn, reg=reg, **opts)
    elif method == "partial":
        ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg, m=m, **opts)
    elif method == "unbalanced":
        if reg_m is None:
            msg = f"{method=} requires `reg_m` to be a `float`, `None` found"
            raise ValueError(msg)
        ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m, **opts)
    else:
        msg = f"{method=} is not found, please specify a custom `method` in `ot_fn`"
        raise ValueError(msg)

    # computing coupling matrix
    coupling_matrix = ot_fn(src_weights, tgt_weights, distance_matrix.detach().cpu().numpy())
    source_idxs, target_idxs = _select_indices(coupling_matrix=coupling_matrix, size=source_lin.shape[0], rng=rng)
    if return_matrix:
        return source_idxs, target_idxs, coupling_matrix
    return source_idxs, target_idxs


@overload
def ot_quadratic_coupling(
    *,
    source_lin: TensorLike | None = ...,
    target_lin: TensorLike | None = ...,
    source_quad: TensorLike | None = ...,
    target_quad: TensorLike | None = ...,
    return_matrix: Literal[False] = False,
    cost_fn: CostFN | None = ...,
    method: QuadCouplingMethod,
    scale_cost: ScaleMethod,
    rng: np.random.Generator,
    epsilon: float,
    max_iter: int,
    tol: float,
    alpha: float | None = None,
) -> tuple[NumpyArray, NumpyArray]: ...


@overload
def ot_quadratic_coupling(
    *,
    source_lin: TensorLike | None = ...,
    target_lin: TensorLike | None = ...,
    source_quad: TensorLike | None = ...,
    target_quad: TensorLike | None = ...,
    return_matrix: Literal[True] = True,
    cost_fn: CostFN | None = ...,
    method: QuadCouplingMethod,
    scale_cost: ScaleMethod,
    rng: np.random.Generator,
    epsilon: float,
    max_iter: int,
    tol: float,
    alpha: float | None = None,
) -> tuple[NumpyArray, NumpyArray, NumpyArray]: ...


def ot_quadratic_coupling(
    *,
    source_lin: TensorLike | None = None,
    target_lin: TensorLike | None = None,
    source_quad: TensorLike | None = None,
    target_quad: TensorLike | None = None,
    return_matrix: bool = False,
    cost_fn: CostFN | None = None,
    method: QuadCouplingMethod,
    scale_cost: ScaleMethod,
    rng: np.random.Generator,
    epsilon: float,
    max_iter: int,
    tol: float,
    alpha: float | None = None,
) -> tuple[NumpyArray, NumpyArray] | tuple[NumpyArray, NumpyArray, NumpyArray]:
    """Matches source and target groups using an optimal transport-based quadratic coupling (Gromov-Wasserstein or fused Gromov-Wasserstein) and returns the respective indices.

    The solve is delegated to POT, so the solvers' behaviour and defaults follow the installed POT version.

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

    :param method: Quadratic optimal transport solver to use: ``"entropic_gromov_wasserstein"`` or
        ``"entropic_fused_gromov_wasserstein"``.
    :type method: class:`QuadCouplingMethod`

    :param rng: Generator the matched indices are drawn from.
    :type rng: class:`numpy.random.Generator`

    :param epsilon: Entropic regularization.
    :type epsilon: float

    :param alpha: Trade-off between the quadratic and linear terms, required by the fused solver.
    :type alpha: float, optional

    :param max_iter: Maximum number of outer iterations.
    :type max_iter: int

    :param tol: Stopping tolerance.
    :type tol: float

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the indices of matched samples in the source and target distributions.

    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        If ``return_matrix`` is ``True``, also returns the quadratic optimal transport coupling
        matrix computed between the source and target distributions.
    """
    # check that quadratic terms were passed
    if source_quad is None:
        msg = "Quadratic coupling requires quadratic source representation, None found."
        raise ValueError(msg)
    if target_quad is None:
        msg = "Quadratic coupling requires quadratic target representation, None found."
        raise ValueError(msg)

    cost_fn = cost_fn or (lambda source, target: torch.cdist(source, target) ** 2)

    # moving arrays to torch tensors
    source_quad = to_torch_tensor(source_quad)
    target_quad = to_torch_tensor(target_quad)

    # No linear term under pure Gromov-Wasserstein (`use_linear=False`).
    source_lin = to_torch_tensor(source_lin) if source_lin is not None else None
    target_lin = to_torch_tensor(target_lin) if target_lin is not None else None

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

    opts = {"epsilon": epsilon, "max_iter": max_iter, "tol": tol}
    if method == "entropic_gromov_wasserstein":
        ot_fn = partial(pot.gromov.entropic_gromov_wasserstein, **opts)
        # computing coupling matrix
        coupling_matrix = ot_fn(
            C1=distance_matrix_source_quad,
            C2=distance_matrix_target_quad,
        )
    elif method == "entropic_fused_gromov_wasserstein":
        if distance_matrix_lin is None:
            msg = f"{method=} requires `source_lin` and `target_lin`, `None` found"
            raise ValueError(msg)
        if alpha is None:
            msg = f"{method=} requires `alpha` to be a `float`, `None` found"
            raise ValueError(msg)
        ot_fn = partial(pot.gromov.entropic_fused_gromov_wasserstein, alpha=alpha, **opts)
        # computing coupling matrix
        coupling_matrix = ot_fn(
            C1=distance_matrix_source_quad,
            C2=distance_matrix_target_quad,
            M=distance_matrix_lin,
        )
    else:
        msg = f"{method=} is not found, please specify a custom `method` in `ot_fn`"
        raise ValueError(msg)
    source_idxs, target_idxs = _select_indices(
        coupling_matrix=coupling_matrix.detach().cpu().numpy(), size=source_quad.shape[0], rng=rng
    )
    if return_matrix:
        return source_idxs, target_idxs, coupling_matrix.detach().cpu().numpy()
    return source_idxs, target_idxs
