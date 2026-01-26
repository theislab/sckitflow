import logging
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, overload

import numpy as np
import ot as pot
import torch

logger = logging.getLogger(__name__)

ScaleMethod = Literal["mean", "max", "median"] | float


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


def scale_distance_matrix(
    distance_matrix: torch.Tensor,
    scale_method: ScaleMethod,
) -> torch.Tensor:
    """
    Scale a distance matrix according to a given method.

    Parameters
    ----------
    distance_matrix : torch.Tensor
        The distance matrix to be scaled.
    scale_method : {"mean", "max", "median"} or float
        Scaling strategy. If a float is provided, the matrix is divided
        by this value directly.

    Returns
    -------
    torch.Tensor
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


def independent_coupling(
    source: torch.Tensor,
    target: torch.Tensor,
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
    source: torch.Tensor,
    target: torch.Tensor,
    cost_fn: Callable[..., Any] | None = ...,
    scale_cost: float | Literal["mean", "max", "median"] = ...,
    method: Literal["exact", "sinkhorn", "partial", "unbalanced", None] = ...,
    of_fn: Callable[..., Any] | None = ...,
    reg: float = ...,
    reg_m: float = ...,
    return_matrix: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def ot_linear_coupling(
    source: torch.Tensor,
    target: torch.Tensor,
    cost_fn: Callable[..., Any] | None = ...,
    scale_cost: float | Literal["mean", "max", "median"] = ...,
    method: Literal["exact", "sinkhorn", "partial", "unbalanced", None] = ...,
    of_fn: Callable[..., Any] | None = ...,
    reg: float = ...,
    reg_m: float = ...,
    return_matrix: bool = True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def ot_linear_coupling(
    source: torch.Tensor,
    target: torch.Tensor,
    cost_fn: Callable | None = None,
    scale_cost: float | Literal["mean", "max", "median"] = "mean",
    method: Literal["exact", "sinkhorn", "partial", "unbalanced", None] = None,
    of_fn: Callable | None = None,
    reg: float = 5e-1,
    reg_m: float = 1.0,
    return_matrix: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`torch.Tensor`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`torch.Tensor`

    #TODO
    """
    if method is None:
        method = "sinkhorn"

    if cost_fn is None:
        cost_fn = lambda source, target: torch.cdist(source, target) ** 2

    # computing weights
    src_weights = pot.unif(source.shape[0])
    tgt_weights = pot.unif(target.shape[0])
    # moving arrays to torch tensors
    if isinstance(source, np.ndarray):
        source = torch.from_numpy(source)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    # flattening tensors
    source = torch.flatten(source, start_dim=1)
    target = torch.flatten(target, start_dim=1)
    # computing cost matrix
    distance_matrix = cost_fn(source, target)
    # normalization of cost
    distance_matrix = scale_distance_matrix(distance_matrix, scale_method=scale_cost)

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
    # checking for numerical errors in the coupling matrix
    coupling_matrix = sanitize_coupling_matrix(coupling_matrix=coupling_matrix.numpy())
    # retrieving coupling probabilities
    coupling_probs = coupling_matrix.flatten()
    coupling_probs = coupling_probs / coupling_probs.sum()
    # sampling indices
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
    src_xx_cell_coupling: torch.Tensor,
    tgt_yy_cell_coupling: torch.Tensor,
    src_xy_cell_coupling: torch.Tensor | None = ...,
    tgt_xy_cell_coupling: torch.Tensor | None = ...,
    cost_fn: Callable[..., Any] | None = ...,
    scale_cost: float | Literal["mean", "max_cost", "median"] = ...,
    method: Literal[
        "entropic_gromov_wasserstein",
        "entropic_fused_gromov_wasserstein",
        None,
    ] = ...,
    of_fn: Callable[..., Any] | None = ...,
    reg: float = ...,
    reg_m: float = ...,
    return_matrix: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def ot_quadratic_coupling(
    src_xx_cell_coupling: torch.Tensor,
    tgt_yy_cell_coupling: torch.Tensor,
    src_xy_cell_coupling: torch.Tensor | None = ...,
    tgt_xy_cell_coupling: torch.Tensor | None = ...,
    cost_fn: Callable[..., Any] | None = ...,
    scale_cost: float | Literal["mean", "max_cost", "median"] = ...,
    method: Literal[
        "entropic_gromov_wasserstein",
        "entropic_fused_gromov_wasserstein",
        None,
    ] = ...,
    of_fn: Callable[..., Any] | None = ...,
    reg: float = ...,
    reg_m: float = ...,
    return_matrix: bool = True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def ot_quadratic_coupling(
    src_xx_cell_coupling: torch.Tensor,
    tgt_yy_cell_coupling: torch.Tensor,
    src_xy_cell_coupling: torch.Tensor | None = None,
    tgt_xy_cell_coupling: torch.Tensor | None = None,
    cost_fn: Callable | None = None,
    scale_cost: float | Literal["mean", "max_cost", "median"] = 1.0,
    method: Literal[
        "entropic_gromov_wasserstein",
        "entropic_fused_gromov_wasserstein",
        None,
    ] = None,
    of_fn: Callable | None = None,
    reg: float = 5e-1,
    reg_m: float = 1.0,
    return_matrix: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`torch.Tensor`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`torch.Tensor`

    #TODO
    """
    if cost_fn is None:
        cost_fn = lambda source, target: torch.cdist(source, target) ** 2
    if method is None:
        method = "entropic_gromov_wasserstein"

    # moving arrays to torch tensors
    if isinstance(src_xx_cell_coupling, np.ndarray):
        src_xx_cell_coupling = torch.from_numpy(src_xx_cell_coupling)
    if isinstance(tgt_yy_cell_coupling, np.ndarray):
        tgt_yy_cell_coupling = torch.from_numpy(tgt_yy_cell_coupling)

    # computing cost matrix
    distance_matrix_xx = cost_fn(src_xx_cell_coupling, src_xx_cell_coupling)
    distance_matrix_yy = cost_fn(tgt_yy_cell_coupling, tgt_yy_cell_coupling)
    if src_xy_cell_coupling is not None:
        distance_matrix_xy = cost_fn(src_xy_cell_coupling, tgt_xy_cell_coupling)
    else:
        distance_matrix_xy = None
    # normalization of cost
    distance_matrix_xx = scale_distance_matrix(distance_matrix=distance_matrix_xx, scale_method=scale_cost)
    distance_matrix_yy = scale_distance_matrix(distance_matrix=distance_matrix_yy, scale_method=scale_cost)

    if distance_matrix_xy is not None:
        distance_matrix_xy - scale_distance_matrix(distance_matrix=distance_matrix_xy, scale_method=scale_cost)

    if method == "entropic_gromov_wasserstein":
        ot_fn = partial(pot.gromov.entropic_gromov_wasserstein, epsilon=1.0, alpha=1.0)
        # computing coupling matrix
        coupling_matrix = ot_fn(
            C1=distance_matrix_xx,
            C2=distance_matrix_yy,
        )
    elif method == "entropic_fused_gromov_wasserstein" and distance_matrix_xy is not None:
        ot_fn = partial(pot.gromov.entropic_fused_gromov_wasserstein, epsilon=1.0, alpha=0.5)
        # computing coupling matrix
        coupling_matrix = ot_fn(
            C1=distance_matrix_xx,
            C2=distance_matrix_yy,
            M=distance_matrix_xy,
        )
    else:
        msg = f"{method=} is not found, please specify a custom `method` in `ot_fn`"
        raise ValueError(msg)

    # checking for numerical errors in the coupling matrix
    coupling_matrix = sanitize_coupling_matrix(coupling_matrix=coupling_matrix.numpy())

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
