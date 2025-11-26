import logging
from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np
import ot as pot
import torch

logger = logging.getLogger(__name__)


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


def ot_linear_coupling(
    source: torch.Tensor,
    target: torch.Tensor,
    cost_fn: Callable | None = None,
    scale_cost: float | Literal["mean", "max_cost", "median"] = "mean",
    method: Literal["exact", "sinkhorn", "partial", "unbalanced"] = "sinkhorn",
    of_fn: Callable | None = None,
    reg: float = 5e-1,
    reg_m: float = 1.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`torch.Tensor`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`torch.Tensor`

    #TODO
    """
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
    if scale_cost == "mean":
        distance_matrix = distance_matrix / distance_matrix.mean()
    elif scale_cost == "max":
        distance_matrix = distance_matrix / distance_matrix.max()
    elif scale_cost == "median":
        distance_matrix = distance_matrix / distance_matrix.median()
    elif isinstance(scale_cost, float):
        distance_matrix = distance_matrix / scale_cost

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
    elif method in ["exact", "sinkhorn", "partial", "unbalanced"] and ot_fn is None:
        msg = f"{method=} is not found, please specify a custom `method` in `ot_fn`"
        raise ValueError(msg)

    # computing coupling matrix
    coupling_matrix = ot_fn(src_weights, tgt_weights, distance_matrix.detach().cpu().numpy())
    # checking for numerical errors in the coupling matrix
    if not np.all(np.isfinite(coupling_matrix)):
        msg = f"Non finite values found in `coupling_matrix` \n {coupling_matrix=} \n {source=} \n {target=} \n {distance_matrix.mean()=} \n {distance_matrix.max()=}"
        logger.warning(msg)
    if np.abs(coupling_matrix.sum()) < 1e-8:
        msg = ""
        logger.warning(msg)
        coupling_matrix = np.ones_like(coupling_matrix) / coupling_matrix.size
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
    return source_idxs, target_idxs


def ot_quadratic_coupling(
    source: torch.Tensor,
    src_xx_cell_coupling: torch.Tensor,
    tgt_yy_cell_coupling: torch.Tensor,
    src_xy_cell_coupling: torch.Tensor | None = None,
    tgt_xy_cell_coupling: torch.Tensor | None = None,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`torch.Tensor`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`torch.Tensor`

    #TODO
    """
    if cost_fn is None:
        cost_fn = lambda source, target: torch.cdist(source, target) ** 2

    # moving arrays to torch tensors
    if isinstance(src_xx_cell_coupling, np.ndarray):
        src_xx_cell_coupling = torch.from_numpy(src_xx_cell_coupling)
    if isinstance(tgt_yy_cell_coupling, np.ndarray):
        tgt_yy_cell_coupling = torch.from_numpy(tgt_yy_cell_coupling)
    # flattening tensors
    source = torch.flatten(src_xx_cell_coupling, start_dim=1)
    target = torch.flatten(tgt_yy_cell_coupling, start_dim=1)
    # computing cost matrix
    distance_matrix_xx = cost_fn(src_xx_cell_coupling, src_xx_cell_coupling)
    distance_matrix_yy = cost_fn(tgt_yy_cell_coupling, tgt_yy_cell_coupling)
    if src_xy_cell_coupling is not None:
        distance_matrix_xy = cost_fn(src_xy_cell_coupling, tgt_xy_cell_coupling)
    else:
        distance_matrix_xy = None
    # normalization of cost
    if scale_cost == "mean":
        distance_matrix_xx = distance_matrix_xx / distance_matrix_xx.mean()
        distance_matrix_yy = distance_matrix_yy / distance_matrix_yy.mean()
    elif scale_cost == "max":
        distance_matrix_xx = distance_matrix_xx / distance_matrix_xx.max()
        distance_matrix_yy = distance_matrix_yy / distance_matrix_yy.max()
    elif scale_cost == "median":
        distance_matrix_xx = distance_matrix_xx / distance_matrix_xx.median()
        distance_matrix_yy = distance_matrix_yy / distance_matrix_yy.median()
    elif isinstance(scale_cost, float):
        distance_matrix_xx = distance_matrix_xx / scale_cost
        distance_matrix_yy = distance_matrix_yy / scale_cost

    if distance_matrix_xy is not None:
        if scale_cost == "mean":
            distance_matrix_xy = distance_matrix_xy / distance_matrix_xy.mean()
        elif scale_cost == "max":
            distance_matrix_xy = distance_matrix_xy / distance_matrix_xy.max()
        elif scale_cost == "median":
            distance_matrix_xy = distance_matrix_xy / distance_matrix_xy.median()
        elif isinstance(scale_cost, float):
            distance_matrix_xy = distance_matrix_xy / scale_cost

    if method == "entropic_gromov_wasserstein":
        ot_fn = partial(pot.gromov.entropic_fused_gromov_wasserstein, epsilon=1.0, alpha=1.0)
    elif method == "entropic_fused_gromov_wasserstein" and distance_matrix_xy is not None:
        ot_fn = partial(pot.gromov.entropic_fused_gromov_wasserstein, epsilon=1.0, alpha=0.5)
    elif method in ["entropic_gromov_wasserstein"] and ot_fn is None:
        msg = f"{method=} is not found, please specify a custom `method` in `ot_fn`"
        raise ValueError(msg)

    # computing coupling matrix
    coupling_matrix = ot_fn(
        C1=distance_matrix_xx,
        C2=distance_matrix_yy,
        M=distance_matrix_xy,
    )
    # checking for numerical errors in the coupling matrix
    if not np.all(np.isfinite(coupling_matrix)):
        msg = f"Non finite values found in `coupling_matrix` \n {coupling_matrix=} \n {source=} \n {target=} \n {distance_matrix_xx.mean()=} \n {distance_matrix_xx.max()=}"
        logger.warning(msg)
    if np.abs(coupling_matrix.sum()) < 1e-8:
        msg = ""
        logger.warning(msg)
        coupling_matrix = np.ones_like(coupling_matrix) / coupling_matrix.size
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
    return source_idxs, target_idxs
