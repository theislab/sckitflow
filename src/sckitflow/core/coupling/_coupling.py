from functools import partial
from typing import Literal, overload

import ot as pot
import torch

from sckitflow.core._types import (
    CostFN,
    LinCouplingMethod,
    QuadCouplingMethod,
    ScaleMethod,
    TensorLike,
)
from sckitflow.core._utils import to_torch_tensor


def _sanitize_coupling_matrix(
    coupling_matrix: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Checks a coupling matrix for numerical issues and applies safe fallbacks.

    :param coupling_matrix: Coupling matrix to be checked for numerical stability.
    :type coupling_matrix: class:`torch.Tensor`

    :param eps: Threshold below which the sum of the coupling matrix is considered
        numerically zero.
    :type eps: float

    Returns
    -------
    torch.Tensor
        A numerically stable coupling matrix.
    """
    # Branchless, so the check stays on device: a non-finite or all-zero plan falls back to uniform, and the batch
    # is then matched at random (logged as `coupling_fallback`). Scale the cost (`scale_cost='mean'`) or raise
    # `epsilon` if that happens.
    return torch.where(
        degenerate_plan(coupling_matrix, eps),
        torch.full_like(coupling_matrix, 1 / coupling_matrix.numel()),
        coupling_matrix,
    )


def degenerate_plan(coupling_matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Whether a plan is non-finite or all zero, so matching falls back to random pairs; a 0-d bool tensor on device."""
    return ~torch.isfinite(coupling_matrix).all() | (coupling_matrix.sum().abs() < eps)


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

    # a zero denominator is not checked, as that would sync with the host; the non-finite plan it gives falls back to uniform
    return distance_matrix / denom


def _select_indices(
    coupling_matrix: torch.Tensor, size: int, *, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples matching indices from a coupling matrix.

    Draws index pairs i.i.d. from the normalized coupling matrix, with replacement, and returns the
    corresponding source and target indices. Drawing without replacement would not follow the plan: each draw
    renormalizes the remaining mass, pulling pair frequencies toward uniform.

    :param coupling_matrix: Optimal transport coupling matrix encoding
        matching probabilities between source and target samples.
    :type coupling_matrix: class:`torch.Tensor`

    :param size: Number of index pairs to sample from the coupling matrix.
    :type size: int

    :param generator: Generator the matched indices are drawn from, on the inputs' device.
    :type generator: class:`torch.Generator`

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:

        - An array of sampled source indices.
        - An array of sampled target indices.

        The two arrays have identical length equal to :param:`size`.
    """
    # checking for numerical errors in the coupling matrix
    coupling_matrix = _sanitize_coupling_matrix(coupling_matrix=coupling_matrix)

    # inverse CDF over the flattened plan, as ott-jax's `sample_joint`: on the plan's device, any plan size
    # (`torch.multinomial` stops at 2**24 categories, a 4096 x 4096 plan)
    cdf = coupling_matrix.flatten().cumsum(0)
    u = torch.rand(size, generator=generator, device=cdf.device, dtype=cdf.dtype) * cdf[-1]
    choices = torch.searchsorted(cdf, u, right=True).clamp_(max=cdf.numel() - 1)
    return choices // coupling_matrix.shape[1], choices % coupling_matrix.shape[1]


def independent_coupling(
    *,
    source_lin: TensorLike | None,
    target_lin: TensorLike | None,
    source_quad: TensorLike | None = None,
    target_quad: TensorLike | None = None,
    generator: torch.Generator,
    return_matrix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

    :param source: A tensor of values containing the data coming from the source distribution.
    :type source: class:`TensorLike`

    :param target: A tensor of values containing the data coming from the target distribution.
    :type target: class:`TensorLike`

    :param generator: Generator the permutations are drawn from.
    :type generator: class:`torch.Generator`

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
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
    device = generator.device
    idxs = (
        torch.randperm(source.shape[0], generator=generator, device=device)[:n],
        torch.randperm(target.shape[0], generator=generator, device=device)[:n],
    )
    if return_matrix:  # the plan it samples from: uniform
        return *idxs, torch.full(
            (source.shape[0], target.shape[0]), 1 / (source.shape[0] * target.shape[0]), device=device
        )
    return idxs


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
    generator: torch.Generator,
    m: float | None = None,
    stop_thr: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...


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
    generator: torch.Generator,
    m: float | None = None,
    stop_thr: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


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
    generator: torch.Generator,
    m: float | None = None,
    stop_thr: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    :param generator: Generator the matched indices are drawn from, on the inputs' device.
    :type generator: class:`torch.Generator`

    :param m: Transported mass, used only by ``"partial"``. ``None`` is the solver's default.
    :type m: float, optional

    :param num_iter_max: POT's ``numItermax``. ``None`` is the solver's default.
    :type num_iter_max: int, optional

    :param stop_thr: POT's ``stopThr``, not taken by ``"exact"``. ``None`` is the solver's default.
    :type stop_thr: float, optional

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing the indices of matched samples in the source and target tensors.

    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        If ``return_matrix`` is ``True``, also returns the optimal transport coupling matrix
        computed between the source and target distributions.
    """  # noqa
    if source_lin is None or target_lin is None:
        raise ValueError("Linear coupling requires linear source and target representations, None found.")
    cost_fn = cost_fn or (lambda source, target: torch.cdist(source, target) ** 2)

    # moving arrays to torch tensors
    source_lin = to_torch_tensor(source_lin)
    target_lin = to_torch_tensor(target_lin)

    # flattening tensors
    source_lin = torch.flatten(source_lin, start_dim=1)
    target_lin = torch.flatten(target_lin, start_dim=1)
    # computing cost matrix
    distance_matrix = cost_fn(source_lin, target_lin)
    # normalization of cost
    distance_matrix = _scale_distance_matrix(distance_matrix, scale_method=scale_cost).detach()
    # uniform weights on the cost's device and dtype, so POT's torch backend solves there
    n_src, n_tgt = distance_matrix.shape
    src_weights = distance_matrix.new_full((n_src,), 1 / n_src)
    tgt_weights = distance_matrix.new_full((n_tgt,), 1 / n_tgt)

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
    coupling_matrix = ot_fn(src_weights, tgt_weights, distance_matrix)
    source_idxs, target_idxs = _select_indices(
        coupling_matrix=coupling_matrix, size=source_lin.shape[0], generator=generator
    )
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
    generator: torch.Generator,
    epsilon: float,
    max_iter: int,
    tol: float,
    alpha: float | None = None,
    inner_num_iter_max: int | None = None,
    inner_stop_thr: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...


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
    generator: torch.Generator,
    epsilon: float,
    max_iter: int,
    tol: float,
    alpha: float | None = None,
    inner_num_iter_max: int | None = None,
    inner_stop_thr: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


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
    generator: torch.Generator,
    epsilon: float,
    max_iter: int,
    tol: float,
    alpha: float | None = None,
    inner_num_iter_max: int | None = None,
    inner_stop_thr: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    :param generator: Generator the matched indices are drawn from, on the inputs' device.
    :type generator: class:`torch.Generator`

    :param epsilon: Entropic regularization.
    :type epsilon: float

    :param alpha: Trade-off between the quadratic and linear terms, required by the fused solver.
    :type alpha: float, optional

    :param max_iter: Maximum number of outer iterations.
    :type max_iter: int

    :param tol: Stopping tolerance.
    :type tol: float

    :param inner_num_iter_max: POT's ``numItermax`` of the inner Sinkhorn solves. ``None`` is the solver's default.
    :type inner_num_iter_max: int, optional

    :param inner_stop_thr: POT's ``stopThr`` of the inner Sinkhorn solves. ``None`` is the solver's default, which
        float32 may never reach, running every solve to ``numItermax``.
    :type inner_stop_thr: float, optional

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing the indices of matched samples in the source and target distributions.

    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
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

    # No linear term under pure Gromov-Wasserstein (no `rep_lin`).
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

    inner = {"numItermax": inner_num_iter_max, "stopThr": inner_stop_thr}
    opts = {"epsilon": epsilon, "max_iter": max_iter, "tol": tol, **{k: v for k, v in inner.items() if v is not None}}
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
    coupling_matrix = coupling_matrix.detach()
    source_idxs, target_idxs = _select_indices(
        coupling_matrix=coupling_matrix, size=source_quad.shape[0], generator=generator
    )
    if return_matrix:
        return source_idxs, target_idxs, coupling_matrix
    return source_idxs, target_idxs
