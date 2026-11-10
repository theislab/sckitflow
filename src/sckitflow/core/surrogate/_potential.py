from typing import Any

import torch

from sckitflow._random import generators
from sckitflow.core._data_utils import get_tensor_dict_from_data
from sckitflow.core._types import StepData
from sckitflow.core.methods._base import SupportsInference, _placement_of
from sckitflow.core.surrogate._config import Potential, Readout
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._dims import DataDimensions
from sckitflow.data._utils import AnnDataLike

__all__ = ["SurrogatePotential", "SurrogateResponse"]


def _pair_leaf(
    step_data: StepData,
    cond_dict: dict[str, torch.Tensor],
) -> StepData:
    r"""Lays a leaf out against the condition dictionary for one prediction.

    A leaf has ``S`` source rows (when a source state is present) and ``T`` target rows (when it carries
    per-observation conditions that are not optimized). Its group data describes the whole leaf, one row,
    and broadcasts wherever it is used.

    With neither, the $N$ conditions are the batch, the layout prediction reads for generation from noise.
    Otherwise the ``P`` pairs are the Cartesian product of the source and target rows, laid out as
    $(P, N, \dots)$. ``ODEInference`` reads that layout only with a source and ``generate_from_noise=False``,
    where ``P`` acts as its sample axis.

    Target fields prediction does not read (states, couplings, responses) are dropped, since the
    conditions replace the target side. Nothing is moved: the leaf and the conditions are on one device.
    """
    optimized = set(cond_dict)
    extra = {
        k: v for k, v in get_tensor_dict_from_data(step_data.get("target_condition_data")).items() if k not in optimized
    }
    source = step_data.get("source_state")
    S = None if source is None else source.shape[0]
    T = None
    if len(extra) > 0:
        counts = {k: v.shape[0] for k, v in extra.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(f"Condition covariates disagree on their rows within a leaf: {counts}.")
        T = next(iter(counts.values()))
    n = next(iter(cond_dict.values())).shape[0]
    groups = step_data.get("target_group_data")
    kept: StepData = {key: (None if key.startswith("target_") else value) for key, value in step_data.items()}

    if S is None and T is None:
        conds = dict(cond_dict)
        return {**kept, "target_condition_data": conds, "target_group_data": groups, "n_obs": n}

    s = 1 if S is None else S
    t = 1 if T is None else T
    device = next(iter(cond_dict.values())).device
    src_idx = torch.arange(s, device=device).repeat_interleave(t)
    tgt_idx = torch.arange(t, device=device).repeat(s)
    n_pairs = s * t

    def lay_out(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = x[idx]
        return x.unsqueeze(1).expand(n_pairs, n, *x.shape[1:])

    def lay_out_container(container: Any, idx: torch.Tensor) -> Any:
        if container is None:
            return None
        if isinstance(container, torch.Tensor):
            return lay_out(container, idx)
        return {k: lay_out(v, idx) for k, v in get_tensor_dict_from_data(container).items()}

    conds = {k: v.unsqueeze(0).expand(n_pairs, *v.shape) for k, v in cond_dict.items()}
    paired: StepData = {
        **kept,
        **{key: lay_out_container(value, src_idx) for key, value in step_data.items() if key.startswith("source_")},
        "target_group_data": groups,
        "target_condition_data": {**{k: lay_out(v, tgt_idx) for k, v in extra.items()}, **conds},
        "n_obs": n,
    }
    return paired


def _verify_continuous_conditions_dims(
    dims: DataDimensions,
    cond_dict: dict[str, torch.Tensor],
) -> None:
    """Verifies that the continuous condition dictionary respects the dimensionality contract.

    The condition tensor will be checked against the dimensionality contract over the
    trailing dimensions -- that needs to correspond to the entry with the same key in
    `dims`.

    :param dims: `DataDimensions` containing the contract for the data dimensionalities.
    :param cond_dict: Dictionary of `torch.Tensors` whose dimension to verify.
    """
    # Parse the data dimensionality to get the registry for continuous condition
    # covariates. This can be `None`, so we raise an error immediately when this is the
    # case, as the check is to be done over the continuous covariates.
    ref_continuous_dims: dict[str, int] | None = dims.condition_continuous_dims
    # We immediately raise ValueError when the reference continuous dimensions are None.
    if ref_continuous_dims is None:
        raise ValueError("The reference dimensions do not include continuous conditions covariate.")

    # Iterate over the condition dictionary items and verify the trailing dimensions.
    for cond_key, cond_data in cond_dict.items():
        # Raise error here, because at the moment we only support two-dimensional
        # condition data for optimization.
        if cond_data.ndim != 2:
            raise ValueError(f"Only two-dimensional inputs supported for the moment, but {cond_data.ndim} were found.")

        # Raise error when the condition key does not appear in the dimensionality.
        if cond_key not in ref_continuous_dims:
            raise ValueError(
                f"Condition key {cond_key} does not appear in the reference dimensions {sorted(ref_continuous_dims)}."
            )
        # Raise error when the dimensionalities mismatch
        ref_dim = ref_continuous_dims[cond_key]
        query_dim = cond_data.shape[-1]
        if query_dim != ref_dim:
            raise ValueError(f"Shape mismatch for condition key {cond_key} -- got {query_dim} but expected {ref_dim}.")


class SurrogateResponse(torch.nn.Module):
    """The response a trained inference method predicts for continuous condition covariates.

    For a dictionary of conditions it predicts each group of the context data under those conditions
    (``inferer.predict``), maps every generated cell through ``readout`` and averages them, one response per
    condition and group. The context is read once per device, onto the inference module's device.

    :param inferer: An instantiated object satisfying the `SupportsInference` protocol.
    :param datamodule: The `FlowDataModule` whose schema reads ``adata``.
    :param adata: The context data, one prediction per group.
    :param readout: A per-cell map ``(..., D) -> (..., D')``, e.g. a cell-type classifier's probabilities. It is
        applied before the mean because a nonlinear map does not commute with it. ``None`` is the identity: the
        mean generated cell.
    :param max_per_group: Per-group cap on the context observations; ``None`` reads them all.
    :param require_target_state: Whether ``adata`` carries a target state to read.
    :param seed: Seeds the generated cells.
    """

    def __init__(
        self,
        *,
        inferer: SupportsInference,
        datamodule: FlowDataModule,
        adata: AnnDataLike,
        readout: Readout | None = None,
        max_per_group: int | None = None,
        require_target_state: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.inferer, self.datamodule, self.adata, self.readout, self.seed = inferer, datamodule, adata, readout, seed
        self._eval_kwargs = {"max_per_group": max_per_group, "require_target_state": require_target_state}
        self._leaves: dict[str, list[StepData]] = {}  # per device: the context's leaves, read once

    def forward(self, cond_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """The response of each condition in each context group.

        :param cond_dict: A dictionary of input tensors, each of shape ``(N, D)``.
        :returns: Shape ``(n_groups, N, D')``.
        """
        return self.population_responses(cond_dict, n_populations=1)[0]

    def population_responses(self, cond_dict: dict[str, torch.Tensor], n_populations: int) -> torch.Tensor:
        """The responses of ``n_populations`` independently drawn populations.

        Each population draws its own generator per group, so the spread across populations is the
        forward model's predictive uncertainty.

        :param cond_dict: A dictionary of input tensors, each of shape ``(N, D)``.
        :returns: Shape ``(n_populations, n_groups, N, D')``.
        """
        _verify_continuous_conditions_dims(self.datamodule.data_dims, cond_dict)
        if len(cond_dict) == 0:
            raise ValueError("Cannot compute the response of an empty condition dictionary.")
        n = next(iter(cond_dict.values())).shape[0]
        for cond_key, cond_data in cond_dict.items():
            if cond_data.shape[0] != n:
                raise ValueError(
                    f"Condition data at key {cond_key} has the wrong number of optimization samples -- "
                    f"found {cond_data.shape[0]} but expected {n}."
                )
        device, _ = _placement_of(self.inferer.module)
        if off := [k for k, v in cond_dict.items() if v.device != device]:
            raise ValueError(f"Conditions {off} are not on the inference module's device {device}.")
        leaves = self.context_leaves(device)
        if len(leaves) == 0:
            raise ValueError("The context data module has no prediction leaves.")
        populations = []
        for population in range(n_populations):
            groups = []
            for i, step_data in enumerate(leaves):
                # population 0 keeps the single-population stream, so `forward` draws what it always drew
                generator = generators(self.seed, i, *((population,) if population else ()), device=device)
                paired_step_data = _pair_leaf(step_data, cond_dict)
                pred = self.inferer.predict(paired_step_data, generator=generator)
                cells = pred.raw_samples if pred.raw_samples is not None else pred.X
                if self.readout is not None:
                    cells = self.readout(cells)
                    if cells.shape[-2] != n:
                        raise ValueError(
                            f"The readout must keep one row per condition on axis -2: got {cells.shape[-2]}, "
                            f"expected {n}."
                        )
                # average over every axis before the conditions' (samples, the leaf's pairs)
                groups.append(cells.reshape(-1, *cells.shape[-2:]).mean(0))
            populations.append(torch.stack(groups))
        return torch.stack(populations)

    def context_leaves(self, device: torch.device) -> list[StepData]:
        """The context's leaves on ``device``, read on first use and kept."""
        key = str(device)
        if key not in self._leaves:
            loader = self.datamodule.dm.get_eval_loader(
                self.adata, dtype=self.datamodule.dtype, device=key, **self._eval_kwargs
            )
            self._leaves[key] = [step_data for step_data, _ in loader]
        return self._leaves[key]


class SurrogatePotential(torch.nn.Module):
    """A differentiable potential of continuous condition covariates: ``potential(response(x), ystar)``.

    Scores the (masked) response of each context group against the target ``ystar`` and averages over the groups.

    :param response: The response the potential scores.
    :param ystar: The target response, of shape ``(D',)``.
    :param potential: ``(y, ystar)``, the masked responses ``(..., N, D')`` and target ``(D',)`` -> ``(..., N)``.
    :param mask: A boolean mask of shape ``(D',)`` selecting the response dimensions to score. Coerced to bool;
        index lists are not supported. ``None`` selects all of them.
    """

    ystar: torch.Tensor
    mask: torch.Tensor

    def __init__(
        self,
        response: SurrogateResponse,
        *,
        ystar: torch.Tensor,
        potential: Potential,
        mask: torch.Tensor | None = None,
    ) -> None:
        ystar = torch.atleast_1d(torch.squeeze(ystar))
        if ystar.ndim != 1:
            raise ValueError(f"Invalid shape for target response, found {ystar.shape}")
        if mask is not None and mask.dtype not in (torch.bool, torch.uint8):
            raise ValueError("mask must be a per-dimension boolean mask, not an index list.")
        mask = (torch.atleast_1d(torch.squeeze(mask)) if mask is not None else torch.ones_like(ystar)).bool()
        if mask.shape != ystar.shape:
            raise ValueError(
                "The mask and the target tensor should have the same shape -- "
                f"found mask of shape {mask.shape}, but target has shape {ystar.shape}."
            )
        super().__init__()
        self.register_buffer("ystar", ystar)
        self.register_buffer("mask", mask)
        self.response, self.potential = response, potential

    def forward(self, cond_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """The potential of each condition, averaged over the context groups.

        :param cond_dict: A dictionary of input tensors, each of shape ``(N, D)``.
        :returns: Shape ``(N,)``.
        """
        return self.population_potentials(cond_dict, n_populations=1)[0]

    def population_potentials(self, cond_dict: dict[str, torch.Tensor], n_populations: int) -> torch.Tensor:
        """The potential of each condition for ``n_populations`` independently drawn populations.

        The spread across populations is the forward model's predictive uncertainty:
        ``population_potentials(c, n).std(0)``.

        :param cond_dict: A dictionary of input tensors, each of shape ``(N, D)``.
        :returns: Shape ``(n_populations, N)``.
        """
        ys = self.response.population_responses(cond_dict, n_populations)
        if ys.shape[-1] != self.ystar.shape[-1]:
            raise ValueError(
                f"The response has last dim {ys.shape[-1]}, but the target response has dim {self.ystar.shape[-1]}."
            )
        n_populations, n_groups, n = ys.shape[:3]
        potentials = self.potential(ys[..., self.mask], self.ystar[self.mask])
        if potentials.shape != (n_populations, n_groups, n):
            raise ValueError(
                f"The potential has the wrong shape -- got {potentials.shape}, but expected "
                f"{(n_populations, n_groups, n)}. It must broadcast over leading axes, one value per row."
            )
        return potentials.mean(1)
