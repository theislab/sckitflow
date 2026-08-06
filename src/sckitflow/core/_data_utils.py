from typing import Any, Literal

import numpy as np
import torch

from sckitflow.core._types import StepData, TensorMixin, TNoiseSamplerFn, new_step_data
from sckitflow.core._utils import to_torch_tensor
from sckitflow.data import mixins
from sckitflow.data._composite import MatchedData
from sckitflow.data.containers import CategoricalData, CouplingData, DistributionData, MixedTypeData, StateData

__all__ = [
    "batchmixin_to_torch",
    "extract_state_data",
    "extract_coupling_data",
    "extract_distribution_data",
    "get_tensor_dict_from_data",
    "extract_step_data",
    "align_step_data",
    "subscript_step_data",
    "write_continuous_cond_cov_to_step_data",
    "expand_conditioning",
    "prepare_latent_train",
    "prepare_latent_inference",
    "TorchMixedTypeData",
]


def batchmixin_to_torch(
    data: mixins.BatchMixin | dict[str, np.ndarray | torch.Tensor] | None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Converts the elements of a mixin to torch tensors."""
    if data is None:
        return {}
    data_dict = data.mapping if isinstance(data, mixins.BatchMixin) else data
    if not isinstance(data_dict, dict):
        raise ValueError(f"Data dictionary of the wrong type, expected `dict` got {type(data_dict)}.")
    return {k: to_torch_tensor(v, dtype=dtype, device=device) for k, v in data_dict.items()}


def extract_state_data(
    state_data: StateData | None, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> torch.Tensor | None:
    """Extracts torch tensors from the state data."""
    if state_data is None:
        return None
    X_state = state_data.X
    X_state = to_torch_tensor(X_state, device=device, dtype=dtype)
    return X_state


def extract_coupling_data(
    distribution_data: DistributionData,
    mode: Literal["source", "target"],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extracts torch tensors from the coupling data."""
    # retrieve coupling data
    coupling_data: CouplingData | None = getattr(distribution_data, f"{mode}_coupling_data")

    # no coupling data available (e.g. inference without a target state)
    if coupling_data is None:
        return None, None

    # parse coupling data
    state_lin: StateData | None = coupling_data.state_lin
    state_quad: StateData | None = coupling_data.state_quad

    # get states for linear term
    X_lin = extract_state_data(state_lin, device=device, dtype=dtype)
    X_quad = extract_state_data(state_quad, device=device, dtype=dtype)
    return X_lin, X_quad


def extract_distribution_data(
    distribution_data: DistributionData,
    mode: Literal["source", "target"],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[Any]:
    """Extracts torch tensors from the distribution data."""
    coupling_lin, coupling_quad = extract_coupling_data(distribution_data, mode)
    state_data = extract_state_data(distribution_data.state_data, device=device, dtype=dtype)
    condition_data = distribution_data.condition_data
    groups_data = distribution_data.groups_data
    return (coupling_lin, coupling_quad, state_data, condition_data, groups_data)


def get_tensor_dict_from_data(
    data: MixedTypeData | CategoricalData | dict[str, np.ndarray | torch.Tensor] | None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Extracts torch tensors from the distribution data."""
    if data is None:
        return {}
    data_dict = data.extract_reps() if isinstance(data, MixedTypeData | CategoricalData) else data
    return batchmixin_to_torch(data_dict, device=device, dtype=dtype)


def extract_step_data(
    matched_distr: MatchedData, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> StepData:
    """Extracts torch tensors from the matched distribution data."""
    # parse dictionary of matched distributions
    source_data_dict: DistributionData | None = matched_distr.get("source")
    target_data_dict: DistributionData | None = matched_distr["target"]

    # parse target data dictionary
    if target_data_dict is not None:
        (target_coupling_lin, target_coupling_quad, target_state, target_condition_data, target_group_data) = (
            extract_distribution_data(target_data_dict, "target", device=device, dtype=dtype)
        )
        # target-side covariates: not consumed by the model, kept for output reconstruction
        target_response_data = target_data_dict.response_data
    else:
        target_coupling_lin = None
        target_coupling_quad = None
        target_state = None
        target_condition_data = None
        target_group_data = None
        target_response_data = None

    # optionally parse target data dictionary
    if source_data_dict is not None:
        (source_coupling_lin, source_coupling_quad, source_state, source_condition_data, source_group_data) = (
            extract_distribution_data(source_data_dict, "source", device=device, dtype=dtype)
        )
    else:
        source_coupling_lin = None
        source_coupling_quad = None
        source_state = None
        source_condition_data = None
        source_group_data = None

    # return structured output
    return new_step_data(
        target_state=target_state,
        target_coupling_lin=target_coupling_lin,
        target_coupling_quad=target_coupling_quad,
        target_condition_data=target_condition_data,
        target_group_data=target_group_data,
        target_response_data=target_response_data,
        source_state=source_state,
        source_coupling_lin=source_coupling_lin,
        source_coupling_quad=source_coupling_quad,
        source_condition_data=source_condition_data,
        source_group_data=source_group_data,
    )


_SOURCE_FIELDS = (
    "source_state",
    "source_coupling_lin",
    "source_coupling_quad",
    "source_condition_data",
    "source_group_data",
)

_TARGET_FIELDS = (
    "target_state",
    "target_coupling_lin",
    "target_coupling_quad",
    "target_condition_data",
    "target_group_data",
    # target-side covariates: not consumed by the model, but must stay row-aligned with
    # ``target_state`` when the target is permuted/subsampled by matching.
    "target_response_data",
)


def _index_obj(value: Any, idx: Any | None) -> Any:
    """Row-index one :class:`StepData` field by ``idx``.

    Handles the three field shapes that appear in a ``StepData``: ``None`` (passthrough), a
    plain ``dict[str, tensor]`` (each value is indexed; ``None`` values pass through), and any
    row-indexable object -- a ``torch.Tensor`` / array or a :class:`BaseData` container that
    implements ``__getitem__``. A ``None`` index is a no-op.
    """
    if value is None or idx is None:
        return value
    if isinstance(value, dict):
        return {k: (None if v is None else v[idx]) for k, v in value.items()}
    return value[idx]


def subscript_step_data(
    step_data: StepData,
    src_idxs: Any | None = None,
    tgt_idxs: Any | None = None,
) -> StepData:
    """Row-slice a :class:`StepData`: the source side by ``src_idxs``, the target side by ``tgt_idxs``.

    Applies a coupling/matching permutation (or subselection) to a batch. Each side is sliced
    independently; a ``None`` index leaves that side untouched and a ``None`` field stays ``None``.
    The target side includes ``target_response_data``, so target covariates carried for output
    reconstruction remain aligned with ``target_state``. Replaces ``BaseMethod._safe_subscript_obj``.

    :param step_data: The batch to slice.
    :type step_data: class: `StepData`

    :param src_idxs: Index/mask applied to every source field. ``None`` leaves the source untouched.
    :type src_idxs: class: `Any | None`

    :param tgt_idxs: Index/mask applied to every target field. ``None`` leaves the target untouched.
    :type tgt_idxs: class: `Any | None`
    """
    updates = {field: _index_obj(step_data.get(field), tgt_idxs) for field in _TARGET_FIELDS}
    updates.update({field: _index_obj(step_data.get(field), src_idxs) for field in _SOURCE_FIELDS})
    return {**step_data, **updates}


def _n_obs(*fields: Any) -> int | None:
    """Number of observations from the first non-``None`` field (tensor or container)."""
    for field in fields:
        if field is not None:
            return len(field)
    return None


def align_step_data(step_data: StepData) -> StepData:
    """Aligns the source side to the target length by slicing or tiling.

    Replaces the former ``MatchedData.align``. Source and target distributions are not
    constrained to hold the same number of observations, but when the target carries
    continuous condition covariates (per-observation conditioning), a real source of a
    different length cannot be broadcast against it. In that case every source field is
    reindexed with ``arange(n_target) % n_source``, which:

    * slices the source to the first ``n_target`` rows when the source is longer, and
    * tiles the source (whole repeats plus a remainder) when it is shorter.

    It is a no-op when the target has no continuous condition covariates, when there is
    no source (generation from noise), or when the two sides already match in length.
    """
    # no-op unless the target carries continuous (per-observation) condition covariates
    condition_data = step_data["target_condition_data"]
    if not getattr(condition_data, "has_continuous_covariates", False):
        return step_data

    n_source = _n_obs(*(step_data[f] for f in _SOURCE_FIELDS))
    n_target = _n_obs(
        step_data["target_state"],
        step_data["target_condition_data"],
        step_data["target_group_data"],
        step_data["target_coupling_lin"],
        step_data["target_coupling_quad"],
    )

    # no-op when there is no source or the sides already match
    if n_source is None or n_target is None or n_source == n_target:
        return step_data

    # arange(n_target) % n_source slices (n_target < n_source) or tiles (n_target > n_source)
    idx = np.arange(n_target) % n_source
    aligned_source = {f: _index_obj(step_data[f], idx) for f in _SOURCE_FIELDS}
    return {**step_data, **aligned_source}


def write_continuous_cond_cov_to_step_data(
    condition_key: str,
    x: torch.Tensor,
    base_data: MatchedData | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> StepData:
    """Overrides the base data to store the input tensor at the specified key."""
    # ---- 1. No base data provided ----
    if base_data is None:
        condition_dict = {condition_key: x}
        return new_step_data(target_condition_data=condition_dict)
    else:
        # ---- Retrieve Metadata ----
        step_data: StepData = extract_step_data(base_data, device=device, dtype=dtype)
        target_condition_data: MixedTypeData = step_data["target_condition_data"]

        # --- Update covariates ----
        updated_condition_data = TorchMixedTypeData.from_mixed_type_data(
            condition_key, x, base_container=target_condition_data
        )
        updated_step_data: StepData = {**step_data, "target_condition_data": updated_condition_data}
        return updated_step_data


def expand_conditioning(
    latent: torch.Tensor,
    condition_dict: dict[str, torch.Tensor],
    source: torch.Tensor | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Replicate condition_dict and source along the leading dimensions of latent

    Assumes original condition_dict values and source have shape (batch_size, ...)
    and latent has shape (n_samples, batch_size, dim) or (batch_size, dim).
    Returns expanded copies.
    """
    if latent.dim() <= 2:  # no extra sample dimension
        return condition_dict, source

    n_samples = latent.shape[0]
    expanded_cond = {}
    for k, v in condition_dict.items():
        # v shape: (batch_size, ...)
        expanded_cond[k] = v.unsqueeze(0).expand(n_samples, *v.shape)
    expanded_source = None
    if source is not None:
        expanded_source = source.unsqueeze(0).expand(n_samples, *source.shape)
    return expanded_cond, expanded_source


def prepare_latent_train(
    source: torch.Tensor | None,
    target: torch.Tensor,
    noise_sampler: TNoiseSamplerFn,
    generate_from_noise: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Called from compute_loss - always returns single noise per batch element."""
    if source is None or generate_from_noise:
        samples = noise_sampler(target.shape)
        if dtype:
            samples = samples.to(dtype)
        if device:
            samples = samples.to(device)
        return samples
    return source


def prepare_latent_inference(
    source: torch.Tensor | None,
    target_reference: torch.Tensor,
    noise_sampler: TNoiseSamplerFn,
    n_samples: int | None = None,
    generate_from_noise: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Called from infer.

    - If source is given and we are NOT generating from noise, return source unchanged.
    - Otherwise sample noise with shape:
        if n_samples is None:  (batch_size, dim)
        else:                  (n_samples, batch_size, dim)
    """
    if source is None or generate_from_noise:
        shape = target_reference.shape
        if n_samples is not None:
            shape = (n_samples, *shape)
        samples = noise_sampler(shape)
        if dtype:
            samples = samples.to(dtype)
        if device:
            samples = samples.to(device)
        return samples
    return source


class TorchMixedTypeData(MixedTypeData):
    """"""  # noqa

    def extract_reps(self) -> TensorMixin:
        """Extracts the representations for the underlying data."""
        # extract categorical covariates
        if self.categorical_covariates is not None:
            cat_reps = self.categorical_covariates.extract_reps()
        else:
            cat_reps = TensorMixin({})

        # update with continuous covariates
        if self.continuous_covariates is not None:
            return TensorMixin(
                {
                    **self.continuous_covariates.mapping,
                    **cat_reps.mapping,
                }
            )
        # otherwise return only categorical covariates
        return cat_reps

    @classmethod
    def from_mixed_type_data(
        cls, key: str, x_cond: torch.Tensor, base_container: MixedTypeData | None = None
    ) -> "TorchMixedTypeData":
        """Constructor from input data and optional base container.

        Instaties the torch mixed data from a base container, while overriding the continuous covariates at the given key.

        :param key: The contiuous covariates key to override.
        :type key: class: `str`

        :param x_cond: The data for the continuous covariates to override with.
        :type x_cond: class: `torch.Tensor`

        :param base_container: Optional base container from which to retain the other condition covariates.
            Defaults to `None`, in which case the container will be created from scratch.
        :type base_container: class: `MixedTypeData | None`
        """
        # ---- 1. Retrieve additional covariates from base container when available ----
        if base_container is not None:
            categorical_covariates = base_container.categorical_covariates
            continuous_covariates = base_container.continuous_covariates
        else:
            categorical_covariates = None
            continuous_covariates = None

        # ---- 2. Override continuous covariates with the input data at the given key ----
        if continuous_covariates is not None:
            mapping = continuous_covariates.mapping
            mapping[key] = x_cond
        else:
            mapping = {key: x_cond}

        # ---- 3. Handle data type and device for other covariates and initialize mixin ----
        mapping = {k: to_torch_tensor(v, dtype=x_cond.dtype, device=x_cond.device) for k, v in mapping.items()}
        mixin = TensorMixin(mapping)

        # ---- 4. Initialize class ----
        return cls(categorical_covariates=categorical_covariates, continuous_covariates=mixin)
