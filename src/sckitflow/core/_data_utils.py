from dataclasses import replace
from typing import Any, Literal

import numpy as np
import torch

from sckitflow.core._types import StepData, TensorMixin, TNoiseSamplerFn
from sckitflow.core._utils import to_torch_tensor
from sckitflow.data import mixins
from sckitflow.data._composite import MatchedDistributions
from sckitflow.data.containers import CategoricalData, CouplingData, DistributionData, MixedTypeData, StateData

__all__ = [
    "batchmixin_to_torch",
    "extract_state_data",
    "extract_coupling_data",
    "extract_distribution_data",
    "get_tensor_dict_from_data",
    "extract_step_data",
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
    coupling_data: CouplingData = getattr(distribution_data, f"{mode}_coupling_data")

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
    matched_distr: MatchedDistributions, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> StepData:
    """Extracts torch tensors from the matched distribution data."""
    # parse dictionary of matched distributions
    source_data_dict: DistributionData | None = matched_distr.source_distribution
    target_data_dict: DistributionData | None = matched_distr.target_distribution

    # parse target data dictionary
    if target_data_dict is not None:
        (target_coupling_lin, target_coupling_quad, target_state, target_condition_data, target_group_data) = (
            extract_distribution_data(target_data_dict, "target", device=device, dtype=dtype)
        )
    else:
        target_coupling_lin = None
        target_coupling_quad = None
        target_state = None
        target_condition_data = None
        target_group_data = None

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
    return StepData(
        target_state=target_state,
        target_coupling_lin=target_coupling_lin,
        target_coupling_quad=target_coupling_quad,
        target_condition_data=target_condition_data,
        target_group_data=target_group_data,
        source_state=source_state,
        source_coupling_lin=source_coupling_lin,
        source_coupling_quad=source_coupling_quad,
        source_condition_data=source_condition_data,
        source_group_data=source_group_data,
    )


def write_continuous_cond_cov_to_step_data(
    condition_key: str,
    x: torch.Tensor,
    base_data: MatchedDistributions | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> StepData:
    """Overrides the base data to store the input tensor at the specified key."""
    # ---- 1. No base data provided ----
    if base_data is None:
        condition_dict = {condition_key: x}
        return StepData(target_condition_data=condition_dict)
    else:
        # ---- Retrieve Metadata ----
        step_data: StepData = extract_step_data(base_data, device=device, dtype=dtype)
        target_condition_data: MixedTypeData = step_data.target_condition_data

        # --- Update covariates ----
        updated_condition_data = TorchMixedTypeData.from_mixed_type_data(
            condition_key, x, base_container=target_condition_data
        )
        updated_step_data = replace(step_data, target_condition_data=updated_condition_data)
        print(type(updated_step_data))
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
    """Called from _step_fn - always returns single noise per batch element."""
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
    """Called from _predict.

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
