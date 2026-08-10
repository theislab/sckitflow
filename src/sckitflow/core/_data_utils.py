from typing import Any

import numpy as np
import torch

from sckitflow.core._types import StepData, TensorMixin, TNoiseSamplerFn
from sckitflow.core._utils import to_torch_tensor
from sckitflow.data import mixins
from sckitflow.data.containers import CategoricalData, MixedTypeData

__all__ = [
    "batchmixin_to_torch",
    "get_tensor_dict_from_data",
    "subscript_step_data",
    "expand_conditioning",
    "prepare_latent_train",
    "prepare_latent_inference",
    "TorchMixedTypeData",
]


def batchmixin_to_torch(
    data: mixins.BatchMixin | dict[str, np.ndarray | torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Converts the elements of a mixin to torch tensors."""
    if data is None:
        return {}
    data_dict = data.mapping if isinstance(data, mixins.BatchMixin) else data
    if not isinstance(data_dict, dict):
        raise ValueError(f"Data dictionary of the wrong type, expected `dict` got {type(data_dict)}.")
    return {k: to_torch_tensor(v) for k, v in data_dict.items()}


def get_tensor_dict_from_data(
    data: MixedTypeData | CategoricalData | dict[str, np.ndarray | torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Extracts torch tensors from the distribution data."""
    if data is None:
        return {}
    data_dict = data.extract_reps() if isinstance(data, MixedTypeData | CategoricalData) else data
    return batchmixin_to_torch(data_dict)


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
) -> torch.Tensor:
    """Called from compute_loss - always returns single noise per batch element."""
    if source is None or generate_from_noise:
        return noise_sampler(target.shape)
    return source


def prepare_latent_inference(
    source: torch.Tensor | None,
    target_reference: torch.Tensor,
    noise_sampler: TNoiseSamplerFn,
    n_samples: int | None = None,
    generate_from_noise: bool = False,
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
        return noise_sampler(shape)
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

        # ---- 3. Handle data type for other covariates and initialize mixin ----
        mapping = {k: to_torch_tensor(v) for k, v in mapping.items()}
        mixin = TensorMixin(mapping)

        # ---- 4. Initialize class ----
        return cls(categorical_covariates=categorical_covariates, continuous_covariates=mixin)
