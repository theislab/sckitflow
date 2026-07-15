from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from sc_flow.backends.torch._utils import to_torch_tensor
from sc_flow.data import mixins
from sc_flow.data.containers import MixedTypeData

__all__ = ["MappedTensor", "TensorMixin", "TorchMixedTypeData"]


@dataclass(frozen=True)
class MappedTensor(mixins.MappedTree):
    """"""  # noqa

    _REQUIRED_VALUE_TYPE: ClassVar[type[Any]] = torch.Tensor


@dataclass(frozen=True)
class TensorMixin(mixins.BatchMixin[str, torch.Tensor]):
    """"""  # noqa

    _REQUIRED_VALUE_TYPE: ClassVar[type[Any]] = torch.Tensor


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
