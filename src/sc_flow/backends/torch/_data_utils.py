from dataclasses import dataclass
from typing import Any, ClassVar

import torch

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
