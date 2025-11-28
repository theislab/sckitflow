from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from sc_flow._types import MappedArray
from sc_flow.data._mixins import BatchMixin

__all__ = [
    "CategoricalDataContainer",
    "StateDataContainer",
    "TargetDataContainer",
    "ConditionDataContainer",
    "DistributionContainer",
    "MatchedDistributionsContainer",
]


@dataclass
class BaseDataContainer:
    """"""  # noqa

    pass


@dataclass
class CategoricalDataContainer(BaseDataContainer):
    """"""  # noqa

    column_values: Collection[tuple[str]]
    repr_dict: MappedArray


@dataclass
class CombinatorialCategoricalDataContainer(CategoricalDataContainer):
    """"""  # noqa


@dataclass
class StateDataContainer(BaseDataContainer):
    """"""  # noqa

    X: np.ndarray


@dataclass
class TargetDataContainer(BaseDataContainer):
    """"""  # noqa

    categorical_covariates: CategoricalDataContainer | None = None
    continuous_covariates: BatchMixin | None = None


@dataclass
class ConditionDataContainer(BaseDataContainer):
    """"""  # noqa

    condition_reps: CombinatorialCategoricalDataContainer | None = None
    condition_covariates: BatchMixin | None = None


@dataclass
class DistributionContainer(BaseDataContainer):
    """"""  # noqa

    state_data: StateDataContainer
    target_data: TargetDataContainer | None = None
    condition_data: ConditionDataContainer | None = None


@dataclass
class MatchedDistributionsContainer(BaseDataContainer):
    """"""  # noqa

    target_distr: DistributionContainer
    source_distr: DistributionContainer | None = None

    @property
    def has_source(self) -> bool:
        """"""  # noqa
        return self.source_distr is not None
