from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NewType

import numpy as np

from sc_flow._types import MappedArray

## TODO: remove when felix writes this stuff
CouplingData = NewType("CouplingData", MappedArray)


@dataclass
class CombinationData:
    """"""  # noqa

    columns: Sequence[str]
    comb_id_to_unique_values: dict[int, Sequence[Any]]
    comb_id_to_indices: dict[int, np.ndarray]

    def __post_init__(
        self,
    ) -> None:
        """"""  # noqa
        for comb_id, unique_value in self.comb_id_to_unique_values.items():
            if len(unique_value) != len(self.columns):
                msg = (
                    f"Found incorrect unique values for column {comb_id}."
                    f"Expected {len(self.columns)}, found {len(unique_value)}."
                )
                raise ValueError(msg)


@dataclass
class TargetData:
    """"""  # noqa

    categorical_target_data: MappedArray
    continuous_target_data: MappedArray


@dataclass
class StateData:
    """"""  # noqa

    cell_states: np.ndarray


@dataclass
class ConditionData:
    """"""  # noqa

    condition_reps: MappedArray
    condition_covariates: MappedArray | None

    def __post_init__(
        self,
    ) -> None:
        """"""  # noqa
        # ensuring representation has a leading singleton dim
        for condition_id, condition_rep in self.condition_reps.items():
            if condition_rep.ndim == 1:
                condition_rep = condition_rep[None]
            if condition_rep.ndim != 2:
                msg = (
                    f"Representation {condition_rep} for condition {condition_id}"
                    f"has incorrect shape {condition_rep.shape}."
                )
                raise ValueError(msg)
            self.condition_reps[condition_id] = condition_rep
