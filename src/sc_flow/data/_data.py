from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, NewType

import numpy as np

from sc_flow._types import MappedArray

## TODO: remove when felix writes this stuff
CouplingData = NewType("CouplingData", MappedArray)


@dataclass
class CombinationData:
    """Data for combinations over multiple columns.

    :param columns: The columns over which the combinations are computed.
    :type columns: class: `Sequence[str]`

    :param comb_id_to_unique_values: Mapping from combination integer identifier to
        actual column values.
    :type comb_id_to_unique_values: class: `dict[int, Sequence[Any]]`

    :param comb_id_to_unique_values: Mapping from combination integer identifier to
        the row indices corresponding to such combination.
    :type comb_id_to_unique_values: class: `dict[int, Sequence[Any]]`
    """

    columns: Sequence[str]
    comb_values_to_indices: dict[int, np.ndarray]

    def __post_init__(
        self,
    ) -> None:
        """Verifies that each unique values has the same length as the specified columns."""
        for unique_value, _ in self.comb_values_to_indices.items():
            if len(unique_value) != len(self.columns):
                msg = (
                    f"Found incorrect unique values for column {unique_value}."
                    f"Expected {len(self.columns)}, found {len(unique_value)}."
                )
                raise ValueError(msg)

    @property
    def unique_values_to_comb_id(
        self,
    ) -> dict[tuple[Any], int]:
        """"""  # noqa
        return {v: k for k, v in self.comb_id_to_unique_values}


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

    condition_reps: dict[Any, MappedArray]
    condition_covariates: MappedArray

    def __post_init__(
        self,
    ) -> None:
        """"""  # noqa
        # ensuring representation has a leading singleton dim
        for condition_id, condition_rep_dict in self.condition_reps.items():
            for cond_value_id, cond_value_rep in condition_rep_dict.items():
                if cond_value_rep.ndim == 1:
                    cond_value_rep = cond_value_rep[None]
                if cond_value_rep.ndim != 2:
                    msg = (
                        f"Representation {cond_value_rep} for condition {condition_id}"
                        f"has incorrect shape {cond_value_rep.shape}."
                    )
                    raise ValueError(msg)
                condition_rep_dict[cond_value_id] = cond_value_rep
            self.condition_reps[condition_id] = condition_rep_dict


@dataclass
class UnconditionalDataset:
    """"""  # noqa

    cell_states_data: StateData
    target_data: TargetData = dc_field(default_factory=lambda: {})
    cell_covariates_data: dict[str, MappedArray] = dc_field(default_factory=lambda: {})


@dataclass
class ConditionalDataset(UnconditionalDataset):
    """"""  # noqa

    condition_data: ConditionData | None = None
    group_to_populations_indices: dict[tuple[Any], MappedArray] = dc_field(default_factory=lambda: {})

    @classmethod
    def init_from_unconditional(
        cls,
        unconditional_dataset: UnconditionalDataset,
        condition_data: TargetData | None = None,
        group_to_populations_indices: dict[tuple[Any], MappedArray] | None = None,
    ) -> "ConditionalDataset":
        """"""  # noqa

        return ConditionalDataset(
            unconditional_dataset.cell_states_data,
            target_data={} if unconditional_dataset.target_data is None else unconditional_dataset.target_data,
            group_to_populations_indices={} if group_to_populations_indices is None else group_to_populations_indices,
            condition_data=condition_data,
        )
