from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from sc_flow.data.containers._base import BaseData
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.data.containers._state import StateData

__all__ = ["DistributionData"]


@dataclass(frozen=True)
class DistributionData(BaseData):
    """Container class for distribution data.

    :param state_data: Container storing the state data.
    :type state_data: class: `StateData`

    :param target_data: Container storing the target data.
        Defaults to `None`.
    :type target_data: class: `MixedTypeData | None`

    :param condition_data: Container storing the condition data.
        Defaults to `None`.
    :type condition_data: class: `MixedTypeData | None`

    :param groups_data: Container storing the group data.
        Defaults to `None`.
    :type groups_data: class: `CategoricalData | None`
    """

    state_data: StateData
    target_data: MixedTypeData | None = None
    condition_data: MixedTypeData | None = None
    groups_data: CategoricalData | None = None
    source_coupling_data: CouplingData | None = None
    target_coupling_data: CouplingData | None = None

    def __post_init__(self) -> None:
        if self.target_data is not None:
            self.state_data._assert_same_n_obs(self.target_data)
        if self.condition_data is not None:
            self.state_data._assert_same_n_obs(self.condition_data)
        if self.groups_data is not None:
            self.state_data._assert_same_n_obs(self.groups_data)
        if self.source_coupling_data is not None:
            self.state_data._assert_same_n_obs(self.source_coupling_data)
        if self.target_coupling_data is not None:
            self.state_data._assert_same_n_obs(self.target_coupling_data)

    def __repr__(self) -> str:
        parts = [f"\n * n_obs={self.n_obs}"]
        to_plot = [
            ("state", self.state_data),
            ("target", self.target_data),
            ("condition", self.condition_data),
            ("groups", self.groups_data),
            ("source_coupling", self.source_coupling_data),
            ("target_coupling", self.target_coupling_data),
        ]
        for prefix, comp in to_plot:
            if comp is None:
                parts.append(f"{prefix}={comp}")
            else:
                parts.append(f"{prefix}={comp!r}")
        return f"{self.__class__.__name__}:" + "\n ".join(parts)

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "DistributionData":
        state_data = self.state_data.slice_with_array(idxs)
        target_data = None if self.target_data is None else self.target_data.slice_with_array(idxs)
        condition_data = None if self.condition_data is None else self.condition_data.slice_with_array(idxs)
        groups_data = None if self.groups_data is None else self.groups_data.slice_with_array(idxs)
        source_coupling_data = (
            None if self.source_coupling_data is None else self.source_coupling_data.slice_with_array(idxs)
        )
        target_coupling_data = (
            None if self.target_coupling_data is None else self.target_coupling_data.slice_with_array(idxs)
        )
        return self.__class__(
            state_data,
            target_data=target_data,
            condition_data=condition_data,
            groups_data=groups_data,
            source_coupling_data=source_coupling_data,
            target_coupling_data=target_coupling_data,
        )

    @property
    def ann_df(self) -> pd.DataFrame:
        """Returns the annotation data frame constructed jointly from condition and groups data."""
        dfs = []
        if self.condition_data and self.condition_data.categorical_covariates:
            dfs.append(self.condition_data.categorical_covariates.ann_df)
        if self.groups_data:
            dfs.append(self.groups_data.ann_df)
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    @property
    def n_obs(self) -> int:
        return self.state_data.n_obs

    @cached_property
    def index(self) -> pd.Index:
        """"""  # noqa
        return self.ann_df.index
