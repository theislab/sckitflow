from collections.abc import Collection
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from sc_flow.data._mixins import BatchMixin
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

    :param response_data: Container storing the target data.
        Defaults to `None`.
    :type response_data: class: `MixedTypeData | None`

    :param condition_data: Container storing the condition data.
        Defaults to `None`.
    :type condition_data: class: `MixedTypeData | None`

    :param groups_data: Container storing the group data.
        Defaults to `None`.
    :type groups_data: class: `CategoricalData | None`

    :param source_coupling_data: The representation of the data
        as source distribution, needed for the matching.
        Defaults to `None`.
    :type source_coupling_data: class: CouplingData | Non

    :param target_coupling_data: The representation of the data
        as target distribution, needed for the matching.
        Defaults to `None`.
    :type target_coupling_data: class: CouplingData | Non
    """

    state_data: StateData
    response_data: MixedTypeData | None = None
    condition_data: MixedTypeData | None = None
    groups_data: CategoricalData | None = None
    source_coupling_data: CouplingData | None = None
    target_coupling_data: CouplingData | None = None

    def __post_init__(self) -> None:
        if self.response_data is not None:
            self.state_data.assert_same_len(self.response_data)
        if self.condition_data is not None:
            self.state_data.assert_same_len(self.condition_data)
        if self.groups_data is not None:
            self.state_data.assert_same_len(self.groups_data)
        if self.source_coupling_data is not None:
            self.state_data.assert_same_len(self.source_coupling_data)
        if self.target_coupling_data is not None:
            self.state_data.assert_same_len(self.target_coupling_data)

    def __repr__(self) -> str:
        parts = [f"\n * n_obs={len(self)}"]
        to_plot = [
            ("state", self.state_data),
            ("target", self.response_data),
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

    def __len__(self) -> int:
        return len(self.state_data)

    def __getitem__(
        self,
        idxs: np.ndarray | slice,
    ) -> "DistributionData":
        state_data = self.state_data[idxs]
        response_data = None if self.response_data is None else self.response_data[idxs]
        condition_data = None if self.condition_data is None else self.condition_data[idxs]
        groups_data = None if self.groups_data is None else self.groups_data[idxs]
        source_coupling_data = None if self.source_coupling_data is None else self.source_coupling_data[idxs]
        target_coupling_data = None if self.target_coupling_data is None else self.target_coupling_data[idxs]
        return self.__class__(
            state_data,
            response_data=response_data,
            condition_data=condition_data,
            groups_data=groups_data,
            source_coupling_data=source_coupling_data,
            target_coupling_data=target_coupling_data,
        )

    def get_metadata_dict(self) -> BatchMixin:
        """Gets the metadata associated to the current distribution."""
        # extract condition data
        if self.condition_data is not None:
            condition_reps = self.condition_data.extract_reps()
        else:
            condition_reps = BatchMixin({})

        # extract group data
        if self.groups_data is not None:
            groups_reps = self.groups_data.extract_reps()
        else:
            groups_reps = BatchMixin({})

        # extract response data
        if self.response_data is not None:
            response_reps = self.response_data.extract_reps()
        else:
            response_reps = BatchMixin({})

        return BatchMixin(
            {
                "condition": condition_reps,
                "groups": groups_reps,
                "response_reps": response_reps,
            }
        )

    def view_on_condition_space(self, state_key: str):
        """Views the current distribution as being defined on the condition space.

        Only continuous condition covariates can be modeled as states.

        :param state_key: The identifier for the condition covariate to model as state.
        :type state_key: class: `str`
        """
        # check that conditions are defined
        if self.condition_data is None:
            raise TypeError("Cannot view as condition: no condition data provided.")

        # get state data from condition and update condition
        cond_state_data = self.condition_data.view_as_state_data(state_key)
        updated_condition_data = self.condition_data.pop_key(state_key)

        # update coupling data and return
        source_coupling_data = CouplingData.init_from_state_data(cond_state_data)
        target_coupling_data = CouplingData.init_from_state_data(cond_state_data)
        return self.__class__(
            cond_state_data,
            response_data=self.response_data,
            condition_data=updated_condition_data,
            groups_data=self.groups_data,
            source_coupling_data=source_coupling_data,
            target_coupling_data=target_coupling_data,
        )

    @property
    def is_sorted(self) -> bool:
        """Whether the data is sorted lexicographically by annotation columns."""
        df = self.ann_df
        if df.shape[1] == 0:
            return True
        sort_keys = [df.iloc[:, i] for i in reversed(range(df.shape[1]))]
        order = np.lexsort(sort_keys)
        return bool(np.all(order[1:] >= order[:-1]))

    def sort(self) -> "DistributionData":
        """"""  # noqa
        df = self.ann_df
        if df.shape[1] == 0:
            return self
        sort_keys = [df.iloc[:, i] for i in reversed(range(df.shape[1]))]
        sorted_idxs = np.lexsort(sort_keys)
        return self[sorted_idxs]

    @cached_property
    def ann_df(self) -> pd.DataFrame:
        """Returns the annotation data frame from groups then conditions.

        Column order matches the hierarchy (groups first, conditions second)
        so that lexsort-based ``is_sorted`` / ``sort()`` agree with
        ``DataManager.sort_adata``.
        """
        dfs = []
        if self.groups_data:
            dfs.append(self.groups_data.ann_df)
        if self.condition_data and self.condition_data.categorical_covariates:
            dfs.append(self.condition_data.categorical_covariates.ann_df)

        if len(dfs) == 0:
            return pd.DataFrame()
        res = pd.DataFrame(
            {col: df[col].values for df in dfs for col in df.columns},
            index=dfs[0].index,
        )
        return res

    @cached_property
    def index(self) -> pd.Index:
        """Returns the index of the annotation data frame."""
        return self.ann_df.index

    @classmethod
    def concat_collection(
        cls,
        collection: "Collection[DistributionData]",
    ) -> "DistributionData":
        """Concatenates a collection of instances into a single object."""
        # define store for data
        state_data_list = []
        response_data_list = []
        condition_data_list = []
        groups_data_list = []
        source_coupling_data_list = []
        target_coupling_data_list = []

        # define base settings
        has_response_data = False
        has_condition_data = False
        has_groups_data = False
        has_source_coupling_data = False
        has_target_coupling_data = False

        # iterate over elements
        for idx, element in enumerate(collection):
            # get reference settings from first element
            if idx == 0:
                has_response_data = element.response_data is not None
                has_condition_data = element.condition_data is not None
                has_groups_data = element.groups_data is not None
                has_source_coupling_data = element.source_coupling_data is not None
                has_target_coupling_data = element.target_coupling_data is not None

            # --- State data ---
            state_data_list.append(element.state_data)

            # --- Response data ---
            if has_response_data:
                if element.response_data is None:
                    raise ValueError("Trying to concatenate incompatible objects.")
                response_data_list.append(element.response_data)
            elif element.response_data is not None:
                raise ValueError("Trying to concatenate incompatible objects.")

            # --- Condition data ---
            if has_condition_data:
                if element.condition_data is None:
                    raise ValueError("Trying to concatenate incompatible objects.")
                condition_data_list.append(element.condition_data)
            elif element.condition_data is not None:
                raise ValueError("Trying to concatenate incompatible objects.")

            # --- Groups data ---
            if has_groups_data:
                if element.groups_data is None:
                    raise ValueError("Trying to concatenate incompatible objects.")
                groups_data_list.append(element.groups_data)
            elif element.groups_data is not None:
                raise ValueError("Trying to concatenate incompatible objects.")

            # --- Source coupling data ---
            if has_source_coupling_data:
                if element.source_coupling_data is None:
                    raise ValueError("Trying to concatenate incompatible objects.")
                source_coupling_data_list.append(element.source_coupling_data)
            elif element.source_coupling_data is not None:
                raise ValueError("Trying to concatenate incompatible objects.")

            # --- Target coupling data ---
            if has_target_coupling_data:
                if element.target_coupling_data is None:
                    raise ValueError("Trying to concatenate incompatible objects.")
                target_coupling_data_list.append(element.target_coupling_data)
            elif element.target_coupling_data is not None:
                raise ValueError("Trying to concatenate incompatible objects.")

        # ---- Concatenate ----
        # state data
        state_data = StateData.concat_collection(state_data_list)

        # response data
        if len(response_data_list) > 0:
            response_data = MixedTypeData.concat_collection(response_data_list)
        else:
            response_data = None

        # condition data
        if len(condition_data_list) > 0:
            condition_data = MixedTypeData.concat_collection(condition_data_list)
        else:
            condition_data = None

        # groups data
        if len(groups_data_list) > 0:
            groups_data = CategoricalData.concat_collection(groups_data_list)
        else:
            groups_data = None

        # source coupling data
        if len(source_coupling_data_list) > 0:
            source_coupling_data = CouplingData.concat_collection(source_coupling_data_list)
        else:
            source_coupling_data = None

        # target coupling data
        if len(target_coupling_data_list) > 0:
            target_coupling_data = CouplingData.concat_collection(target_coupling_data_list)
        else:
            target_coupling_data = None

        return cls(
            state_data,
            response_data=response_data,
            condition_data=condition_data,
            groups_data=groups_data,
            source_coupling_data=source_coupling_data,
            target_coupling_data=target_coupling_data,
        )

    @property
    def has_continuous_condition_covariates(self) -> bool:
        """Whether the distribution entails continuous condition covariates."""
        if self.condition_data is not None:
            return self.condition_data.has_continuous_covariates
        return False

    @property
    def has_categorical_condition_covariates(self) -> bool:
        """Whether the distribution entails categorical covariates."""
        if self.condition_data is not None:
            return self.condition_data.has_categorical_covariates
        return False
