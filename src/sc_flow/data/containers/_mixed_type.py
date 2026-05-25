from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers._base import BaseData
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._state import StateData

__all__ = ["MixedTypeData"]


@dataclass(frozen=True)
class MixedTypeData(BaseData):
    """Container class for mixed categorical and continuous covariates.

    :param categorical_covariates: Container storing the categorical covariates.
    :type categorical_covariates: class: `CategoricalData | None`

    :param continuous_covariates: Container storing the continuous covariates.
    :type continuous_covariates: class: `BatchMixin | None`
    """

    categorical_covariates: CategoricalData | None = None
    continuous_covariates: BatchMixin | None = None

    def __post_init__(self) -> None:
        # check matching shapes if both are provided
        if self.categorical_covariates is not None and self.continuous_covariates is not None:
            n_obs_cat = self.categorical_covariates.ann_df.shape[0]
            n_obs_cont = len(self.continuous_covariates)
            if n_obs_cat != n_obs_cont:
                msg = (
                    "Shape mismatch between categorical and continuous covariates. "
                    f"Found {n_obs_cat} observations for categorical covariates and "
                    f"{n_obs_cont} observations for the continuous covariates."
                )
                raise ValueError(msg)

        # need at least one condition representation
        if self.categorical_covariates is None and self.continuous_covariates is None:
            msg = f"{self.__class__.__name__} must contain at least one covariate container."
            raise ValueError(msg)

    def __repr__(self) -> str:
        parts = [f"n_obs={len(self)}"]

        if self.categorical_covariates is not None:
            cat = self.categorical_covariates
            cols = list(cat.ann_df.columns)
            parts.append(f"categorical(n_vars={len(cols)}, columns={cols})")
        else:
            parts.append("categorical=None")

        if self.continuous_covariates is not None:
            cont = self.continuous_covariates
            keys = list(cont.mapping.keys())
            spatial_dims = {k: v.shape[1] for k, v in cont.mapping.items()}
            parts.append(f"continuous(keys={keys}, spatial_dims={spatial_dims})")
        else:
            parts.append("continuous=None")

        return f"{self.__class__.__name__}(" + ", ".join(parts)

    def __len__(self) -> int:
        if self.categorical_covariates is not None:
            return len(self.categorical_covariates)

        if self.continuous_covariates is not None:
            return len(self.continuous_covariates)

    def __getitem__(
        self,
        idxs: np.ndarray | slice,
    ) -> "MixedTypeData":
        def _take(e, idxs=idxs):
            e = e[idxs]
            return e

        categorical_covariates = None if self.categorical_covariates is None else self.categorical_covariates[idxs]
        continuous_covariates = None if self.continuous_covariates is None else self.continuous_covariates.apply(_take)
        return self.__class__(
            categorical_covariates=categorical_covariates, continuous_covariates=continuous_covariates
        )

    def extract_reps(self) -> BatchMixin:
        """Extracts the representations for the underlying data."""
        # extract categorical covariates
        if self.categorical_covariates is not None:
            cat_reps = self.categorical_covariates.extract_reps()
        else:
            cat_reps = BatchMixin({})

        # update with continuous covariates
        if self.continuous_covariates is not None:
            return BatchMixin(
                {
                    **cat_reps.mapping,
                    **self.continuous_covariates.mapping,
                }
            )
        # otherwise return only categorical covariates
        return cat_reps

    def view_as_state_data(
        self,
        state_key: str,
    ) -> StateData:
        """Converts a given key to a state data format.

        Only continuous covariates can be viewed as state data.

        :param state_key: The continuous covariate key to view as state data.
        :type state_key: class: str
        """
        # get state raw data from continuous covariates
        if self.continuous_covariates is not None and state_key in self.continuous_covariates.mapping:
            X_state = self.continuous_covariates[state_key]
        elif self.categorical_covariates and state_key in self.categorical_covariates.category_realms:
            raise NotImplementedError("State conversion is not implemented for categorical covariates.")
        else:
            raise KeyError(f"Key {state_key} not found.")

        # check that the dimensions are correct
        if len(X_state.shape) != 2:
            raise ValueError(f"Converted data should have two dimensions, {X_state.shape} found.")
        return StateData(X_state)

    def absorb_state_data(self, state_key: str, state_data: StateData, allow_override: bool = False) -> "MixedTypeData":
        """Absorbs a state data as continuous condition at the specified key.

        :param state_key: The continuous covariate key to add.
        :type state_key: class: str

        :param state_data: The state data container for the new continuous covariates.
        :type state_data: class: `StateData`

        :param allow_override: Whether to allow ovveriding keys when :param: `state_key` is already
            present as a continuous covariate. Defaults to `False`.
        :type allow_override: class: `bool`
        """
        # check that continuous covariates are actually provided
        if self.continuous_covariates is None:
            raise TypeError("Cannot absorb state data: no continuous covariates present.")

        # check that the key doesnt appear already
        if state_key in self.continuous_covariates.mapping and not allow_override:
            raise ValueError(f"Key {state_key} already present in the data, enable override if you are sure.")

        # check that the state data has the correct number of dimensions
        if len(state_data) != len(self):
            raise ValueError(
                f"Number of observations in state_data ({len(state_data)}) does not match the container ({len(self)})."
            )

        # update continuous covariates
        original_mapping = dict(self.continuous_covariates.mapping)
        original_mapping[state_key] = state_data.X
        updated_continuous_covs = BatchMixin(original_mapping)

        return self.__class__(
            categorical_covariates=self.categorical_covariates,
            continuous_covariates=updated_continuous_covs,
        )

    def pop_key(self, key: str) -> "MixedTypeData | None":
        """Removes a key from the condition data.

        Only keys for continous covariates can be removed.

        :param key: The identifier for the continuous covariate to remove.
        :type key: class: `str`
        """
        # throw error if no continuous covariates
        if self.continuous_covariates is None:
            raise TypeError("Cannot remove key: no continuous covariates present.")

        # remove key from categorical data
        if self.categorical_covariates and key in self.categorical_covariates.category_realms:
            raise NotImplementedError("State conversion is not implemented for categorical covariates.")
        elif key in self.continuous_covariates.mapping:
            original_mapping = dict(self.continuous_covariates.mapping)
            original_mapping.pop(key)

            # when no covariates are left, return None
            if not len(original_mapping) and self.categorical_covariates is None:
                return None
            elif not len(original_mapping):
                updated_continuous_covs = None
            else:
                updated_continuous_covs = BatchMixin(original_mapping)
            return self.__class__(
                categorical_covariates=self.categorical_covariates,
                continuous_covariates=updated_continuous_covs,
            )
        else:
            raise KeyError(f"Key {key} not found.")

    @classmethod
    def concat_collection(
        cls,
        collection: "Collection[MixedTypeData]",
    ) -> "MixedTypeData":
        """Concatenates a collection of instances into a single object."""
        # define store for data
        categorical_covariates_list = []
        continuous_covariates_list = []

        # reference settings
        has_categorical_covs = False
        has_continuous_covs = False
        continuous_covs_keys = []

        # iterate over collection
        for idx, element in enumerate(collection):
            # get reference settings from first element
            if idx == 0:
                has_categorical_covs = element.categorical_covariates is not None
                has_continuous_covs = element.continuous_covariates is not None
                if has_continuous_covs:
                    continuous_covs_keys = list(element.continuous_covariates.mapping.keys())

            # ---- Categorical Covariates ----
            # check that the current element has matching settings
            if element.categorical_covariates is not None:
                if not has_categorical_covs:
                    raise ValueError("Trying to concatenate incompatible objects")

                # update store
                categorical_covariates_list.append(element.categorical_covariates)

            # raise error when categorical covariates are missing but required
            elif has_categorical_covs:
                raise ValueError("Trying to concatenate incompatible objects")

            # ---- Continuous Covariates ----
            # check that the current element has matching settings
            if element.continuous_covariates is not None:
                # continuous covariates are required
                if not has_continuous_covs:
                    raise ValueError("Trying to concatenate incompatible objects")

                # check that we have the same keys
                check_sequence_query_against_reference(
                    continuous_covs_keys,
                    list(element.continuous_covariates.mapping.keys()),
                    allow_missing_from_reference=False,
                    allow_missing_from_query=False,
                )

                # update store
                continuous_covariates_list.append(element.continuous_covariates)

            # raise error when continuous covariates are missing but required
            elif has_continuous_covs:
                raise ValueError("Trying to concatenate incompatible objects")

        # concatenate categorical covariates
        if len(categorical_covariates_list) > 0:
            categorical_covariates = CategoricalData.concat_collection(categorical_covariates_list)
        else:
            categorical_covariates = None

        # concatenate continuous covariates
        if len(continuous_covariates_list) > 0:
            mapping = defaultdict(list)

            for element in continuous_covariates_list:
                for key, v in element.mapping.items():
                    mapping[key].append(v)

            mapping = {key: np.concatenate(val, axis=0) for key, val in mapping.items()}
            continuous_covariates = BatchMixin(mapping)

        else:
            continuous_covariates = None

        return cls(
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )

    @property
    def has_continuous_covariates(self) -> bool:
        """Whether the mixed type data entails continuous covariates."""
        return self.continuous_covariates is not None

    @property
    def has_categorical_covariates(self) -> bool:
        """Whether the mixed type data entails categorical covariates."""
        return self.categorical_covariates is not None
