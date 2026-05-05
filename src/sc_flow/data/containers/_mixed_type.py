from dataclasses import dataclass

import numpy as np

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
        """Converts a given key to a state data format."""
        # get state raw data from continuous covariates
        if state_key in self.continuous_covariates.mapping:
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
        """Absorbs a state data as continuous condition at the specified key."""
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
        """Removes a key from the condition data."""
        # throw errror if no continuous covariates
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

            updated_continuous_covs = BatchMixin(original_mapping)
            return self.__class__(
                categorical_covariates=self.categorical_covariates,
                continuous_covariates=updated_continuous_covs,
            )
        else:
            raise KeyError(f"Key {key} not found.")
