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
        # check that it appears in the covariate identifiers
        if state_key not in self.covariates_keys:
            raise KeyError(f"Key {state_key} not found in self.covariate_keys")

        # get state raw data from continuous covariates
        if state_key in self.continuous_covariates.mapping:
            X_state = self.continuous_covariates[state_key]
        else:  # get them from the categorical covariates
            categorical_reps = self.categorical_covariates.extract_reps()
            X_state = categorical_reps[state_key]  # N, 1, D

            # check shape
            if len(X_state.shape) != 3:
                raise ValueError(f"Categorical data expected to have three dimensions, {X_state.shape} found.")
            if X_state[1] != 1:
                raise ValueError("Conversion of combinatorial categorical data is not supported for the moment.")

            # get single element of the dimension
            X_state = X_state[:, 0, :]

        # check that the dimensions are correct
        if len(X_state) != 2:
            raise ValueError(f"Converted data should have two dimensions, {X_state.shape} found.")
        return StateData(X_state)

    @property
    def covariates_keys(self) -> list[str]:
        """Returns the list of all covariate identifiers."""
        # define store for all covariates
        covariate_keys = []

        # get keys from continuous data
        if self.continuous_covariates is not None:
            keys = list(self.continuous_covariates.mapping.keys())
            covariate_keys.extend(keys)

        # get keys from categorical data
        if self.categorical_covariates is not None:
            keys.extend(self.categorical_covariates.category_realms)
        return covariate_keys
