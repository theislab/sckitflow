from dataclasses import dataclass

import numpy as np

from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers._base import BaseData
from sc_flow.data.containers._categorical import CategoricalData

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
        msg = f"{self.__class__.__name__} must contain at least one covariate container."
        raise ValueError(msg)

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
