from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from types import MappingProxyType

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray, TargetCovariatesEncoderCls
from sc_flow.data.containers._base import BaseData

__all__ = ["CategoricalData"]


@dataclass(frozen=True)
class CategoricalData(BaseData):
    """Container class for categorical data.

    Any categorical data is defined over a set of column, stored in a :class: `pandas.DataFrame`.
    There are two possible ways to represent categorical variables using this container.
    The first one is to pass pre-computed representations, with the :param repr_dict: argument.
    Otherwise it is possible to specify some pre-defined encoders to transform the categorical
    values stored in the data frame into suitable representations.

    :param ann_df: The data frame storing the original values.
    :type ann_df: class: `pandas.DataFrame`

    :param repr_dict: Dictionary storing the pre-computed representations.
    :type repr_dict: class: `dict[str, MappedArray]`

    :param categorical_encoders: Mapping storing the covariate encoders class.
    :type categorical_encoders: Mapping[str, TargetCovariatesEncoderCls]
    """

    ann_df: pd.DataFrame
    repr_dict: Mapping[str, MappedArray] = dc_field(default_factory=lambda: MappingProxyType({}))
    categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] = dc_field(
        default_factory=lambda: MappingProxyType({})
    )

    def __repr__(self) -> str:
        n_obs, n_vars = self.ann_df.shape
        cols = list(self.ann_df.columns)

        repr_keys = list(self.repr_dict.keys())
        encoder_keys = list(self.categorical_encoders.keys())

        return (
            f"{self.__class__.__name__}("
            f"n_obs={n_obs}, "
            f"n_vars={n_vars}, "
            f"columns={cols}, "
            f"repr_dict_keys={repr_keys}, "
            f"categorical_encoders_keys={encoder_keys})"
        )

    def __getitem__(
        self,
        idxs: np.ndarray | slice,
    ) -> "CategoricalData":
        ann_df = self.ann_df.take(idxs)
        return self.__class__(ann_df, repr_dict=self.repr_dict, categorical_encoders=self.categorical_encoders)

    def __len__(self) -> int:
        return self.ann_df.shape[0]
