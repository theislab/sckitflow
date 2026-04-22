from collections import defaultdict
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sc_flow._types import TargetCovariatesEncoderCls
from sc_flow.data._mixins import BatchMixin, MappedArray
from sc_flow.data._utils import convert_to_categorical_in_place
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
    repr_dict: Mapping[str, MappedArray] = dc_field(default_factory=dict)
    categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] = dc_field(default_factory=dict)
    categorical_reps_map: Mapping[str, str] = dc_field(default_factory=dict)

    def __post_init__(self):
        # check that the representations are provided
        for col in self.ann_df.columns:
            # each column needs to appear in the categorical reps map
            if col not in self.categorical_reps_map:
                raise KeyError(f"Column {col} is not mapped to any realm.")
            col_realm = self.categorical_reps_map[col]

            # we need an associated representation for the corresponding realms
            if col_realm not in self.repr_dict and col_realm not in self.categorical_encoders:
                raise KeyError(f"No representation found for column {col} associated to realm {col_realm}.")

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
        ann_df = self.ann_df.iloc[idxs]
        return self.__class__(
            ann_df,
            repr_dict=self.repr_dict,
            categorical_encoders=self.categorical_encoders,
            categorical_reps_map=self.categorical_reps_map,
        )

    def __len__(self) -> int:
        return self.ann_df.shape[0]

    def _extract_col_repr(
        self,
        col: str,
        realm_repr_dict: dict[Hashable, np.ndarray],
    ) -> np.ndarray:
        col_values = self.ann_df[col].map(lambda e: realm_repr_dict[e].reshape(1, -1))
        return np.concatenate(col_values.tolist(), axis=0)

    def _encode_col(
        self,
        col: str,
        encoder: TargetCovariatesEncoderCls,
    ) -> np.ndarray:
        col_values = self.ann_df[col].values.reshape(-1, 1)
        tranformed_values = encoder.transform(col_values)
        if isinstance(tranformed_values, csr_matrix):
            return tranformed_values.toarray()
        return tranformed_values

    def extract_reps(self) -> BatchMixin:
        """Extracts the representations for the underlying data."""
        # dictionary to store the data
        data_dict = defaultdict(list)

        # iterate over annotation df columns
        for col in self.ann_df.columns:
            realm = self.categorical_reps_map[col]
            if realm in self.repr_dict:
                realm_repr_dict = self.repr_dict[realm]
                col_repr = self._extract_col_repr(col, realm_repr_dict)
            elif realm in self.categorical_encoders:
                realm_encoder = self.categorical_encoders[realm]
                col_repr = self._encode_col(col, realm_encoder)
            data_dict[realm].append(col_repr)

        # stack arrays
        data_dict = {k: np.stack(v, axis=-2) for k, v in data_dict.items()}
        return BatchMixin(mapping=data_dict)

    @classmethod
    def from_pandas(
        cls,
        ann_df: pd.DataFrame,
        repr_dict: Mapping[str, MappedArray] | None = None,
        categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] | None = None,
        inplace: bool = False,
    ) -> "CategoricalData":
        """Create a CategoricalData object from a pandas DataFrame.

        TODO: document properly but most importantly note that for better performance it is recommended to pass the data in place.
        """
        if not inplace:
            ann_df = ann_df.copy()
        convert_to_categorical_in_place(ann_df, ann_df.columns)
        return cls(
            ann_df,
            repr_dict={} if repr_dict is None else repr_dict,
            categorical_encoders={} if categorical_encoders is None else categorical_encoders,
        )
