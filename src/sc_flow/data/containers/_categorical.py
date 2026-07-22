from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import pandas as pd

from sc_flow.data._encoders import Encoder, one_hot
from sc_flow.data._mixins import BatchMixin
from sc_flow.data._utils import convert_to_categorical_in_place
from sc_flow.data.containers._base import BaseData

__all__ = ["CategoricalData"]


@dataclass(frozen=True)
class CategoricalData(BaseData):

    ann_df: pd.DataFrame
    encoders: Mapping[str, Encoder] = dc_field(default_factory=dict)
    categorical_reps_map: Mapping[str, str] = dc_field(default_factory=dict)

    def __post_init__(self):
        for col in self.ann_df.columns:
            if col not in self.categorical_reps_map:
                raise KeyError(f"Column {col} is not mapped to any realm.")
            col_realm = self.categorical_reps_map[col]
            if col_realm not in self.encoders:
                raise KeyError(f"No encoder found for column {col} associated to realm {col_realm}.")

    def __repr__(self) -> str:
        n_obs, n_vars = self.ann_df.shape
        cols = list(self.ann_df.columns)
        return (
            f"{self.__class__.__name__}("
            f"n_obs={n_obs}, "
            f"n_vars={n_vars}, "
            f"columns={cols}, "
            f"encoder_realms={list(self.encoders)})"
        )

    def extract_reps(self) -> BatchMixin:
        data_dict = defaultdict(list)
        for col in self.ann_df.columns:
            realm = self.categorical_reps_map[col]
            col_repr = np.asarray(self.encoders[realm].transform(self.ann_df[col].to_numpy()))
            data_dict[realm].append(col_repr)
        data_dict = {k: np.stack(v, axis=-2) for k, v in data_dict.items()}
        return BatchMixin(mapping=data_dict)

    @classmethod
    def from_pandas(
        cls,
        ann_df: pd.DataFrame,
        encoders: Mapping[str, Encoder] | None = None,
        inplace: bool = False,
        categorical_reps_map: Mapping[str, str] | None = None,
    ) -> "CategoricalData":
        if not inplace:
            ann_df = ann_df.copy()
        convert_to_categorical_in_place(ann_df, ann_df.columns)

        categorical_reps_map = (
            {col: col for col in ann_df.columns} if categorical_reps_map is None else categorical_reps_map
        )
        encoders = dict(encoders) if encoders is not None else {}

        # default encoder: for a realm with none, fit a one-hot on the union of its COLUMNS' values (a
        # realm may group several columns, e.g. drug1/drug2 — the combination slots), so the encoder
        # learns all categories across slots; `extract_reps` then applies it per column.
        for realm in set(categorical_reps_map.values()):
            if realm not in encoders:
                realm_cols = [col for col, r in categorical_reps_map.items() if r == realm]
                union = ann_df.loc[:, realm_cols].to_numpy().reshape(-1)
                encoders[realm] = one_hot().fit(union)

        return cls(ann_df, encoders=encoders, categorical_reps_map=categorical_reps_map)

    @property
    def category_realms(self) -> list[str]:
        return list(set(self.categorical_reps_map.values()))
