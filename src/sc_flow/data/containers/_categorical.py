from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import pandas as pd

from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.data._encoders import Encoder, one_hot
from sc_flow.data._mixins import BatchMixin
from sc_flow.data._utils import convert_to_categorical_in_place
from sc_flow.data.containers._base import BaseData

__all__ = ["CategoricalData"]


@dataclass(frozen=True)
class CategoricalData(BaseData):
    """Container for categorical data — columns plus one :class:`~sc_flow.data._encoders.Encoder` per realm.

    Categorical data is a set of columns (a :class:`pandas.DataFrame`) together with, for each *realm*
    (a group of columns sharing a representation, e.g. the combination slots ``drug1``/``drug2``), a
    single fitted encoder. After schema-generalization Change 2 there is no ``reps`` vs ``encoding``
    split: a ``.uns`` lookup is just a :class:`~sc_flow.data._encoders.Lookup` encoder, so every realm
    has exactly one encoder regardless of whether its parameters came from ``.uns`` or from the data.

    :param ann_df: The data frame storing the original categorical values.
    :param encoders: Mapping ``{realm: Encoder}`` — one fitted encoder per realm.
    :param categorical_reps_map: Mapping ``{column: realm}``.
    """

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

    def __getitem__(
        self,
        idxs: np.ndarray | slice,
    ) -> "CategoricalData":
        ann_df = self.ann_df.iloc[idxs]
        return self.__class__(
            ann_df,
            encoders=self.encoders,
            categorical_reps_map=self.categorical_reps_map,
        )

    def __len__(self) -> int:
        return self.ann_df.shape[0]

    def extract_reps(self) -> BatchMixin:
        """Encode each column via its realm's encoder and stack per realm → ``{realm: (n, slots, dim)}``."""
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
        """Create a CategoricalData from a DataFrame, defaulting any un-encoded realm to a fitted one-hot.

        For better performance pass the data in place. Realms without an entry in ``encoders`` get a
        :func:`~sc_flow.data._encoders.one_hot` fit on the union of that realm's columns' values.
        """
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
        """Returns the category realms associated to the object."""
        return list(set(self.categorical_reps_map.values()))

    @classmethod
    def concat_collection(
        cls,
        collection: "Collection[CategoricalData]",
    ) -> "CategoricalData":
        """Concatenates a collection of instances into a single object."""
        ann_dfs_list = []
        encoders = {}
        categorical_reps_map = {}
        ref_cols = None
        for idx, element in enumerate(collection):
            if idx == 0:
                ref_cols = element.ann_df.columns
            check_sequence_query_against_reference(
                element.ann_df.columns,
                ref_cols,
                allow_missing_from_reference=False,
                allow_missing_from_query=False,
            )
            ann_dfs_list.append(element.ann_df)
            encoders.update(element.encoders)
            categorical_reps_map.update(element.categorical_reps_map)

        ann_df = pd.concat(ann_dfs_list, axis=0)
        return cls(ann_df, encoders=encoders, categorical_reps_map=categorical_reps_map)
