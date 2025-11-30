from collections.abc import Collection, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

import pandas as pd
from anndata import AnnData

from sc_flow._types import MappedArray, TargetCovariatesEncoderCls, TargetCovariatesEncodingId
from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.data._structures import CategoricalData
from sc_flow.data._utils import get_covariates_encoders_from_dict
from sc_flow.data.schemas._base_schema import BaseDataSchema


@dataclass
class GroupsDataSchema(BaseDataSchema):
    """"""  # noqa

    groups: Collection[str] = dc_field(default_factory=lambda: [])
    groups_reps: dict[str, str] = dc_field(default_factory=lambda: {})
    groups_encoding: dict[str, TargetCovariatesEncodingId] = dc_field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """"""  # noqa
        check_sequence_query_against_reference(
            self.groups,
            set(self.groups_reps.keys()).union(set(self.groups_encoding.keys())),
            allow_missing_from_query=False,
            allow_missing_from_reference=False,
        )
        shared_keys = set(self.groups_reps.keys()).intersection(self.groups_encoding.keys())
        if len(shared_keys):
            msg = "Each group column should have only one representation"
            raise ValueError(msg)

    def _verify_groups(self, adata: AnnData) -> None:
        """"""  # noqa
        for group in self.groups:
            self._check_key_found_in_adata_field(adata, "obs", group)

    def _verify_groups_reps(self, adata: AnnData) -> None:
        """"""  # noqa
        for rep in self.groups_reps.values():
            self._check_key_found_in_adata_field(adata, "uns", rep)

    def _verify_schema(self, adata):
        """"""  # noqa
        self._verify_groups(adata)
        self._verify_groups_reps(adata)

    def _get_covs_df(
        self,
        adata: AnnData,
    ) -> pd.DataFrame:
        """"""  # noqa
        return adata.obs.loc[:, self.groups]

    def _get_covs_repr_dict(
        self,
        adata: AnnData,
    ) -> dict[str, MappedArray]:
        """"""  # noqa
        return {col: adata.uns[rep] for col, rep in self.groups_reps.items()}

    def _get_encoders_dict(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        raise NotImplementedError

    def _get_data(self, adata: AnnData) -> CategoricalData:
        """"""  # noqa
        covs_df_dict: pd.DataFrame = self._get_covs_df(adata)
        repr_dict: dict[str, MappedArray] = self._get_covs_repr_dict(adata)
        encoders_dict: Mapping[str, TargetCovariatesEncoderCls] = get_covariates_encoders_from_dict(
            self.groups_encoding, covs_df_dict
        )
        return CategoricalData(
            covs_df_dict,
            repr_dict=repr_dict,
            categorical_encoders=encoders_dict,
        )
