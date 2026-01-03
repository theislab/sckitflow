from collections.abc import Collection, Mapping

import pandas as pd
from anndata import AnnData

from sc_flow._types import MappedArray, TargetCovariatesEncoderCls, TargetCovariatesEncodingId
from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.data._structures import CategoricalData
from sc_flow.data._utils import get_covariates_encoders_from_dict
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["GroupsDataSchema"]


class GroupsDataSchema(StrictDataSchema):
    """"""  # noqa

    def __init__(
        self,
        groups: Collection[str] | None = None,
        groups_reps: dict[str, str] | None = None,
        groups_encoding: dict[str, TargetCovariatesEncodingId] | None = None,
    ) -> None:
        """"""  # noqa
        self._groups = [] if groups is None else groups
        self._groups_reps = {} if groups_reps is None else groups_reps
        self._groups_encoding = {} if groups_encoding is None else groups_encoding
        super().__init__()

    def _verify_args(self) -> None:
        """"""  # noqa
        check_sequence_query_against_reference(
            self._groups,
            set(self._groups_reps.keys()).union(set(self._groups_encoding.keys())),
            allow_missing_from_query=False,
            allow_missing_from_reference=False,
        )
        shared_keys = set(self._groups_reps.keys()).intersection(self._groups_encoding.keys())
        if len(shared_keys):
            msg = "Each group column should have only one representation"
            raise ValueError(msg)
        self._check_is_valid_encoder_id_dict(self._groups_encoding)

    def _verify_groups(self, adata: AnnData) -> None:
        """"""  # noqa
        for group in self._groups:
            self._check_key_found_in_adata_field(adata, group, "obs")

    def _verify_groups_reps(self, adata: AnnData) -> None:
        """"""  # noqa
        for rep in self._groups_reps.values():
            self._check_key_found_in_adata_field(adata, rep, "uns")

    def _verify_schema(self, adata):
        """"""  # noqa
        self._verify_groups(adata)
        self._verify_groups_reps(adata)

    def _get_covs_df(
        self,
        adata: AnnData,
    ) -> pd.DataFrame:
        """"""  # noqa
        return adata.obs.loc[:, self._groups]

    def _get_covs_repr_dict(
        self,
        adata: AnnData,
    ) -> dict[str, MappedArray]:
        """"""  # noqa
        return {col: adata.uns[rep] for col, rep in self._groups_reps.items()}

    def _get_data(self, adata: AnnData) -> CategoricalData:
        """"""  # noqa
        covs_df_dict: pd.DataFrame = self._get_covs_df(adata)
        repr_dict: dict[str, MappedArray] = self._get_covs_repr_dict(adata)
        encoders_dict: Mapping[str, TargetCovariatesEncoderCls] = get_covariates_encoders_from_dict(
            self._groups_encoding, covs_df_dict
        )
        return CategoricalData(
            covs_df_dict,
            repr_dict=repr_dict,
            categorical_encoders=encoders_dict,
        )

    @property
    def groups(self) -> Collection[str]:
        """"""  # noqa
        return self._groups

    @property
    def groups_reps(self) -> dict[str, str]:
        """"""  # noqa
        return self._groups_reps

    @property
    def groups_encoding(self) -> dict[str, TargetCovariatesEncodingId]:
        """"""  # noqa
        return self._groups_encoding
