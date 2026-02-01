from collections.abc import Callable, Collection, Mapping

import pandas as pd
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncoderCls, TargetCovariatesEncodingId
from sc_flow.data._mixins import BatchMixin
from sc_flow.data._utils import get_covariates_encoders_from_dict
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["ResponseDataSchema"]


class ResponseDataSchema(StrictDataSchema):
    """"""  # noqa

    def __init__(
        self,
        categorical_covs_dict: dict[str, TargetCovariatesEncodingId] | None = None,
        continuous_covs: Collection[str] | None = None,
        encoding_transform_fn: dict[str, Callable] | None = None,
        encoding_inverse_transform_fn: dict[str, Callable] | None = None,
    ) -> None:
        """"""  # noqa
        self._categorical_covs_dict = {} if categorical_covs_dict is None else categorical_covs_dict
        self._continuous_covs = [] if continuous_covs is None else continuous_covs
        self._encoding_transform_fn = {} if encoding_transform_fn is None else encoding_transform_fn
        self._encoding_inverse_transform_fn = (
            {} if encoding_inverse_transform_fn is None else encoding_inverse_transform_fn
        )
        super().__init__()

    def _verify_args(self) -> None:
        """"""  # noqa
        self._check_is_valid_encoder_id_dict(self._categorical_covs_dict)

    def _verify_schema_categorical_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the categorical target covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        for target_covariate in self._categorical_covs_dict.keys():
            self._check_key_found_in_adata_field(adata, target_covariate, "obs")

    def _verify_schema_continuous_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the continuous target covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        for target_covariate in self._continuous_covs:
            self._check_key_found_in_adata_field(adata, target_covariate, "obsm")

    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        self._verify_schema_categorical_covariates(adata)
        self._verify_schema_continuous_covariates(adata)

    def _get_covariates_df(
        self,
        adata: AnnData,
    ) -> pd.DataFrame:
        """"""  # noqa
        return adata.obs.loc[:, self.categorical_covariates]

    def _get_categorical_covariates(self, adata: AnnData) -> CategoricalData | None:
        """"""  # noqa
        if not self.has_categorical_covariates:
            return None
        covariates_df: pd.DataFrame = self._get_covariates_df(adata)
        encoders_dict: Mapping[str, TargetCovariatesEncoderCls] = get_covariates_encoders_from_dict(
            self._categorical_covs_dict,
            covariates_df,
            fn_dict=self._encoding_transform_fn,
            inverse_fn_dict=self._encoding_inverse_transform_fn,
        )
        return CategoricalData(covariates_df, categorical_encoders=encoders_dict)

    def _get_continuous_covariates(
        self,
        adata: AnnData,
    ) -> BatchMixin:
        """"""  # noqa
        if not self.has_continuous_covariates:
            return None
        covariates_dict = {}
        for covariate in self._continuous_covs:
            covariates_dict[covariate] = adata.obsm[covariate]
        return BatchMixin(covariates_dict)

    def _get_data(
        self,
        adata: AnnData,
    ) -> MixedTypeData | None:
        """"""  # noqa
        categorical_covariates: CategoricalData = self._get_categorical_covariates(adata)
        continuous_covariates: BatchMixin = self._get_continuous_covariates(adata)
        if categorical_covariates is None and continuous_covariates is None:
            return None
        return MixedTypeData(categorical_covariates=categorical_covariates, continuous_covariates=continuous_covariates)

    @property
    def categorical_covariates(self) -> tuple[str]:
        """"""  # noqa
        return tuple(self.categorical_covs_dict.keys())

    @property
    def categorical_covs_dict(self) -> dict[str, TargetCovariatesEncodingId]:
        """"""  # noqa
        return self._categorical_covs_dict

    @property
    def continuous_covs(self) -> Collection[str]:
        """"""  # noqa
        return self._continuous_covs

    @property
    def has_categorical_covariates(self) -> bool:
        """"""  # noqa
        return len(self._categorical_covs_dict) > 0

    @property
    def has_continuous_covariates(self) -> bool:
        """"""  # noqa
        return len(self._continuous_covs) > 0
