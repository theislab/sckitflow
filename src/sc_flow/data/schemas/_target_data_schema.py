from collections.abc import Collection, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

import pandas as pd
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncoderCls, TargetCovariatesEncodingId
from sc_flow.data._mixins import BatchMixin
from sc_flow.data._structures import CategoricalData, TargetData
from sc_flow.data._utils import get_covariates_encoders_from_dict
from sc_flow.data.schemas._base_schema import BaseDataSchema

__all__ = ["TargetDataSchema"]


@dataclass(frozen=True)
class TargetDataSchema(BaseDataSchema):
    """"""  # noqa

    categorical_covs_dict: dict[str, TargetCovariatesEncodingId] = dc_field(default_factory=lambda: {})
    continuous_covs_dict: Collection[str] = dc_field(default_factory=lambda: [])

    def _verify_schema_categorical_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the categorical target covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        if self.categorical_covs_dict is not None:
            for target_covariate, encoder_id in self.categorical_covs_dict.items():
                self._check_key_found_in_adata_field(adata, target_covariate, "obs")
                if encoder_id not in ["label", "one-hot"]:
                    msg = (
                        f"Encoder identifier {encoder_id} for target covariate encoding is not supported."
                        'Possible options are `"label", "one-hot"`'
                    )
                    raise ValueError(msg)

    def _verify_schema_continuous_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the continuous target covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        if self.continuous_covs_dict is not None:
            for target_covariate in self.continuous_covs_dict:
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

    def _get_categorical_covariates(self, adata: AnnData) -> CategoricalData:
        """"""  # noqa
        if self.categorical_covs_dict is None:
            return None
        covariates_df: pd.DataFrame = self._get_covariates_df(adata)
        encoders_dict: Mapping[str, TargetCovariatesEncoderCls] = get_covariates_encoders_from_dict(
            self.categorical_covs_dict, covariates_df
        )
        return CategoricalData(covariates_df, categorical_encoders=encoders_dict)

    def _get_continuous_covariates(
        self,
        adata: AnnData,
    ) -> BatchMixin:
        """"""  # noqa
        covariates_dict = {}
        for covariate in self.continuous_covs_dict:
            covariates_dict[covariate] = adata.obsm[covariate]
        return BatchMixin(covariates_dict)

    def _get_data(
        self,
        adata: AnnData,
    ) -> TargetData:
        """"""  # noqa
        categorical_covariates: CategoricalData = self._get_categorical_covariates(adata)
        continuous_covariates: BatchMixin = self._get_continuous_covariates(adata)
        return TargetData(categorical_covariates=categorical_covariates, continuous_covariates=continuous_covariates)

    @property
    def categorical_covariates(self) -> tuple[str]:
        """"""  # noqa
        return tuple(self.categorical_covs_dict.keys())
