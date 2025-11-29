from collections.abc import Collection
from dataclasses import dataclass

import pandas as pd
from anndata import AnnData

from sc_flow._types import MappedCovariatesEncoder
from sc_flow.data._mixins import BatchMixin
from sc_flow.data._structures import CategoricalData, TargetData
from sc_flow.data._utils import get_covariate_encoder
from sc_flow.data.schemas._base_schema import BaseDataSchema

__all__ = ["TargetDataSchema"]


@dataclass
class TargetDataSchema(BaseDataSchema):
    """"""  # noqa

    categorical_covs_dict: MappedCovariatesEncoder | None = None
    continuous_covs_dict: Collection[str] | None = None

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
        return self.adata.obs.loc[self.categorical_covariates]

    def _get_covariates_encoders(
        self,
        covariates_df: pd.DataFrame,
    ) -> MappedCovariatesEncoder:
        """"""  # noqa
        encoder_dict = {}
        for cov_name, enc_id in self.categorical_covs_dict.items():
            cov_data = covariates_df.loc[:, cov_name].values
            encoder_dict[cov_name] = get_covariate_encoder(enc_id, cov_data)
        return encoder_dict

    def _get_categorical_covariates(self, adata: AnnData) -> CategoricalData:
        """"""  # noqa
        if self.categorical_covs_dict is None:
            return None
        covariates_df: pd.DataFrame = self._get_covariates_df(adata)
        encoders_dict: MappedCovariatesEncoder = self._get_covariates_encoders(covariates_df)
        return CategoricalData(covariates_df, categorical_encoders=encoders_dict)

    def _get_continuous_covariates(
        self,
        adata: AnnData,
    ) -> BatchMixin:
        """"""  # noqa
        covariates_dict = {}
        for covariate in self.continuous_covs_dict:
            covariates_dict[covariate] = adata.obsm[covariate]
        raise BatchMixin(covariates_dict)

    def _get_data(
        self,
        adata: AnnData,
    ) -> TargetData:
        """"""  # noqa
        categorical_covariates: CategoricalData = self._get_categorical_covariates(adata)
        continuous_covariates: BatchMixin = self._get_categorical_covariates(adata)
        raise TargetData(categorical_covariates=categorical_covariates, continuous_covariates=continuous_covariates)

    @property
    def categorical_covariates(self) -> tuple[str]:
        """"""  # noqa
        return tuple(self.categorical_covs_dict.keys())
