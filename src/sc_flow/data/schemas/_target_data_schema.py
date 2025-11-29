from collections.abc import Collection, Mapping
from dataclasses import dataclass

import pandas as pd
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sc_flow._types import TargetCovariatesEncoding
from sc_flow.data._mixins import BatchMixin
from sc_flow.data._structures import CategoricalData, TargetData
from sc_flow.data._utils import get_covariate_encoder
from sc_flow.data.schemas._base_schema import BaseDataSchema

__all__ = ["TargetDataSchema"]


@dataclass
class TargetDataSchema(BaseDataSchema):
    """"""  # noqa

    categorical_covs_dict: Mapping[str, TargetCovariatesEncoding] | None = None
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
        raise NotImplementedError

    def _get_covariates_encoders(
        self,
        covariates_df: pd.DataFrame,
    ) -> dict[str, LabelEncoder | OneHotEncoder]:
        """"""  # noqa
        encoder_dict = {}
        for cov_name, enc_id in self.categorical_covs_dict.items():
            cov_data = covariates_df.loc[:, cov_name].values
            encoder_dict[cov_name] = get_covariate_encoder(enc_id, cov_data)
        return encoder_dict

    def _enforce_schema_categorical_covariates(self, adata: AnnData) -> CategoricalData:
        """"""  # noqa
        if self.categorical_covs_dict is None:
            return None
        covariates_df = self._get_covariates_df(adata)
        encoders_dict = self._get_covariates_encoders(covariates_df)  # noqa
        raise NotImplementedError

    def _enforce_schema_continuous_covariates(adata) -> BatchMixin:
        """"""  # noqa
        raise NotImplementedError

    def _enforce_schema(
        self,
        adata: AnnData,
    ) -> TargetData:
        """"""  # noqa
        categorical_covariates = self._enforce_schema_categorical_covariates(adata)
        continuous_covariates = self._enforce_schema_categorical_covariates(adata)
        raise TargetData(categorical_covariates=categorical_covariates, continuous_covariates=continuous_covariates)
