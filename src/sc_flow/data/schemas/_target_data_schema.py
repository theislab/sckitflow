from collections.abc import Collection, Mapping
from dataclasses import dataclass

from anndata import AnnData

from sc_flow._types import TargetCovariatesEncoding
from sc_flow.data._mixins import BatchMixin
from sc_flow.data._structures import CategoricalData, TargetData
from sc_flow.data.schemas._base_schema import BaseDataSchema

__all__ = ["TargetDataSchema"]


@dataclass
class TargetDataSchema(BaseDataSchema):
    """"""  # noqa

    categorical_target_covariates: Mapping[str, TargetCovariatesEncoding] | None = None
    continuous_target_covariates: Collection[str] | None = None

    def _verify_schema_categorical_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the categorical target covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        if self.categorical_target_covariates is not None:
            for target_covariate, encoder_id in self.categorical_target_covariates.items():
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
        if self.continuous_target_covariates is not None:
            for target_covariate in self.continuous_target_covariates:
                self._check_key_found_in_adata_field(adata, target_covariate, "obsm")

    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        self._verify_schema_categorical_covariates(adata)
        self._verify_schema_continuous_covariates(adata)

    def _enforce_schema_categorical_covariates(adata) -> CategoricalData:
        """"""  # noqa
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
