from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from anndata import AnnData

from sc_flow._types import TargetCovariatesEncoding

__all__ = ["AnnDataContract"]


@dataclass(frozen=True)
class AnnDataContract:
    """Object for handing contracts with anndata objects."""

    sample_rep: str | None = None
    categorical_target_covariates: Mapping[str, TargetCovariatesEncoding] | None = None
    continuous_target_covariates: Collection[str] | None = None
    sample_covariates: Mapping[str, str] | None = None

    @staticmethod
    def _check_key_found_in_adata_field(
        adata: AnnData,
        identifier: str,
        adata_field_key: Literal["obs", "uns", "obsm"],
    ) -> None:
        """Checks that a given key identifier is present in a selected attribute of the input annotated data.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`

        :param identifier: The string identifier for the key to be searched in :param: `adata`.
        :type identifier: class: `str`

        :param adata_field: The attribute of :param: `adata` checked.
        :type adata_field: class: `Literal["obs", "uns", "obsm"]`
        """
        adata_field = getattr(adata, adata_field_key)

        if identifier not in adata_field:
            available = list(adata_field.keys())
            raise KeyError(f"Key '{identifier}' not found in adata.{adata_field_key}. Available keys: {available}")

    def _validate_state_data(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the state representation settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        # when we provide the sample rep key it should appear in `self.adata.obsm`
        if self.sample_rep is not None:
            self._check_key_found_in_adata_field(adata, self.sample_rep, "obsm")

    def _validate_categorical_target_covariates(
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

    def _validate_continuous_target_covariates(
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

    def _validate_sample_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        if self.sample_covariates is not None:
            for cell_covariate_obs_col, covariate_uns_key in self.sample_covariates.items():
                self._check_key_found_in_adata_field(adata, cell_covariate_obs_col, "obs")
                self._check_key_found_in_adata_field(adata, covariate_uns_key, "uns")

    def validate_contract(
        self,
        adata: AnnData,
    ) -> None:
        """Validates the contract on the input :class: `AnnData`

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        self._validate_state_data(adata)
        self._validate_categorical_target_covariates(adata)
        self._validate_continuous_target_covariates(adata)
        self._validate_sample_covariates(adata)
