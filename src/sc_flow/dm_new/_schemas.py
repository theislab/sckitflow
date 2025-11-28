import abc
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from anndata import AnnData

from sc_flow._types import TargetCovariatesEncoding
from sc_flow.data_new._data_structures import (
    BaseDataContainer,
    CategoricalDataContainer,
    CombinatorialCategoricalDataContainer,
    ConditionDataContainer,
    StateDataContainer,
    TargetDataContainer,
)
from sc_flow.data_new._mixins import BatchMixin

__all__ = [
    "BaseDataSchema",
    "TargetDataSchema",
    "ConditionDataSchema",
    "ConditionDataSchema",
]


@dataclass
class BaseDataSchema(abc.ABC):
    """"""  # noqa

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

    @abc.abstractmethod
    def _verify_contract(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        raise NotImplementedError

    @abc.abstractmethod
    def _enforce_contract(
        self,
        adata: AnnData,
    ) -> BaseDataContainer:
        """"""  # noqa
        raise NotImplementedError

    def enforce_contract(
        self,
        adata: AnnData,
    ) -> BaseDataContainer:
        """"""  # noqa
        self._verify_contract(adata)
        return self._enforce_contract(adata)


@dataclass
class StateDataSchema:
    """"""  # noqa

    sample_rep: str | None = None

    def _verify_contract(
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

    def _enforce_contract(
        self,
        adata: AnnData,
    ) -> StateDataContainer:
        """"""  # noqa
        if self.sample_rep is None:
            return adata.X
        return adata.obsm[self.sample_rep]


@dataclass
class TargetDataSchema:
    """"""  # noqa

    categorical_target_covariates: Mapping[str, TargetCovariatesEncoding] | None = None
    continuous_target_covariates: Collection[str] | None = None

    def _verify_contract_categorical_covariates(
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

    def _verify_contract_continuous_covariates(
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

    def _verify_contract(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        self._verify_contract_categorical_covariates(adata)
        self._verify_contract_continuous_covariates(adata)

    def _enforce_contract_categorical_covariates(adata) -> CategoricalDataContainer:
        """"""  # noqa
        raise NotImplementedError

    def _enforce_contract_continuous_covariates(adata) -> BatchMixin:
        """"""  # noqa
        raise NotImplementedError

    def _enforce_contract(
        self,
        adata: AnnData,
    ) -> TargetDataContainer:
        """"""  # noqa
        raise NotImplementedError


@dataclass
class ConditionDataSchema:
    """Implements the logic for conditioning."""

    conditions: dict[str, Collection[str]] | None = None
    conditions_reps: dict[str, str] | None = None
    conditions_covariates: Collection[str] | None = None

    @property
    def all_condition_categories(
        self,
    ) -> Collection[str]:
        """"""  # noqa
        if self.conditions is None:
            return ()
        return tuple(cat for condition in self.conditions.values() for cat in condition)

    @property
    def condition_category_to_realm(
        self,
    ) -> dict[str, str]:
        """"""  # noqa
        cat2realm = {}
        for condition, condition_cats in self.conditions.items():
            for cat in condition_cats:
                cat2realm[cat] = condition
        return cat2realm

    @property
    def allows_ot_coupling(
        self,
    ) -> bool:
        """"""  # noqa
        return self.conditions_covariates is None

    def _verify_contract_categorical_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        raise NotImplementedError

    def _verify_contract_continuous_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        raise NotImplementedError

    def _enforce_contract_categorical_covariates(
        self,
        adata: AnnData,
    ) -> CombinatorialCategoricalDataContainer:
        """"""  # noqa
        raise NotImplementedError

    def _enforce_contract_continuous_covariates(
        self,
        adata: AnnData,
    ) -> BatchMixin:
        """"""  # noqa
        raise NotImplementedError

    def _verify_contract(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        self._verify_contract_categorical_covariates(adata)
        self._verify_contract_continuous_covariates(adata)

    def _enforce_contract(
        self,
        adata: AnnData,
    ) -> ConditionDataContainer:
        """"""  # noqa
        categorical_covariates = self._enforce_contract_categorical_covariates(adata)  # noqa
        continuous_covariates = self._enforce_contract_continuous_covariates(adata)  # noqa
        raise NotImplementedError
