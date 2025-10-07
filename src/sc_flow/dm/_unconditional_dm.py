from collections.abc import Sequence
from typing import Literal, NewType

import numpy as np
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sc_flow._types import MappedArray, TargetCovariatesEncoding

## TODO: remove when felix writes this stuff
StateData = NewType("StateData", np.array)
ConditionData = NewType("ConditionData", MappedArray)
TargetData = NewType("TargetData", MappedArray)
CouplingData = NewType("CouplingData", MappedArray)


__all__ = ["UnconditionalDataManager"]


class UnconditionalDataManager:
    """Class for managing :class: `anndata.AnnData` objects in the unconditional setting.

    It handles the state data, as well as optional target covariates that we are interested
    in predicting on top of the generative model.
    """

    def __init__(
        self,
        sample_rep: str | None = None,
        categorical_target_covariates: dict[str, TargetCovariatesEncoding] | None = None,
        continuous_target_covariates: Sequence[str] | None = None,
    ) -> None:
        """Initializes the unconditional data manager.

        :param sample_rep: (Optional) Cell State representation on which the flow will be
            defined. When provided, it has to appear as a key in `.obsm`. Defaults to `None`,
            in which case the `.X` attribute will be used as representation.
        :type sample_rep: class: `str | None`

        :param categorical_target_covariates: (Optional) Dictionary used for definition of target covariates.
            Each key will represent a target covariate identifier, that must appear as a column in `.obs`.
            The corresponding values must indicate the type of encoding to be done on them, that can be set to
            either `"label"` or `"one-hot"` for label and one hot encoding respectively. Defaults to `None`.
        :type categorical_target_covariates: class: `dict[str, TargetCovariatesEncoding] | None`

        :param continuous_target_covariates: (Optional) Sequence of string identifier of continuous target
            covariates to regress against. These covariates must appear as keys in `.obsm`. Defaults to `None`.
        :type continuous_target_covariates: class:  `Sequence[str]`
        """
        self._sample_rep = sample_rep
        self._categorical_target_covariates = (
            {} if categorical_target_covariates is None else categorical_target_covariates
        )
        self._continuous_target_covariates = (
            () if continuous_target_covariates is None else continuous_target_covariates
        )

        self._categorical_target_encoders: dict[str, LabelEncoder | OneHotEncoder] = {}

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

        if identifier not in adata_field.keys():
            msg = f"Representation {identifier} not found in `adata.{adata_field_key}.keys()`."
            raise KeyError(msg)

    @staticmethod
    def _get_categorical_target_encoder_and_transformed_data(
        covariate_encoder_id: TargetCovariatesEncoding,
        covariate_data: np.ndarray,
    ) -> tuple[LabelEncoder | OneHotEncoder, np.ndarray]:
        """Retrieves the target data and the corresponding encoder.

        :param covariate_encoder_id: The string identifier indicating what type of encoding
            to be applied on the target data.
        :type covariate_encoder_id: class: `TargetCovariatesEncoding`

        :param covariate_data: The covariate data to be encoded.
        :type covariate_data: class: `np.ndarray`
        """
        if covariate_encoder_id == "label":
            if covariate_data.ndim == 2:
                if covariate_data.shape[1] != 1:
                    msg = (
                        'When using "label"  as target representation in `target_covariates`,'
                        f"the last dimension should be singleton, but found data of shape {covariate_data.shape}."
                    )
                    raise ValueError(msg)
                covariate_data = covariate_data.reshape(-1)

            # we require only one dimension when using label encoding
            if covariate_data.ndim != 1:
                msg = (
                    'When using "label" as target representation in `target_covariates`,'
                    f"the data should have only one dimension, but found data of shape {covariate_data.shape}."
                )
                raise ValueError(msg)
            encoder = LabelEncoder().fit(covariate_data)

        elif covariate_encoder_id == "one-hot":
            if covariate_data.ndim == 1:
                covariate_data = covariate_data.reshape(-1, 1)

            if covariate_data.ndim != 2:
                msg = (
                    'When using "one-hot" as target representation in `target_covariates`,'
                    f"you need to pass a 2-dimensional array, found {covariate_data.ndim=}."
                )
                raise ValueError(msg)
            encoder = OneHotEncoder().fit(covariate_data)
        covariate_data = encoder.transform(covariate_data)
        return encoder, covariate_data

    def _validate_state_data(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the state representation settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        # when we provide the sample rep key it should appear in `self.adata.obsm`
        if self._sample_rep is not None:
            self._check_key_found_in_adata_field(adata, self._sample_rep, "obsm")

    def _validate_categorical_target_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the categorical target covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        if self._categorical_target_covariates is None:
            return None
        for target_covariate, encoder_id in self._categorical_target_covariates.items():
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
        if self._continuous_target_covariates is None:
            return None
        for target_covariate in self._continuous_target_covariates:
            self._check_key_found_in_adata_field(adata, target_covariate, "obs")

    def _get_categorical_target_data(
        self,
        adata: AnnData,
    ) -> MappedArray:
        """Retrieves the categorical target covariates from the underlying :class: `anndata.AnnData` object.

        :param adata: The input annotated data from which to retrieve the categorical target covariates.
        :type adata: class: `AnnData`
        """
        self._validate_categorical_target_covariates(adata)
        if self._categorical_target_covariates is None:
            return None
        data = {}
        for covariate_id, covariate_encoder_id in self._categorical_target_covariates.items():
            covariate_data = adata.obs[covariate_id].values
            encoder, covariate_data = self._get_categorical_target_encoder_and_transformed_data(
                covariate_encoder_id,
                covariate_data,
            )
            data[covariate_id] = covariate_data
            self._categorical_target_encoders[covariate_id] = encoder
        return data

    def _get_continuous_target_data(
        self,
        adata: AnnData,
    ) -> MappedArray:
        """Retrieves the continuous target covariates from the underlying :class: `anndata.AnnData` object.

        :param adata: The input annotated data from which to retrieve the continuous target covariates.
        :type adata: class: `AnnData`
        """
        self._validate_continuous_target_covariates(adata)
        if self._continuous_target_covariates is None:
            return None
        return {covariate: adata.obsm[covariate] for covariate in self._continuous_target_covariates}

    def get_state_data(
        self,
        adata: AnnData,
    ) -> StateData:
        """Retrieves the cell state representation from the underlying :class: `anndata.AnnData` object.

        :param adata: The input annotated data from which to retrieve the cell state representations.
        :type adata: class: `AnnData`
        """
        self._validate_state_data(adata)
        if self._sample_rep is not None:
            X = adata.obsm[self._sample_rep]
            return StateData(X)
        return StateData(adata.X)

    def get_target_data(
        self,
        adata: AnnData,
    ) -> TargetData:
        """Retrieves the cell state representation from the underlying :class: `anndata.AnnData` object.

        :param adata: The input annotated data from which to retrieve the cell state representations.
        :type adata: class: `AnnData`
        """
        categorical_targets = self._get_categorical_target_data(adata)
        continouos_targets = self._get_continuous_target_data(adata)
        return TargetData({**categorical_targets, **continouos_targets})

    @property
    def categorical_target_encoders(
        self,
    ) -> dict[str, LabelEncoder | OneHotEncoder]:
        """Returns the encoders associated to each of the target covariates."""
        return self._categorical_target_encoders
