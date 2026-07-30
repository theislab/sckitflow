from collections.abc import Collection

from sckitflow._utils import check_sequence_query_against_reference
from sckitflow.data.containers._distribution import DistributionData
from sckitflow.external._context import ExternalModelContext
from sckitflow.preprocessing.preproc_containers._condition_data_preproc import ConditionPreprocessing
from sckitflow.preprocessing.preproc_containers._state_data_preproc import StatePreprocessing
from sckitflow.preprocessing.transforms._base import BaseTransform

__all__ = ["DataPreprocessor"]


class DataPreprocessor:
    """Class for handling data preprocessing pipelines."""

    def __init__(
        self,
        conditions_covariates: Collection[str] | None = None,
        state_transform: BaseTransform | None = None,
        state_encoder_context: ExternalModelContext | None = None,
        state_decoder_context: ExternalModelContext | None = None,
        state_preproc_repr_name: str | None = None,
        condition_covariates_transform_dict: dict[str, BaseTransform] | None = None,
        condition_covariates_encoder_context_dict: dict[str, ExternalModelContext] | None = None,
        condition_covariates_decoder_context_dict: dict[str, ExternalModelContext] | None = None,
    ) -> None:
        """Initializes the preprocessing module.

        :param condition_covariates: Sequence of string identifiers for continuous condition covariates.
            Defaults to `None`.
        :type conditions_covariates: class: `Collection[str] | None`

        :param state_transform: The transformation to be applied to the state data.
            Defaults to `None`.
        :type state_transform: class: `BaseTransform | None`

        :param state_encoder_context: The context for optional encoder models of state data.
            Defaults to `None`.
        :type state_encoder_context: class: `ExternalModelContext | None`

        :param state_decoder_context: The context for optional decoder models of state data.
            Defaults to `None`.
        :type state_decoder_context: class: `ExternalModelContext | None`

        :param state_preproc_repr_name: Identifier for the variable names after preprocessing.
            Defaults to `None`.
        :type state_preproc_repr_name: class: `str | None`

        :param condition_covariates_transform_dict: Optional dictionary mapping each continuous
            covariates to their respective transformation object. Defaults to `None`.
        :type condition_covariates_transform_dict: class: `dict[str, BaseTransform] | None`

        :param condition_covariates_encoder_context_dict: Optional dictionary mapping each continuous
            covariates to their respective encoder context. Defaults to `None`.
        :type condition_covariates_encoder_context_dict: class `dict[str, ExternalModelContext] | None`

        :param condition_covariates_decoder_context_dict: Optional dictionary mapping each continuous
            covariates to their respective decoder context. Defaults to `None`.
        :type condition_covariates_decoder_context_dict: class `dict[str, ExternalModelContext] | None`
        """
        self._state_preproc = StatePreprocessing(
            repr_name=state_preproc_repr_name,
            transform=state_transform,
            encoder_context=state_encoder_context,
            decoder_context=state_decoder_context,
        )
        self._cond_preproc_dict = self._init_cond_preproc_dict(
            conditions_covariates=conditions_covariates,
            condition_covariates_transform_dict=condition_covariates_transform_dict,
            condition_covariates_encoder_context_dict=condition_covariates_encoder_context_dict,
            condition_covariates_decoder_context_dict=condition_covariates_decoder_context_dict,
        )

    def _init_cond_preproc_dict(
        self,
        conditions_covariates: Collection[str] | None = None,
        condition_covariates_transform_dict: dict[str, BaseTransform] | None = None,
        condition_covariates_encoder_context_dict: dict[str, ExternalModelContext] | None = None,
        condition_covariates_decoder_context_dict: dict[str, ExternalModelContext] | None = None,
    ) -> dict[str, ConditionPreprocessing]:
        # ---- Early return when no continuous covariates are present ----
        # empty dictionary if no condition covariate is present
        if conditions_covariates is None:
            return {}

        # ---- Prepare dictionaries ----
        condition_covariates_transform_dict = (
            {} if condition_covariates_transform_dict is None else condition_covariates_transform_dict
        )
        condition_covariates_encoder_context_dict = (
            {} if condition_covariates_encoder_context_dict is None else condition_covariates_encoder_context_dict
        )
        condition_covariates_decoder_context_dict = (
            {} if condition_covariates_decoder_context_dict is None else condition_covariates_decoder_context_dict
        )

        # ---- Check that the keys are correct ----
        # transforms dictionary
        check_sequence_query_against_reference(condition_covariates_transform_dict.keys(), conditions_covariates)
        # encoder contexts
        check_sequence_query_against_reference(condition_covariates_encoder_context_dict.keys(), conditions_covariates)
        # decoder contexts
        check_sequence_query_against_reference(condition_covariates_decoder_context_dict.keys(), conditions_covariates)

        # ---- Initialize dictionary of preprocessing modules ----
        # initialize preprocessing for each covariate
        preproc_dict = {}

        # loop over condition covariates
        for cov in conditions_covariates:
            # ---- Retrieve preprocessing setting for current covariate ----
            cov_transform = condition_covariates_transform_dict.get(cov, None)
            cov_encoder_ctx = condition_covariates_encoder_context_dict.get(cov, None)
            cov_decoder_ctx = condition_covariates_decoder_context_dict.get(cov, None)

            # ---- Initialize preprocessor and update dictionary
            cov_preproc = ConditionPreprocessing(
                cov, transform=cov_transform, encoder_context=cov_encoder_ctx, decoder_context=cov_decoder_ctx
            )
            preproc_dict[cov] = cov_preproc
        return preproc_dict

    def _fit(self, data: DistributionData) -> None:
        # ---- State preprocessing ----
        self._state_preproc.fit(data)

        # ---- Condition preprocessing ----
        for cond_preproc in self._cond_preproc_dict.values():
            cond_preproc.fit(data)

    def _unload(self) -> None:
        # ---- State preprocessing ----
        self._state_preproc.unload()

        # ---- Condition preprocessing ----
        for cond_preproc in self._cond_preproc_dict.values():
            cond_preproc.unload()

    def _apply_preproc_transforms(
        self,
        data: DistributionData,
    ) -> DistributionData:
        # ---- State preprocessing ----
        data = self._state_preproc.transform(data)

        # ---- Condition preprocessing ----
        for cond_preproc in self._cond_preproc_dict.values():
            data = cond_preproc.transform(data)
        return data

    def _apply_preproc_inverse_transforms(
        self,
        data: DistributionData,
    ) -> DistributionData:
        # ---- State preprocessing ----
        data = self._state_preproc.inverse_transform(data)

        # ---- Condition preprocessing ----
        for cond_preproc in self._cond_preproc_dict.values():
            data = cond_preproc.inverse_transform(data)
        return data

    def fit(self, data: DistributionData) -> None:
        """Fits the preprocessing module on the input data.

        :param data: The input data to fit the preprocessing module on.
        """
        self._fit(data)

    def transform(
        self,
        data: DistributionData,
    ) -> DistributionData:
        """Applies preprocessing transforms on the input data.

        :param data: The input data to transform.
        """
        return self._apply_preproc_transforms(data)

    def inverse_transform(
        self,
        data: DistributionData,
    ) -> DistributionData:
        """Applies preprocessing inverse transforms on the input data.

        :param data: The input data to transform.
        """
        return self._apply_preproc_inverse_transforms(data)

    def unload(self) -> None:
        """Unloads the underlying models from the context."""
        self._unload()

    @property
    def state_preproc(self) -> StatePreprocessing:
        """Returns the underlying state preprocessing object."""
        return self._state_preproc

    @property
    def cond_preproc(self) -> dict[str, ConditionPreprocessing]:
        """Returns the underlying condition preprocessing object."""
        return self._cond_preproc_dict
