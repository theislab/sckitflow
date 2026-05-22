import numpy as np

from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.external._context import ExternalModelContext
from sc_flow.preprocessing._base_preproc import BasePreprocessing
from sc_flow.preprocessing.transforms._base import BaseTransform

__all__ = ["ConditionPreprocessing"]


class ConditionPreprocessing(BasePreprocessing):
    def __init__(
        self,
        cov_key: str,
        transform: BaseTransform | None = None,
        encoder_context: ExternalModelContext | None = None,
        decoder_context: ExternalModelContext | None = None,
    ) -> None:
        """Initializes the preprocessing module.

        :param transform: The transformation to be applied to the data.
            Defaults to `None`.
        :type transform: class: `BaseTransform | None`

        :param encoder_context: The context for optional encoder models.
            Defaults to `None`.
        :type encoder_context: class: `ExternalModelContext | None`

        :param decoder_context: The context for optional decoder models.
            Defaults to `None`.
        :type decoder_context: class: `ExternalModelContext | None`
        """
        super().__init__(transform=transform, encoder_context=encoder_context, decoder_context=decoder_context)
        self._cov_key = cov_key

    def _verify_input_data(self, data: DistributionData) -> None:
        if data.condition_data is None:
            raise ValueError("No condition data available.")
        if data.condition_data.continuous_covariates is None:
            raise ValueError("No continuous covariates found")
        if self._cov_key not in data.condition_data.continuous_covariates.mapping:
            raise KeyError(f"Continuous covariate {self._cov_key} not found.")

    def _extract_data(self, data: DistributionData) -> np.ndarray:
        self._verify_input_data(data)
        return data.condition_data.continuous_covariates[self._cov_key]

    def _write_data(self, X: np.ndarray, data: DistributionData) -> DistributionData:
        self._verify_input_data(data)

        # ---- Update condition data ----
        # create updated mapping
        updated_mapping = data.condition_data.continuous_covariates.mapping.copy()
        updated_mapping.pop(self._cov_key)
        updated_mapping[self._cov_key] = X

        # prepare updated data
        continuous_covariates = BatchMixin(updated_mapping)
        condition_data = MixedTypeData(
            categorical_covariates=data.condition_data.categorical_covariates,
            continuous_covariates=continuous_covariates,
        )

        # return updated distribution
        return DistributionData(
            state_data=data.state_data,
            response_data=data.response_data,
            condition_data=condition_data,
            groups_data=data.groups_data,
            source_coupling_data=data.source_coupling_data,
            target_coupling_data=data.target_coupling_data,
        )
