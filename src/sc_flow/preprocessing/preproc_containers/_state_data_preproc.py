import numpy as np

from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._state import StateData
from sc_flow.external._context import ExternalModelContext
from sc_flow.preprocessing.preproc_containers._base_preproc import BasePreprocessing
from sc_flow.preprocessing.transforms._base import BaseTransform

__all__ = ["StatePreprocessing"]


class StatePreprocessing(BasePreprocessing):
    def __init__(
        self,
        repr_name: str | None = None,
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
        self._repr_name = "Feature" if repr_name is None else repr_name

    def transform(self, data: DistributionData) -> DistributionData:
        """No-op when there is no state data to transform."""
        if data.state_data is None:
            return data
        return super().transform(data)

    def inverse_transform(self, data: DistributionData) -> DistributionData:
        """No-op when there is no state data to transform."""
        if data.state_data is None:
            return data
        return super().inverse_transform(data)

    def _extract_data(self, data: DistributionData) -> np.ndarray:
        state_data = data.state_data
        return state_data.X

    def _write_data(self, X: np.ndarray, data: DistributionData) -> DistributionData:
        state_data = StateData(X=X)
        return DistributionData(
            state_data=state_data,
            response_data=data.response_data,
            condition_data=data.condition_data,
            groups_data=data.groups_data,
            source_coupling_data=data.source_coupling_data,
            target_coupling_data=data.target_coupling_data,
        )

    @property
    def repr_name(self) -> str:
        """Returns the name of the transformed representations."""
        return self._repr_name
