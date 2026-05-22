import numpy as np

from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._state import StateData
from sc_flow.preprocessing._base_preproc import BasePreprocessing

__all__ = ["StatePreprocessing"]


class StatePreprocessing(BasePreprocessing):
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
