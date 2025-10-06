import abc
from collections.abc import Callable
from typing import Any

import numpy as np

from sc_flow._types import ProbabilityPathId

__all__ = ["BaseMethod", "FlowMatching", "OTFlowMatching", "GENOT"]


class BaseMethod(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, ema: int = 1, **kwargs):
        self.ema = ema
        # TODO: add cfg

    @abc.abstractmethod
    def __call__(self, num_iterations: int, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def step_fn(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @abc.abstractmethod
    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value


class FlowMatching(BaseMethod):
    def __init__(
        self,
        vf: Any,  # TODO: adapt once rebased
        probability_path: ProbabilityPathId,
        time_sampler: Callable[[np.ndarray, int], np.ndarray],
        ema: int,
    ):
        super().__init__(ema=ema)
        self.vf = vf
        self.probability_path = probability_path
        self.time_sampler = time_sampler


class OTFlowMatching(FlowMatching):
    @abc.abstractmethod
    def match_data(self, src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """"""
        pass


class GENOT(BaseMethod):
    def __init__(self, *args, **kwargs):
        pass

    def match_data(self, src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass
