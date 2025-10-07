import abc
from collections.abc import Callable
from typing import Any

import numpy as np

from sc_flow import _types

__all__ = ["BaseMethod", "FlowMatching", "OTFlowMatching", "GENOT"]


class BaseMethod(abc.ABC):
    """TODO."""

    def __init__(
        self,
        vf: Any,  # TODO: adapt type
        probability_path: _types.ProbabilityPathId,
        time_sampler: Callable[[np.ndarray, int], np.ndarray],
        ema: int = 1,
    ):
        self.vf = vf
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.ema = ema
        # TODO: add cfg

        self._is_trained = False

    @abc.abstractmethod
    def __call__(self, num_iterations: int, *args: Any, **kwargs: Any) -> Any:
        """TODO."""
        pass

    @abc.abstractmethod
    def step_fn(self, *args: Any, **kwargs: Any) -> Any:
        """TODO."""
        pass

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """TODO."""
        pass

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @abc.abstractmethod
    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value


class FlowMatching(BaseMethod, abc.ABC):
    """TODO."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class OTFlowMatching(FlowMatching, abc.ABC):
    """TODO."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def match_data(self, src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TODO."""
        pass


class GENOT(BaseMethod, abc.ABC):
    """TODO."""

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def match_data(self, src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TODO."""
        pass
