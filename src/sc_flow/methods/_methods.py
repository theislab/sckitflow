from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from sc_flow.data._dims_registry import DataDimensionalitiesRegistry
from sc_flow.data._manager import DataManager
from sc_flow.data.containers._state import StateData

if TYPE_CHECKING:
    from sc_flow.backends.torch._types import TMatchFn as TorchMatchFn
    from sc_flow.backends.torch._types import TNoiseSamplerFn as TorchNoiseSampler
    from sc_flow.backends.torch._types import TTimeSamplerFn as TorchTimeSampler
    from sc_flow.backends.torch.nn import BaseModule as TorchModule
    from sc_flow.backends.torch.probability_paths import BaseProbabilityPath as TorchProbabilityPath
    from sc_flow.backends.torch.solvers import BaseSolver as TorchSolver

__all__ = ["BaseMethod", "BaseGenerativeFlow"]


class BaseMethod(abc.ABC):
    _module_cls: type[TorchModule] | None = None

    def __init__(
        self,
        dims_registry: DataDimensionalitiesRegistry,
        dm: DataManager,
        *args,
        **kwargs,
    ) -> None:
        # initialize attributes
        self._dims_registry = dims_registry
        self._dm = dm

        # check module is passed
        if self._module_cls is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define a `_module_cls` class attribute.")

        # initialize module with dimensionality registry
        self._module = self._module_cls.init_from_dims_registry(self._dims_registry, *args, **kwargs)

    @abc.abstractmethod
    def set_train_mode(self, mode: bool) -> None:
        """Set the underlying module to training (True) or evaluation (False) mode."""
        pass

    @abc.abstractmethod
    def extract_state_data(
        self,
        state_data: StateData | None,
    ) -> Any | None:
        pass

    @abc.abstractmethod
    def train_step(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        pass

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @property
    def module(self) -> TorchModule | None:
        return self._module

    @property
    def dm(self) -> DataManager | None:
        return self._dm

    @property
    def dims_registry(self) -> DataDimensionalitiesRegistry | None:
        return self._dims_registry

    @property
    def is_paired_setting(self) -> bool:
        return self._dm.control_values_dict is not None or self._dm.matched_keys is not None


class BaseGenerativeFlow(BaseMethod):
    _default_solver_cls: type[TorchSolver] | None = None

    def __init__(
        self,
        dims_registry: DataDimensionalitiesRegistry,
        dm: DataManager,
        *args,
        probability_path: TorchProbabilityPath | None = None,
        match_fn: TorchMatchFn | None = None,
        noise_sampler: TorchNoiseSampler | None = None,
        time_sampler: TorchTimeSampler | None = None,
        generate_from_noise: bool = False,
        **kwargs,
    ) -> None:
        # initialize parent class
        super().__init__(dims_registry, dm, *args, **kwargs)

        # set attributes
        self._probability_path = probability_path
        self._match_fn = match_fn
        self._noise_sampler = noise_sampler
        self._time_sampler = time_sampler

        # automatically fall back to noise generation when
        # no control values are provided
        if not self.is_paired_setting:
            generate_from_noise = True
        self._generate_from_noise = generate_from_noise

    @property
    def generate_from_noise(self) -> bool:
        return self._generate_from_noise

    @property
    def probability_path(self) -> TorchProbabilityPath | None:
        return self._probability_path

    @property
    def match_fn(self) -> TorchMatchFn | None:
        return self._match_fn

    @property
    def noise_sampler(self) -> TorchNoiseSampler | None:
        return self._noise_sampler

    @property
    def time_sampler(self) -> TorchTimeSampler | None:
        return self._time_sampler
