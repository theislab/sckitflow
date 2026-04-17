from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from sc_flow.data._dims_registry import DataDimensionalitiesRegistry
from sc_flow.data._manager import DataManager

if TYPE_CHECKING:
    # jax backend
    from sc_flow.backends.jax._types import TMatchFn as JaxMatchFn
    from sc_flow.backends.jax._types import TNoiseSamplerFn as JaxNoiseSampler
    from sc_flow.backends.jax._types import TTimeSamplerFn as JaxTimeSampler
    from sc_flow.backends.jax.nn import BaseModule as JaxModule
    from sc_flow.backends.jax.probability_paths import BaseProbabilityPath as JaxProbabilityPath
    from sc_flow.backends.jax.solvers import BaseSolver as JaxSolver

    # torch backend
    from sc_flow.backends.torch._types import TMatchFn as TorchMatchFn
    from sc_flow.backends.torch._types import TNoiseSamplerFn as TorchNoiseSampler
    from sc_flow.backends.torch._types import TTimeSamplerFn as TorchTimeSampler
    from sc_flow.backends.torch.nn import BaseModule as TorchModule
    from sc_flow.backends.torch.probability_paths import BaseProbabilityPath as TorchProbabilityPath
    from sc_flow.backends.torch.solvers import BaseSolver as TorchSolver

__all__ = ["BaseMethod"]


class BaseMethod(abc.ABC):
    _module_cls: type[JaxModule | TorchModule]

    def __init__(
        self,
        dims_registry: DataDimensionalitiesRegistry,
        dm: DataManager,
        is_paired_setting: bool,
        *args,
        probability_path: JaxProbabilityPath | TorchProbabilityPath | None = None,
        match_fn: JaxMatchFn | TorchMatchFn | None = None,
        noise_sampler: JaxNoiseSampler | TorchNoiseSampler | None = None,
        time_sampler: JaxTimeSampler | TorchTimeSampler | None = None,
        generate_from_noise: bool = False,
        solver_cls: JaxSolver | TorchSolver | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        # initialize attributes
        self._dims_registry = dims_registry
        self._dm = dm
        self._is_paired_setting = is_paired_setting
        self._probability_path = probability_path
        self._match_fn = match_fn
        self._noise_sampler = noise_sampler
        self._time_sampler = time_sampler
        self._solver_cls = solver_cls
        self._solver_kwargs = solver_kwargs

        # automatically fall back to noise generation when
        # no control values are provided
        if not self._is_paired_setting:
            generate_from_noise = True
        self._generate_from_noise = generate_from_noise

        # initialize module with dimensionality registry
        self._module = self._module_cls.init_from_dims_registry(self._dims_registry, *args, **kwargs)

    @abc.abstractmethod
    def train_step(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @property
    def probability_path(self) -> JaxProbabilityPath | TorchProbabilityPath | None:
        return self._probability_path

    @property
    def match_fn(self) -> JaxMatchFn | TorchMatchFn | None:
        return self._match_fn

    @property
    def solver_cls(self) -> JaxSolver | TorchSolver | None:
        return self._solver_cls

    @property
    def noise_sampler(self) -> JaxNoiseSampler | TorchNoiseSampler | None:
        return self._noise_sampler

    @property
    def time_sampler(self) -> JaxTimeSampler | TorchTimeSampler | None:
        return self._time_sampler

    @property
    def module(self) -> JaxModule | TorchModule | None:
        return self._module

    @property
    def generate_from_noise(self) -> bool:
        return self._generate_from_noise

    @property
    def dm(self) -> DataManager | None:
        return self._dm

    @property
    def dims_registry(self) -> DataDimensionalitiesRegistry | None:
        return self._dims_registry
