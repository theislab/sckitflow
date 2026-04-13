from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sc_flow.backends.jax._types import TMatchFn as JaxMatchFn
    from sc_flow.backends.jax._types import TNoiseSamplerFn as JaxNoiseSampler
    from sc_flow.backends.jax._types import TTimeSamplerFn as JaxTimeSampler
    from sc_flow.backends.jax.nn import BaseModule as JaxModule
    from sc_flow.backends.jax.probability_paths import BaseProbabilityPath as JaxProbabilityPath
    from sc_flow.backends.jax.solvers import BaseSolver as JaxSolver
    from sc_flow.backends.torch._types import TMatchFn as TorchMatchFn
    from sc_flow.backends.torch._types import TNoiseSamplerFn as TorchNoiseSampler
    from sc_flow.backends.torch._types import TTimeSamplerFn as TorchTimeSampler
    from sc_flow.backends.torch.nn import BaseModule as TorchModule
    from sc_flow.backends.torch.probability_paths import BaseProbabilityPath as TorchProbabilityPath
    from sc_flow.backends.torch.solvers import BaseSolver as TorchSolver

__all__ = ["BaseMethod"]


class BaseMethod(abc.ABC):
    def __init__(
        self,
        module: JaxModule | TorchModule,
        probability_path: JaxProbabilityPath | TorchProbabilityPath,
        match_fn: JaxMatchFn | TorchMatchFn,
        solver: JaxSolver | TorchSolver,
        noise_sampler: JaxNoiseSampler | TorchNoiseSampler,
        time_sampler: JaxTimeSampler | TorchTimeSampler,
        generate_from_noise: bool,
    ) -> None:
        self._module = module
        self._probability_path = probability_path
        self._match_fn = match_fn
        self._solver = solver
        self._noise_sampler = noise_sampler
        self._time_sampler = time_sampler
        self._generate_from_noise = generate_from_noise

    @abc.abstractmethod
    def train_step(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @property
    def probability_path(self) -> JaxProbabilityPath | TorchProbabilityPath:
        return self._probability_path

    @property
    def match_fn(self) -> JaxMatchFn | TorchMatchFn:
        return self._match_fn

    @property
    def solver(self) -> JaxSolver | TorchSolver:
        return self._solver

    @property
    def noise_sampler(self) -> JaxNoiseSampler | TorchNoiseSampler:
        return self._noise_sampler

    @property
    def time_sampler(self) -> JaxTimeSampler | TorchTimeSampler:
        return self._time_sampler

    @property
    def module(self) -> JaxModule | TorchModule:
        return self._module

    @property
    def generate_from_noise(self) -> bool:
        return self._generate_from_noise
