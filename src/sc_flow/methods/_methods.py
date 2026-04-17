from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from anndata import AnnData

from sc_flow.data._dim import DataDimensionalitiesRegistry
from sc_flow.data._manager import DataManager

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
    _module_cls: type[JaxModule] | type[TorchModule]
    _dm_cls: DataManager | None = None
    _data_dimensionalities_register_cls: Any | None = None

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

        # check that data was prepared
        if self.__class__._dm_cls is None:
            raise RuntimeError(
                f"Data has not been registered with {self.__class__.__name__}. "
                "Please call .register_adata(adata, ...) before initializing the model."
            )

        # register class attributes to instance
        self._dm = self.__class__._dm_cls
        self._data_dimensionalities_register = self.__class__._data_dimensionalities_register_cls

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

    @property
    def dm(self) -> DataManager | None:
        return self._dm

    @property
    def data_dimensionalities_register(self) -> DataDimensionalitiesRegistry | None:
        return self._data_dimensionalities_register

    @abc.abstractmethod
    def _init_module(
        self,
    ) -> JaxModule | TorchModule: ...

    @classmethod
    def register_adata(
        cls,
        adata: AnnData,
        **kwargs,
    ) -> None:
        # initialize data manager
        cls._dm_cls = DataManager(**kwargs)
        cls._data_dimensionalities_register_cls = cls._dm_cls.get_data_dimensionalities(adata)
