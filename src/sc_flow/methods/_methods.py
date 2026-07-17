from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from sc_flow.config._capabilities import MethodCapabilities
from sc_flow.data._dims_registry import DataDimensionalitiesRegistry
from sc_flow.data._manager import DataManager
from sc_flow.data.containers._state import StateData

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

__all__ = ["BaseMethod", "BaseGenerativeFlow"]


class BaseMethod(abc.ABC):
    #: Method capabilities used by the config/builder layer for generic
    #: validation. Subclasses (or ``register_method``) may override; the default
    #: is a permissive "general" descriptor.
    _capabilities: MethodCapabilities | None = None

    @classmethod
    def capabilities(cls) -> MethodCapabilities:
        """Return this method's :class:`MethodCapabilities` descriptor."""
        return cls._capabilities if cls._capabilities is not None else MethodCapabilities()

    def __init__(
        self,
        dims_registry: DataDimensionalitiesRegistry,
        dm: DataManager,
        is_paired_setting: bool,
        *args,
        **kwargs,
    ) -> None:
        # initialize attributes
        self._dims_registry = dims_registry
        self._dm = dm
        self._is_paired_setting = is_paired_setting

        # build the method's module through the overridable construction seam
        self._module = self.build_module(*args, **kwargs)

    def build_module(self, *args: Any, **kwargs: Any) -> "JaxModule | TorchModule":
        """Construct this method's neural module from the dimensionality registry.

        The construction seam every method must provide — either a concrete method
        (e.g. :class:`~sc_flow.backends.torch.methods.library._cfm.CFM`) or one built
        by :func:`~sc_flow.methods._custom.register_method` from a user class's
        ``module_cls``. Kept a method (not a class attribute) so construction is
        polymorphic and test doubles override it per subclass rather than mutating a
        shared class global.
        """
        raise NotImplementedError(f"{type(self).__name__} must override `build_module`.")

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

    def make_lightning_module(self, optim_config: Any, grad_clip: float | None = None) -> Any:
        """Build the ``LightningModule`` that trains this method (the harness seam).

        Default: wrap this method in :class:`~sc_flow.trainer._lightning.LitSCFlowModule`,
        which drives training by calling :meth:`train_step` (all numerics in torch).
        Override to train under a different ``LightningModule`` — e.g. the JAX-compute
        bridge returns a ``CellFlowJaxModule`` (numerics in JAX, torch only optimizes).

        This is the seam that lets both training paths flow through the one
        :func:`~sc_flow.trainer._lightning.fit_with_lightning` entry point.
        """
        from sc_flow.trainer._lightning import LitSCFlowModule

        return LitSCFlowModule(self, optim_config, grad_clip=grad_clip)

    def make_datamodule(self, train_sampler: Any, n_steps: int, val_samplers_dict: dict[str, Any] | None = None) -> Any:
        """Build the ``LightningDataModule`` feeding :meth:`make_lightning_module`.

        Default: :class:`~sc_flow.trainer._lightning.SCFlowDataModule`, which yields the
        sampler "nodes" that :class:`LitSCFlowModule` consumes. Override alongside
        :meth:`make_lightning_module` when a training path needs a different batch shape
        (e.g. the JAX bridge wants tensor dicts).
        """
        from sc_flow.trainer._lightning import SCFlowDataModule

        return SCFlowDataModule(train_sampler, n_steps, val_samplers_dict)

    @property
    def module(self) -> JaxModule | TorchModule | None:
        return self._module

    @property
    def dm(self) -> DataManager | None:
        return self._dm

    @property
    def dims_registry(self) -> DataDimensionalitiesRegistry | None:
        return self._dims_registry

    @property
    def is_paired_setting(self) -> bool:
        return self._is_paired_setting


class BaseGenerativeFlow(BaseMethod):
    _default_solver_cls: type[JaxSolver | TorchSolver] | None = None

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
        **kwargs,
    ) -> None:
        # initialize parent class
        super().__init__(dims_registry, dm, is_paired_setting, *args, **kwargs)

        # set attributes
        self._probability_path = probability_path
        self._match_fn = match_fn
        self._noise_sampler = noise_sampler
        self._time_sampler = time_sampler

        # automatically fall back to noise generation when
        # no control values are provided
        if not self._is_paired_setting:
            generate_from_noise = True
        self._generate_from_noise = generate_from_noise

    @property
    def generate_from_noise(self) -> bool:
        return self._generate_from_noise

    @property
    def probability_path(self) -> JaxProbabilityPath | TorchProbabilityPath | None:
        return self._probability_path

    @property
    def match_fn(self) -> JaxMatchFn | TorchMatchFn | None:
        return self._match_fn

    @property
    def noise_sampler(self) -> JaxNoiseSampler | TorchNoiseSampler | None:
        return self._noise_sampler

    @property
    def time_sampler(self) -> JaxTimeSampler | TorchTimeSampler | None:
        return self._time_sampler
