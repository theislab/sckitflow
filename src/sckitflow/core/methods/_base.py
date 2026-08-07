import abc
from typing import Any

import torch

from sckitflow.core._data_utils import subscript_step_data
from sckitflow.core._types import PredictionData, StepData, TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sckitflow.core.nn._modules import BaseModule
from sckitflow.core.probability_paths import BaseProbabilityPath
from sckitflow.core.solvers import BaseSolver
from sckitflow.data._dims_registry import DataDimensionalitiesRegistry
from sckitflow.data._manager import DataManager

__all__ = ["BaseMethod", "GenerativeFlow"]


class BaseMethod(abc.ABC):
    _module_cls: type[BaseModule] | None = None

    def __init__(
        self,
        dims_registry: DataDimensionalitiesRegistry,
        dm: DataManager,
        *args,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
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

        # set attributes
        self._dtype = dtype
        self._device_id = device_id

        # move module to device
        self._module.to(self._dtype).to(self._device_id)

    @abc.abstractmethod
    def compute_loss(
        self,
        step_data: StepData,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...

    @abc.abstractmethod
    def infer(
        self,
        step_data: StepData,
        *args,
        **kwargs,
    ) -> PredictionData: ...

    def _train_step_forward(
        self,
        step_data: StepData,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        step_data = self._match_observations(step_data)
        return self.compute_loss(
            step_data,
            *args,
            **kwargs,
        )

    def _match_observations(
        self,
        step_data: StepData,
    ) -> StepData:
        return step_data

    def set_train_mode(self, mode: bool) -> None:
        """"""  # noqa
        if mode:
            self.module.train()
        else:
            self.module.eval()

    def train_step(
        self,
        step_data: StepData,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Single training step on a ready :class:`StepData` batch.

        Callers pass a :class:`StepData` already assembled by the data loaders.

        :param step_data: Ready-to-consume batch of torch tensors.
        :type step_data: class: `StepData`
        """
        return self._train_step_forward(step_data, *args, **kwargs)

    def predict(
        self,
        step_data: StepData,
        *args,
        no_grad: bool = True,
        **kwargs,
    ) -> PredictionData:
        """Prediction on a ready :class:`StepData` batch.

        Callers pass a :class:`StepData` already assembled by the data loaders.
        """
        # optionally stop gradients
        if no_grad:
            with torch.no_grad():
                return self.infer(
                    step_data,
                    *args,
                    **kwargs,
                )
        else:
            return self.infer(
                step_data,
                *args,
                **kwargs,
            )

    @property
    def module(self) -> BaseModule | None:
        return self._module

    @property
    def dm(self) -> DataManager | None:
        return self._dm

    @property
    def dims_registry(self) -> DataDimensionalitiesRegistry | None:
        return self._dims_registry

    @property
    def is_paired_setting(self) -> bool:
        return self._dm.control_values_dict is not None

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype


class GenerativeFlow(BaseMethod):
    _default_solver_cls: type[BaseSolver] | None = None

    def __init__(
        self,
        dims_registry: DataDimensionalitiesRegistry,
        dm: DataManager,
        *args,
        probability_path: BaseProbabilityPath | None = None,
        match_fn: TMatchFn | None = None,
        noise_sampler: TNoiseSamplerFn | None = None,
        time_sampler: TTimeSamplerFn | None = None,
        generate_from_noise: bool = False,
        **kwargs,
    ) -> None:
        # call parent constructor
        super().__init__(
            dims_registry,
            dm,
            *args,
            **kwargs,
        )

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

    def _call_match_fn_safe(
        self,
        source_lin: torch.Tensor | None,
        source_quad: torch.Tensor | None,
        target_lin: torch.Tensor | None,
        target_quad: torch.Tensor | None,
    ):
        # case 0: no source, do nothing
        if source_lin is None and source_quad is None:
            src_idxs = None
            tgt_idxs = None
            return src_idxs, tgt_idxs

        # case 1: source, match groups
        src_idxs, tgt_idxs = self._match_fn(
            source_lin=source_lin,
            target_lin=target_lin,
            source_quad=source_quad,
            target_quad=target_quad,
        )
        return src_idxs, tgt_idxs

    def _match_observations(
        self,
        step_data: StepData,
    ) -> StepData:
        # Get matching indices
        src_idxs, tgt_idxs = self._call_match_fn_safe(
            step_data["source_coupling_lin"],
            step_data["source_coupling_quad"],
            step_data["target_coupling_lin"],
            step_data["target_coupling_quad"],
        )

        # Case: no source distribution → return step_data unchanged
        if src_idxs is None and tgt_idxs is None:
            return step_data

        # Apply the matching permutation to both sides. The target side includes
        # ``target_response_data`` so target covariates stay row-aligned with ``target_state``.
        return subscript_step_data(step_data, src_idxs=src_idxs, tgt_idxs=tgt_idxs)

    @abc.abstractmethod
    def infer(
        self,
        step_data: StepData,
        *args,
        solver_cls: type[BaseSolver] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
        n_steps: int = 100,
        latent: torch.Tensor | None = None,
        n_samples: int | None = None,
        **kwargs,
    ) -> PredictionData: ...

    @property
    def generate_from_noise(self) -> bool:
        return self._generate_from_noise

    @property
    def probability_path(self) -> BaseProbabilityPath | None:
        return self._probability_path

    @property
    def match_fn(self) -> TMatchFn | None:
        return self._match_fn

    @property
    def noise_sampler(self) -> TNoiseSamplerFn | None:
        return self._noise_sampler

    @property
    def time_sampler(self) -> TTimeSamplerFn | None:
        return self._time_sampler
