import abc
from typing import Any, TypeVar

import torch

from sckitflow.core._data_utils import extract_step_data
from sckitflow.core._types import PredictionData, StepData, TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sckitflow.core.nn._modules import BaseModule
from sckitflow.core.probability_paths import BaseProbabilityPath
from sckitflow.core.solvers import BaseSolver
from sckitflow.data._composite import MatchedDistributions
from sckitflow.data._dims_registry import DataDimensionalitiesRegistry
from sckitflow.data._manager import DataManager

__all__ = ["BaseMethod", "GenerativeFlow"]

T = TypeVar("T")


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

    @staticmethod
    def _safe_subscript_obj(data: T | None, idx: Any | None) -> T | None:  # TODO: Probably remove from here
        if data is None:
            return None
        if idx is None:
            return data
        return data[idx]

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
        matched_distr: MatchedDistributions,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Single step function of the solver.

        :param matched_distr: Input `MatchedDistributions` object.
        :type matched_distr: dict[str, torch.Tensor]
        """
        step_data = extract_step_data(matched_distr, device=self._device_id, dtype=self._dtype)
        return self._train_step_forward(step_data, *args, **kwargs)

    def predict(
        self,
        data: MatchedDistributions | StepData,
        *args,
        no_grad: bool = True,
        **kwargs,
    ) -> PredictionData:
        """Prediction on node."""
        # extract step data and prepare latent state
        if isinstance(data, MatchedDistributions):
            data = extract_step_data(data, device=self._device_id, dtype=self._dtype)
        if not isinstance(data, StepData):
            raise ValueError(f"Data is of the wrong type, expected `StepData`, but {type(data)} found.")

        # optionally stop gradients
        if no_grad:
            with torch.no_grad():
                return self.infer(
                    data,
                    *args,
                    **kwargs,
                )
        else:
            return self.infer(
                data,
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
        return self._dm.control_values_dict is not None or self._dm.matched_keys is not None

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
            step_data.source_coupling_lin,
            step_data.source_coupling_quad,
            step_data.target_coupling_lin,
            step_data.target_coupling_quad,
        )

        # Case: no source distribution → return step_data unchanged (or with source=None)
        if src_idxs is None and tgt_idxs is None:
            # Already no source; keep target as is
            return step_data

        # Slice source side
        source_state = self._safe_subscript_obj(step_data.source_state, src_idxs)
        source_condition_data = self._safe_subscript_obj(step_data.source_condition_data, src_idxs)
        source_group_data = self._safe_subscript_obj(step_data.source_group_data, src_idxs)
        source_coupling_lin = self._safe_subscript_obj(step_data.source_coupling_lin, src_idxs)
        source_coupling_quad = self._safe_subscript_obj(step_data.source_coupling_quad, src_idxs)

        # Slice target side
        target_state = self._safe_subscript_obj(step_data.target_state, tgt_idxs)
        target_condition_data = self._safe_subscript_obj(step_data.target_condition_data, tgt_idxs)
        target_group_data = self._safe_subscript_obj(step_data.target_group_data, tgt_idxs)
        target_coupling_lin = self._safe_subscript_obj(step_data.target_coupling_lin, tgt_idxs)
        target_coupling_quad = self._safe_subscript_obj(step_data.target_coupling_quad, tgt_idxs)

        # Return new StepData with matched slices
        return StepData(
            target_state=target_state,
            target_coupling_lin=target_coupling_lin,
            target_coupling_quad=target_coupling_quad,
            target_condition_data=target_condition_data,
            target_group_data=target_group_data,
            source_state=source_state,
            source_coupling_lin=source_coupling_lin,
            source_coupling_quad=source_coupling_quad,
            source_condition_data=source_condition_data,
            source_group_data=source_group_data,
        )

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
