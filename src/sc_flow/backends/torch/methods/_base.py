import abc
from typing import Any, TypeVar

import torch

from sc_flow.backends.torch._data_utils import extract_step_data
from sc_flow.backends.torch._types import PredictionData, StepData, TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sc_flow.backends.torch.nn._modules import BaseModule
from sc_flow.backends.torch.probability_paths import BaseProbabilityPath
from sc_flow.backends.torch.solvers import BaseSolver
from sc_flow.data._composite import MatchedDistributions
from sc_flow.methods._methods import BaseGenerativeFlow, BaseMethod

__all__ = ["TorchBaseMethod", "TorchGenerativeFlow"]

T = TypeVar("T")


class TorchBaseMethod(BaseMethod):
    _module_cls: type[BaseModule] | None = None

    def __init__(
        self,
        *args,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> None:
        # call constructor of parent class
        super().__init__(
            *args,
            **kwargs,
        )

        # set attributes
        self._dtype = dtype
        self._device_id = device_id

        # move module to device
        self._module.to(self._dtype).to(self._device_id)

    @staticmethod
    def _safe_subscript_obj(data: T | None, idx: Any | None) -> T | None:
        if data is None:
            return None
        if idx is None:
            return data
        return data[idx]

    @abc.abstractmethod
    def _step_fn(
        self,
        step_data: StepData,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...

    @abc.abstractmethod
    def _predict(
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
        return self._step_fn(
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
                return self._predict(
                    data,
                    *args,
                    **kwargs,
                )
        else:
            return self._predict(
                data,
                *args,
                **kwargs,
            )


class TorchGenerativeFlow(BaseGenerativeFlow, TorchBaseMethod):
    _default_solver_cls: type[BaseSolver] | None = None

    def __init__(
        self,
        *args,
        probability_path: BaseProbabilityPath | None = None,
        match_fn: TMatchFn | None = None,
        noise_sampler: TNoiseSamplerFn | None = None,
        time_sampler: TTimeSamplerFn | None = None,
        generate_from_noise: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            probability_path=probability_path,
            match_fn=match_fn,
            noise_sampler=noise_sampler,
            time_sampler=time_sampler,
            generate_from_noise=generate_from_noise,
            **kwargs,
        )

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
    def _predict(
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
