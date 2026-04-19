import abc
from typing import Any, Literal, TypeVar

import torch

from sc_flow.backends.torch._types import PredictionData, TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sc_flow.backends.torch.methods._utils import OptimizationManager, StepData
from sc_flow.backends.torch.nn._modules import BaseModule
from sc_flow.backends.torch.probability_paths import BaseProbabilityPath
from sc_flow.backends.torch.solvers import BaseSolver
from sc_flow.data._composite import MatchedDistributions
from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.data.containers._state import StateData
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
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr: float = 5e-5,
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_step: Literal["train_step", "validation_step"] = "train_step",
        plan_kwargs: dict[str, Any] | None = None,
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

        # initialize optimization manager
        self._optimization_manager = OptimizationManager.init_from_module(
            self.module,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_step=lr_scheduler_step,
            plan_kwargs=plan_kwargs,
        )

    @staticmethod
    def _safe_subscript_obj(data: T | None, idx: Any | None) -> T | None:
        if data is None:
            return None
        if idx is None:
            return data
        return data[idx]

    def _batchmixin_to_torch(self, batch_mixin: BatchMixin) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(v).to(self._dtype).to(self._device_id) for k, v in batch_mixin.mapping.items()}

    def set_train_mode(self, mode: bool) -> None:
        if mode:
            self.module.train()
        else:
            self.module.eval()

    def _extract_state_data(
        self,
        state_data: StateData | None,
    ) -> torch.Tensor | None:
        if state_data is None:
            return None
        X_state = state_data.X
        X_state = torch.from_numpy(X_state).to(self._dtype).to(self._device_id)
        return X_state

    def _extract_coupling_data(
        self,
        distribution_data: DistributionData,
        mode: Literal["source", "target"],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # retrieve coupling data
        coupling_data: CouplingData = getattr(distribution_data, f"{mode}_coupling_data")

        # parse coupling data
        state_lin: StateData | None = coupling_data.state_lin
        state_quad: StateData | None = coupling_data.state_quad

        # get states for linear term
        X_lin = self._extract_state_data(state_lin)
        X_quad = self._extract_state_data(state_quad)
        return X_lin, X_quad

    def _extract_distribution_data(
        self, distribution_data: DistributionData, mode: Literal["source", "target"]
    ) -> tuple[Any]:
        coupling_lin, coupling_quad = self._extract_coupling_data(distribution_data, mode)
        state_data = self._extract_state_data(distribution_data.state_data)
        condition_data = distribution_data.condition_data
        groups_data = distribution_data.groups_data
        return (coupling_lin, coupling_quad, state_data, condition_data, groups_data)

    def _get_tensor_dict_from_data(self, data: MixedTypeData | CategoricalData | None) -> dict[str, torch.Tensor]:
        if data is None:
            return {}
        group_reps_dict = data.extract_reps()
        return self._batchmixin_to_torch(group_reps_dict)

    def _extract_step_data(
        self,
        matched_distr: MatchedDistributions,
    ) -> StepData:
        # parse dictionary of matched distributions
        source_data_dict: DistributionData | None = matched_distr.source_distribution
        target_data_dict: DistributionData | None = matched_distr.target_distribution

        # parse target data dictionary
        if target_data_dict is not None:
            (target_coupling_lin, target_coupling_quad, target_state_data, target_condition_data, target_group_data) = (
                self._extract_distribution_data(target_data_dict, "target")
            )
        else:
            target_coupling_lin = None
            target_coupling_quad = None
            target_state_data = None
            target_condition_data = None
            target_group_data = None

        # optionally parse target data dictionary
        if source_data_dict is not None:
            (source_coupling_lin, source_coupling_quad, source_state_data, source_condition_data, source_group_data) = (
                self._extract_distribution_data(source_data_dict, "source")
            )
        else:
            source_coupling_lin = None
            source_coupling_quad = None
            source_state_data = None
            source_condition_data = None
            source_group_data = None

        # return structured output
        return StepData(
            target_state_data,
            target_coupling_lin,
            target_coupling_quad,
            target_condition_data,
            target_group_data,
            source_state_data,
            source_coupling_lin,
            source_coupling_quad,
            source_condition_data,
            source_group_data,
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

    @abc.abstractmethod
    def _compute_loss(
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

    def _extract_matched_observations(
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

    def _train_step_forward(
        self,
        step_data: StepData,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        step_data = self._extract_matched_observations(step_data)
        return self._compute_loss(
            step_data,
            *args,
            **kwargs,
        )

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
        step_data = self._extract_step_data(matched_distr)
        loss, step_dict = self._train_step_forward(step_data, *args, **kwargs)
        self._optimization_manager.backward_pass(loss)
        return step_dict

    def predict(
        self,
        matched_distr: MatchedDistributions,
        *args,
        no_grad: bool = True,
        **kwargs,
    ) -> PredictionData:
        """Prediction on node."""
        # extract step data and prepare latent state
        step_data = self._extract_step_data(matched_distr)

        # optionally stop gradients
        if no_grad:
            with torch.no_grad():
                return self._predict(
                    step_data,
                    *args,
                    **kwargs,
                )
        else:
            return self._predict(
                step_data,
                *args,
                **kwargs,
            )
