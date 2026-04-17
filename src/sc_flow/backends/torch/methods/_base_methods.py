import abc
from typing import Any, Literal

import torch

from sc_flow.backends.torch._types import MappedTensor, TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sc_flow.backends.torch.methods._utils import OptimizationManager, TrainStepInput
from sc_flow.backends.torch.probability_paths import BaseProbabilityPath
from sc_flow.backends.torch.solvers import BaseSolver
from sc_flow.data._composite import MatchedDistributions
from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._state import StateData
from sc_flow.methods import BaseMethod

__all__ = ["BaseGenerativeFlow"]


class BaseGenerativeFlow(BaseMethod):
    def __init__(
        self,
        *args,
        probability_path: BaseProbabilityPath | None = None,
        match_fn: TMatchFn | None = None,
        noise_sampler: TNoiseSamplerFn | None = None,
        time_sampler: TTimeSamplerFn | None = None,
        generate_from_noise: bool = False,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
        solver_cls: type[BaseSolver] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr: float = 5e-5,
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_step: Literal["train_step", "validation_step"] = "train_step",
        plan_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        # set attributes
        self._dtype = dtype
        self._device_id = device_id

        super().__init__(
            *args,
            probability_path=probability_path,
            match_fn=match_fn,
            noise_sampler=noise_sampler,
            time_sampler=time_sampler,
            generate_from_noise=generate_from_noise,
            solver_cls=solver_cls,
            solver_kwargs=solver_kwargs,
            **kwargs,
        )

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

    def _batchmixin_to_torch(self, batch_mixin: BatchMixin) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(v).to(self._dtype).to(self._device_id) for k, v in batch_mixin.mapping.items()}

    @abc.abstractmethod
    def _compute_loss(
        self,
        latent: torch.Tensor,
        source: torch.Tensor | None,
        target: torch.Tensor,
        condition_data: MappedTensor,
        group_data: MappedTensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...

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
        if state_lin is not None:
            X_lin = state_lin.X
            X_lin = torch.from_numpy(X_lin).to(self._dtype).to(self._device_id)
        else:
            X_lin = None

        # get states for quadratic term
        if state_quad is not None:
            X_quad = state_quad.X
            X_quad = torch.from_numpy(X_quad).to(self._dtype).to(self._device_id)
        else:
            X_quad = None

        return X_lin, X_quad

    def _extract_state_data(
        self,
        distribution_data: DistributionData,
    ) -> torch.Tensor:
        state_data: StateData = distribution_data.state_data
        X_state = state_data.X
        X_state = torch.from_numpy(X_state).to(self._dtype).to(self._device_id)
        return X_state

    def _extract_distribution_data(
        self, distribution_data: DistributionData, mode: Literal["source", "target"]
    ) -> tuple[Any]:
        coupling_lin, coupling_quad = self._extract_coupling_data(distribution_data, mode)
        state_data = self._extract_state_data(distribution_data)
        condition_data = distribution_data.condition_data
        groups_data = distribution_data.groups_data
        return (coupling_lin, coupling_quad, state_data, condition_data, groups_data)

    def _extract_step_data(
        self,
        matched_distr: MatchedDistributions,
    ) -> TrainStepInput:
        # parse dictionary of matched distributions
        source_data_dict: DistributionData | None = matched_distr.source_distribution
        target_data_dict: DistributionData = matched_distr.target_distribution

        # parse target data dictionary
        (target_coupling_lin, target_coupling_quad, target_state_data, target_condition_data, target_group_data) = (
            self._extract_distribution_data(target_data_dict, "target")
        )

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
        return TrainStepInput(
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

    def _extract_matched_observations(
        self,
        step_data: TrainStepInput,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        # get matched indices
        src_idxs, tgt_idxs = self._call_match_fn_safe(
            step_data.source_coupling_lin,
            step_data.source_coupling_quad,
            step_data.target_coupling_lin,
            step_data.target_coupling_quad,
        )

        # if they are none, do no operation and early return
        if src_idxs is None and tgt_idxs is None:
            # states
            source = None
            target = step_data.target_state

            # extract condition and groups data
            condition_reps_dict: BatchMixin = step_data.target_condition_data.extract_reps()
            condition_reps_dict = self._batchmixin_to_torch(condition_reps_dict)

            # extract condition and groups data
            group_reps_dict: BatchMixin = step_data.target_group_data.extract_reps()
            group_reps_dict = self._batchmixin_to_torch(group_reps_dict)

            return source, target, condition_reps_dict, group_reps_dict

        # slice with matched indices
        source = step_data.source_state[src_idxs]
        target = step_data.target_state[tgt_idxs]
        condition_data = step_data.target_condition_data[tgt_idxs]
        group_data = step_data.target_group_data[tgt_idxs]

        # extract condition data
        condition_reps_dict = condition_data.extract_reps()
        condition_reps_dict = self._batchmixin_to_torch(condition_reps_dict)

        # extract groups data
        group_reps_dict = group_data.extract_reps()
        group_reps_dict = self._batchmixin_to_torch(group_reps_dict)

        return source, target, condition_reps_dict, group_reps_dict

    def _prepare_latent_state(
        self,
        source: torch.Tensor | None,
        target_reference: torch.Tensor,
    ) -> torch.Tensor:
        if source is None or self._generate_from_noise:
            # Tell the noise sampler to generate noise matching the target's shape
            return self._noise_sampler(target_reference)
        return source

    def _train_step_forward(
        self,
        step_data: TrainStepInput,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        latent = self._prepare_latent_state(step_data.source_state, step_data.target_state)
        source, target, condition_data, group_data = self._extract_matched_observations(step_data)
        return self._compute_loss(
            latent,
            source,
            target,
            condition_data,
            group_data,
        )

    def train_step(
        self,
        matched_distr: MatchedDistributions,
    ) -> tuple[float, dict[str, Any]]:
        """Single step function of the solver.

        :param batch: Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            ``condition``.
        :type batch: dict[str, torch.Tensor]
        """
        step_data = self._extract_step_data(matched_distr)
        loss, step_output = self._train_step_forward(step_data)
        self._optimization_manager.backward_pass(loss)
        return loss.item(), step_output
