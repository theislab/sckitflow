import abc
from typing import Any, Literal

import torch

from sc_flow.backends.torch._types import MappedTensor, TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sc_flow.backends.torch.methods._utils import TrainStepInput
from sc_flow.backends.torch.nn import BaseModule
from sc_flow.backends.torch.probability_paths import BaseProbabilityPath
from sc_flow.backends.torch.solvers import BaseSolver
from sc_flow.methods import BaseMethod

__all__ = ["BaseGenerativeFlow"]


class BaseGenerativeFlow(abc.ABC, BaseMethod):
    def __init__(
        self,
        module: BaseModule,
        probability_path: BaseProbabilityPath,
        match_fn: TMatchFn,
        noise_sampler: TNoiseSamplerFn,
        time_sampler: TTimeSamplerFn,
        generate_from_noise: bool,
        solver_cls: type[BaseSolver],
    ) -> None:
        super().__init__(
            module,
            probability_path,
            match_fn,
            solver_cls,
            noise_sampler,
            time_sampler,
            generate_from_noise,
        )

    @abc.abstractmethod
    def _compute_loss(
        self,
        latent: torch.Tensor,
        source: torch.Tensor | None,
        target: torch.Tensor,
        condition_data: MappedTensor,
        group_data: MappedTensor,
    ) -> torch.Tensor: ...

    def _call_match_fn_safe(
        self,
        source_lin: torch.Tensor | None,
        source_quad: torch.Tensor | None,
        target_lin: torch.Tensor | None,
        target_quad: torch.Tensor | None,
    ):
        # case 0: no source
        if source_lin is None and source_quad is None:
            src_idxs = None
            tgt_idxs = None
            return src_idxs, tgt_idxs

        # case 1: source
        src_idxs, tgt_idxs = self._match_fn(
            source_lin,
            source_quad,
            target_lin,
            target_quad,
        )
        return src_idxs, tgt_idxs

    def _extract_coupling_data(
        self,
        distribution_dict: dict[str, Any],
        mode: Literal["source", "target"],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # retrieve coupling data dictionary
        coupling_data: dict[str, torch.Tensor] = distribution_dict.get(f"{mode}_coupling_data")

        # parse coupling data dictionary
        state_lin: torch.Tensor | None = coupling_data.get("state_lin")
        state_quad: torch.Tensor | None = coupling_data.get("state_quad")
        return state_lin, state_quad

    def _extract_state_data(
        self,
        distribution_dict: dict[str, Any],
    ) -> torch.Tensor:
        state_data: dict[str, torch.Tensor] = distribution_dict.get("state_data")
        return state_data["X"]

    def _extract_distribution_data(self, data_dict: dict[str, Any], mode: Literal["source", "target"]) -> tuple[Any]:
        coupling_lin, coupling_quad = self._extract_coupling_data(data_dict, mode)
        state_data = self._extract_state_data(data_dict)
        condition_data = data_dict.get("condition_data")
        group_data = data_dict.get("group_data")
        return (coupling_lin, coupling_quad, state_data, condition_data, group_data)

    def _extract_step_data(
        self,
        batch_dict: dict[str, Any],
    ) -> TrainStepInput:
        # parse dictionary of matched distributions
        source_data_dict: dict[str, Any] | None = batch_dict.get("source_distribution")
        target_data_dict: dict[str, Any] = batch_dict.get("target_distributions")

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # get matched indices
        src_idxs, tgt_idxs = self._call_match_fn_safe(
            step_data.source_coupling_lin,
            step_data.source_coupling_quad,
            step_data.target_coupling_lin,
            step_data.target_coupling_quad,
        )

        # if they are none, do no operation and early return
        if src_idxs is None and tgt_idxs is None:
            source = ...
            return step_data

        # slice with matched indices
        source = step_data.source_state[src_idxs]
        target = step_data.target_state[tgt_idxs]
        condition_data = step_data.target_condition_data[tgt_idxs]
        group_data = step_data.target_group_data[tgt_idxs]
        return source, target, condition_data, group_data

    def _prepare_latent_state(
        self,
        source: torch.Tensor,
    ) -> torch.Tensor:
        if source is None or self._generate_from_noise:
            return self._noise_sampler(source)
        return source

    def _train_step(
        self,
        step_data: TrainStepInput,
    ) -> torch.Tensor:
        latent = self._prepare_latent_state(step_data.source_state)
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
        batch_dict: dict[str, Any],
    ) -> torch.Tensor:
        """Single step function of the solver.

        :param batch: Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            ``condition``.
        :type batch: dict[str, torch.Tensor]
        """
        step_data = self._extract_step_data(batch_dict)
        return self._train_step(step_data)
