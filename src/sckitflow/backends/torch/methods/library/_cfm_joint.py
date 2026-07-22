from collections.abc import Sequence
from typing import Any

import torch

from sckitflow.backends.torch._types import PredictionData
from sckitflow.backends.torch.methods.library._cfm import CFM
from sckitflow.backends.torch.methods._utils import StepData
from sckitflow.backends.torch.solvers import BaseSolver
from sckitflow.data._composite import MatchedDistributions

__all__ = ["CFM_Joint"]


class CFM_Joint(CFM):
    """CFM with joint multi-time transition training.

    All timepoint transitions are processed together in one forward pass.

    :param timepoints: Sequence of timepoints (length = n_transitions + 1).
    """

    def __init__(self, *args, timepoints: Sequence[float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        if timepoints is None:
            raise ValueError("CFM-J requires `timepoints` to be provided.")
        self._timepoints = tuple(timepoints)

    @property
    def is_joint(self) -> bool:
        return True

    @property
    def timepoints(self) -> tuple[float, ...]:
        return self._timepoints

    # ------------------------------------------------------------------
    # Joint multi-time transition training
    # ------------------------------------------------------------------
    def train_step_joint(
        self,
        nodes: tuple[MatchedDistributions, ...],
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Joint training over all transitions in one forward pass.
        """
        timepoints = self._timepoints

        if len(nodes) != len(timepoints) - 1:
            raise ValueError(
                f"Expected {len(timepoints) - 1} nodes for {len(timepoints)} timepoints, "
                f"got {len(nodes)}."
            )

        all_t = []
        all_xt = []
        all_ut = []
        all_cond = []
        all_latent = []

        for i, node in enumerate(nodes):
            # Extract per transition
            step_data = self._extract_step_data(node)
            step_data = self._extract_matched_observations(step_data)

            target = step_data.target_state
            source = step_data.source_state
            condition_data = self._get_tensor_dict_from_data(step_data.target_condition_data)
            group_data = self._get_tensor_dict_from_data(step_data.target_group_data)

            latent = self._prepare_latent_train(source, target)
            batch_size = latent.shape[0]

            # Time scaling
            t_start = timepoints[i]
            t_end = timepoints[i + 1]
            delta_t = t_end - t_start
            t_local = self._time_sampler((batch_size,), device=latent.device, dtype=latent.dtype)
            t_global = t_local * delta_t + t_start

            # Probability path: interpolant and velocity (scaled by 1/delta_t)
            xt = self._probability_path.compute_xt(t_local, latent, target)
            ut = self._probability_path.compute_ut(t_local, xt, latent, target) / delta_t

            cond = {**condition_data, **group_data}

            all_t.append(t_global)
            all_xt.append(xt)
            all_ut.append(ut)
            all_cond.append(cond)
            all_latent.append(latent)

        # Concatenate across transitions
        t_all = torch.cat(all_t)
        xt_all = torch.cat(all_xt)
        ut_all = torch.cat(all_ut)
        latent_all = torch.cat(all_latent)
        cond_all = {k: torch.cat([c[k] for c in all_cond]) for k in all_cond[0]}

        # Single forward pass
        vt_all = self._module(t_all, xt_all, condition_dict=cond_all, source=latent_all)
        loss = torch.nn.functional.mse_loss(vt_all, ut_all)

        return loss, {"loss": loss.item()}

    # ------------------------------------------------------------------
    # Prediction with t_start / t_end
    # ------------------------------------------------------------------
    def _predict(
        self,
        step_data: StepData,
        *args,
        solver_cls: type[BaseSolver] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
        num_steps: int = 100,
        latent: torch.Tensor | None = None,
        n_samples: int | None = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
        **kwargs,
    ) -> PredictionData:
        # ----- 1. Prepare latent (noise) -----
        if latent is None:
            latent = self._prepare_latent_inference(
                step_data.source_state,
                step_data.target_state,
                n_samples=n_samples,
                n_obs=step_data.target_n_obs,
            )

        # ----- 2. Build conditioning dict -----
        condition_reps_dict = self._get_tensor_dict_from_data(step_data.target_condition_data)
        group_reps_dict = self._get_tensor_dict_from_data(step_data.target_group_data)
        condition_dict = {**condition_reps_dict, **group_reps_dict}

        # ----- 3. Expand conditioning to match latent dimensions -----
        condition_dict, source_expanded = self._expand_conditioning(latent, condition_dict, step_data.source_state)

        # ----- 4. Configure ODE solver -----
        if solver_kwargs is None:
            solver_kwargs = {}
        solver_kwargs.setdefault("method", "euler")
        method = solver_kwargs.pop("method")

        if solver_cls is None:
            solver_cls = self._default_solver_cls

        time_grid = torch.linspace(t_start, t_end, steps=num_steps, device=latent.device, dtype=latent.dtype)
        solver = solver_cls(
            self._module,
            *args,
            method=method,
            vf_kwargs={"condition_dict": condition_dict, "source": source_expanded},
            device_id=self._device_id,
            **kwargs,
        )

        # ----- 5. Integrate -----
        predictions = solver.solve(
            latent,
            time_grid,
            solver_kwargs=solver_kwargs,
            return_trajectory=return_trajectory,
        )

        # ----- 6. Aggregate predictions (averaging, reshaping) -----
        X, traj, raw_samples = self._aggregate_predictions(
            predictions=predictions,
            latent_shape=latent.shape,
            return_trajectory=return_trajectory,
        )
        return PredictionData(X=X, traj=traj, raw_samples=raw_samples)
