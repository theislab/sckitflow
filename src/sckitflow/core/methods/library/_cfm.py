from typing import Any

import torch

from sckitflow.core._data_utils import (
    expand_conditioning,
    get_tensor_dict_from_data,
    prepare_latent_inference,
    prepare_latent_train,
)
from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.coupling._coupling import independent_coupling
from sckitflow.core.methods._base import GenerativeFlow
from sckitflow.core.nn._vf import BaseVelocityField, MLPVelocity
from sckitflow.core.probability_paths._probability_paths import LinearDiracProbabilityPath
from sckitflow.core.solvers import BaseSolver, ODESolver

__all__ = ["CFM"]


class CFM(GenerativeFlow):
    _module_cls: type[BaseVelocityField] = MLPVelocity
    _default_solver_cls: type[BaseSolver] = ODESolver

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._match_fn is None:
            self._match_fn = independent_coupling
        if self._noise_sampler is None:
            self._noise_sampler = torch.randn
        if self._time_sampler is None:
            self._time_sampler = torch.rand
        if self._probability_path is None:
            self._probability_path = LinearDiracProbabilityPath()

    def compute_loss(self, step_data: StepData, *args, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        target = step_data["target_state"]
        source = step_data["source_state"]

        # condition data as tensors
        condition_data = get_tensor_dict_from_data(step_data["target_condition_data"])
        group_data = get_tensor_dict_from_data(step_data["target_group_data"])
        cond = {**condition_data, **group_data}

        # latent (noise) – shape (batch_size, dim)
        latent = prepare_latent_train(
            source,
            target,
            self._noise_sampler,
            generate_from_noise=self._generate_from_noise,
        )
        batch_size = latent.shape[0]

        # sample time
        t = self._time_sampler((batch_size,), device=latent.device, dtype=latent.dtype)

        # probability path: interpolant and velocity
        xt = self._probability_path.compute_xt(t, latent, target)
        ut = self._probability_path.compute_ut(t, xt, latent, target)

        # velocity prediction
        vt = self._module(t, xt, condition_dict=cond, source=source)

        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss, {"loss": loss.item()}

    def _aggregate_predictions(
        self,
        predictions: torch.Tensor,
        latent_shape: torch.Size,
        return_trajectory: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Convert solver output into final X, traj, and raw_samples.

        Parameters
        ----------
        predictions : torch.Tensor
            Solver output from ``ODESolver``. If ``return_trajectory=True``, the shape is
            ``(num_steps, *latent_shape)``. Otherwise, the shape is ``latent_shape``.
        latent_shape : torch.Size
            Original shape of the latent tensor (for example,
            ``(n_samples, batch_size, dim)`` or ``(batch_size, dim)``).
        return_trajectory : bool
            Whether the solver returned a full trajectory.

        Returns
        -------
        X : torch.Tensor
            Final predicted state. If ``latent_shape`` includes a sample dimension,
            this is the mean of the final states over samples with shape
            ``(batch_size, dim)``. Otherwise, this is the final state tensor with
            shape ``(batch_size, dim)``.
        traj : torch.Tensor or None
            Full trajectory tensor, not averaged over samples. When a sample
            dimension is present, the shape is ``(num_steps, n_samples, batch_size, dim)``.
            Otherwise, the returned tensor includes a singleton sample dimension
            and has shape ``(num_steps, 1, batch_size, dim)``. Returns ``None`` if
            ``return_trajectory=False``.
        raw_samples : torch.Tensor or None
            Raw final states for each sample with shape ``(n_samples, batch_size, dim)``
            when a sample dimension is present, or ``None`` otherwise.
        """
        # ---- Determine if we have a sample dimension from latent state ----
        n_latent_dims = len(latent_shape)
        if n_latent_dims == 3:
            n_samples = latent_shape[0]
            batch_size = latent_shape[1]
            has_sample_dim = average_samples = True
        elif n_latent_dims == 2:
            has_sample_dim = average_samples = False
        else:
            raise ValueError(f"Invalid shape for source latent state: {latent_shape}")

        # ---- Reshape output when needed ----
        if has_sample_dim:
            # predictions will be of shape (T, N, B, D)
            n_steps = predictions.shape[0]
            # ---- Handle optional trajectory ----
            if return_trajectory:
                traj_full = predictions.reshape(n_steps, n_samples, batch_size, -1)
                all_final = traj_full[-1]  # (N, B, D)
            else:
                all_final = predictions.reshape(n_samples, batch_size, -1)
                traj_full = None
        else:
            # ---- Handle optional trajectory ----
            if return_trajectory:
                all_final = predictions[-1]  # (B, D)
                traj_full = predictions.unsqueeze(1)  # unsqueeze sample dim (N=1) (T, 1, B, D)
            else:
                all_final = predictions
                traj_full = None

        # ---- Average when generating multiple samples----
        if average_samples:
            X = all_final.mean(dim=0)  # (B, D)
            raw_samples = all_final  # (N, B, D)
        else:
            X = all_final  # (B, D)
            raw_samples = None

        return X, traj_full, raw_samples

    # ------------------------------------------------------------------
    # Inference with optional multiple noise samples
    # ------------------------------------------------------------------
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
    ) -> PredictionData:
        # ---- 0. Guard, when generating from noise we need n_samples ----
        if self._generate_from_noise and n_samples is None:
            raise ValueError("When generating from noise, you need to provide the number of samples with `n_samples`")

        # ----- 1. Prepare latent (noise) -----
        if latent is None:
            latent = prepare_latent_inference(
                step_data["source_state"],
                step_data["target_state"],
                self._noise_sampler,
                n_samples=n_samples,
                generate_from_noise=self._generate_from_noise,
            )

        # ----- 2. Build conditioning dict -----
        condition_reps_dict = get_tensor_dict_from_data(step_data["target_condition_data"])
        group_reps_dict = get_tensor_dict_from_data(step_data["target_group_data"])
        condition_dict = {**condition_reps_dict, **group_reps_dict}

        # ----- 3. Expand conditioning to match latent dimensions -----
        condition_dict, source_expanded = expand_conditioning(
            latent,
            condition_dict,
            step_data["source_state"],
        )

        # ----- 4. Configure ODE solver -----
        if solver_kwargs is None:
            solver_kwargs = {}
        solver_kwargs.setdefault("method", "euler")
        method = solver_kwargs.pop("method")

        if solver_cls is None:
            solver_cls = self._default_solver_cls

        time_grid = torch.linspace(0.0, 1.0, steps=n_steps, device=latent.device, dtype=latent.dtype)
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
