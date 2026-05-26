from typing import Any

import torch

from sc_flow.backends.torch._types import PredictionData
from sc_flow.backends.torch.coupling._coupling import independent_coupling
from sc_flow.backends.torch.methods._base import TorchGenerativeFlow
from sc_flow.backends.torch.methods._utils import StepData
from sc_flow.backends.torch.nn._vf import BaseVelocityField, MLPVelocity
from sc_flow.backends.torch.probability_paths._probability_paths import LinearDiracProbabilityPath
from sc_flow.backends.torch.solvers import BaseSolver, ODESolver

__all__ = ["CFM"]


class CFM(TorchGenerativeFlow):
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

    # ------------------------------------------------------------------
    # Expand conditioning tensors to match latent’s extra dimensions
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_conditioning(
        latent: torch.Tensor,
        condition_dict: dict[str, torch.Tensor],
        source: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        """
        Replicate condition_dict and source along the leading dimensions of latent

        Assumes original condition_dict values and source have shape (batch_size, ...)
        and latent has shape (n_samples, batch_size, dim) or (batch_size, dim).
        Returns expanded copies.
        """
        if latent.dim() <= 2:  # no extra sample dimension
            return condition_dict, source

        n_samples = latent.shape[0]
        expanded_cond = {}
        for k, v in condition_dict.items():
            # v shape: (batch_size, ...)
            expanded_cond[k] = v.repeat_interleave(n_samples, dim=0)
        expanded_source = None
        if source is not None:
            expanded_source = source.repeat_interleave(n_samples, dim=0)
        return expanded_cond, expanded_source

    # ------------------------------------------------------------------
    # Latent preparation (training vs inference)
    # ------------------------------------------------------------------
    def _prepare_latent_train(self, source: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
        """Called from _step_fn - always returns single noise per batch element."""
        if source is None or self._generate_from_noise:
            return self._noise_sampler(target.shape, device=self._device_id, dtype=self._dtype)
        return source

    def _prepare_latent_inference(
        self,
        source: torch.Tensor | None,
        target_reference: torch.Tensor,
        n_samples: int | None = None,
    ) -> torch.Tensor:
        """Called from _predict.

        - If source is given and we are NOT generating from noise, return source unchanged.
        - Otherwise sample noise with shape:
            if n_samples is None:  (batch_size, dim)
            else:                  (n_samples, batch_size, dim)
        """
        if source is None or self._generate_from_noise:
            shape = target_reference.shape
            if n_samples is not None:
                shape = (n_samples, *shape)
            return self._noise_sampler(shape, device=self._device_id, dtype=self._dtype)
        return source

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _step_fn(self, step_data: StepData, *args, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        target = step_data.target_state
        source = step_data.source_state

        # condition data as tensors
        condition_data = self._get_tensor_dict_from_data(step_data.target_condition_data)
        group_data = self._get_tensor_dict_from_data(step_data.target_group_data)
        cond = {**condition_data, **group_data}

        # latent (noise) – shape (batch_size, dim)
        latent = self._prepare_latent_train(source, target)
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

    # ------------------------------------------------------------------
    # Aggregate Predictions
    # ------------------------------------------------------------------
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
            Solver output. If return_trajectory=True, shape is (num_steps, total_batch, dim).
            Otherwise, shape is (total_batch, dim).
        latent_shape : torch.Size
            Original shape of the latent tensor before flattening (e.g., (n_samples, batch_size, dim)).
        return_trajectory : bool
            Whether the solver returned a full trajectory.

        Returns
        -------
        X : torch.Tensor
            Final predicted state, shape (batch_size, dim) after averaging (or original if no samples).
        traj : torch.Tensor or None
            Trajectory array, averaged over samples if applicable, shape (num_steps, batch_size, dim)
            or None if return_trajectory=False.
        raw_samples : torch.Tensor or None
            Raw final states for each sample, shape (n_samples, batch_size, dim) or None if single sample.
        """
        # ---- Determine if we have a sample dimension from latent state ----
        n_latent_dims = len(latent_shape)
        if n_latent_dims == 3:
            n_samples = latent_shape[0]
            batch_size = latent_shape[1]
            has_sample_dim = average_samples = True
        elif n_latent_dims == 2:
            batch_size = latent_shape[0]
            has_sample_dim = average_samples = False
        else:
            raise ValueError(f"Invalid shape for source latent state: {latent_shape}")

        # ---- Reshape output when needed ----
        if has_sample_dim:
            # predictions will be of shape (T, N, B, D)
            n_steps = predictions.shape[0]
            traj_full = predictions.reshape(n_steps, n_samples, batch_size, -1)
            all_final = traj_full[-1]  # (N, B, D)
        else:
            # No sample dimension – everything is flat batch
            all_final = predictions[-1]
            traj_full = predictions.unsqueeze(1)  # unsqueeze sample dim (N=1)

        # ---- Average when generating multiple samples----
        if average_samples:
            X = all_final.mean(dim=0)  # (B, D)
            raw_samples = all_final  # (N, B, D)
        else:
            X = all_final
            raw_samples = None

        # ---- Handle return of additional output ----
        if not return_trajectory:
            traj_full = None

        return X, traj_full, raw_samples

    # ------------------------------------------------------------------
    # Inference with optional multiple noise samples
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
        **kwargs,
    ) -> PredictionData:
        # ----- 1. Prepare latent (noise) -----
        if latent is None:
            latent = self._prepare_latent_inference(step_data.source_state, step_data.target_state, n_samples=n_samples)

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

        time_grid = torch.linspace(0.0, 1.0, steps=num_steps, device=latent.device, dtype=latent.dtype)
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
