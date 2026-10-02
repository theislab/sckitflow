from __future__ import annotations

from typing import Any

import torch
from pydantic import Field, PositiveInt, model_validator
from scfit.registry import component

from sckitflow.core._data_utils import (
    expand_conditioning,
    get_tensor_dict_from_data,
    prepare_latent_inference,
)
from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods._base import AbstractFlowMethod, AbstractFlowMethodConfig, InferenceMethodConfig
from sckitflow.core.methods.inference._utils import aggregate_predictions
from sckitflow.core.solvers import ODESolver

__all__ = ["ODEInference", "ODEInferenceConfig"]


class ODEInference(AbstractFlowMethod):
    """ODE inference from an underlying velocity-field module.

    The module is expected to implement ``.get_vf_fn`` to compile a
    ``vf_fn(t, xt)`` function compatible with ``torchdiffeq``.

    Constructed from the module and the flow configuration:

    .. code-block:: python

        inference = ODEInference(module, ODEInferenceConfig(probability_path=..., n_steps=50))
    """

    def __init__(
        self, module: torch.nn.Module, config: ODEInferenceConfig | None = None, *, latent: torch.Tensor | None = None
    ) -> None:
        """Initializes the ODE inference method.

        :param module: An initialized neural module the method builds upon.
        :param config: The flow and solver settings; ``None`` takes every default.
        :param latent: Optional initial latent state; when provided, sampling
            from the noise distribution is skipped. It is moved to the batch's
            device and dtype. Live, so not part of the config.
        """
        self.config = ODEInferenceConfig() if config is None else config
        super().__init__(module, self.config)
        self._latent = latent

    def predict(self, step_data: StepData, *, generator: torch.Generator) -> PredictionData:
        """Integrates the ODE and returns the aggregated prediction.

        The dynamics are read from ``self.probability_path``, ``self.noise_sampler``, ``self.module``
        and ``self.config``. Device and dtype come from the batch itself.
        """
        # ----- 1. Prepare latent (noise) -----
        target_state = step_data["target_state"]
        if self.latent is None:
            latent = prepare_latent_inference(
                step_data["source_state"],
                step_data["target_state"],
                self.noise_sampler,
                n_samples=self.config.n_samples,
                generate_from_noise=self.generate_from_noise,
                generator=generator,
            )
        else:
            # the only tensor not built by the loader, so the only one to place
            latent = self.latent.to(device=target_state.device, dtype=target_state.dtype)

        # ---- 2. Get optional source ----
        source_state = step_data["source_state"]

        # ----- 3. Build conditioning dict -----
        condition_dict = {
            **get_tensor_dict_from_data(step_data["target_condition_data"]),
            **get_tensor_dict_from_data(step_data["target_group_data"]),
        }

        # ----- 4. Expand conditioning to match latent dimensions -----
        condition_dict, source_expanded = expand_conditioning(
            latent,
            condition_dict,
            source_state,
        )

        # ----- 5. Configure ODE solver -----
        solver_kwargs = dict(self.config.solver_kwargs)
        solver_kwargs.setdefault("method", "euler")
        method = solver_kwargs.pop("method")

        time_grid = torch.linspace(0.0, 1.0, steps=self.config.n_steps, device=latent.device, dtype=latent.dtype)
        solver = ODESolver(
            self.module,
            method=method,
            vf_kwargs={"condition_dict": condition_dict, "source": source_expanded},
            device_id=str(latent.device),
        )

        # ----- 6. Integrate -----
        predictions = solver.solve(
            latent,
            time_grid,
            solver_kwargs=solver_kwargs,
            return_trajectory=self.config.return_trajectory,
        )

        # ----- 7. Aggregate predictions (averaging, reshaping) -----
        X, traj, raw_samples = aggregate_predictions(
            predictions=predictions,
            latent_shape=latent.shape,
            return_trajectory=self.config.return_trajectory,
        )
        return PredictionData(X=X, traj=traj, raw_samples=raw_samples)

    @property
    def latent(self) -> torch.Tensor | None:
        return self._latent


@component("inference_method.ode", builds=ODEInference)
class ODEInferenceConfig(AbstractFlowMethodConfig, InferenceMethodConfig):
    """ODE inference over a trained velocity field: the flow settings plus the solver's."""

    solver_kwargs: dict[str, Any] = {}
    """Forwarded to the ODE solver. ``method`` defaults to ``"euler"``."""
    return_trajectory: bool = False
    """Return the whole trajectory instead of only the endpoint."""
    n_steps: int = Field(default=100, ge=2)
    """Points on the solver's time grid, both ends included."""
    n_samples: PositiveInt | None = None
    """Samples per batch element. Required when ``generate_from_noise`` is ``True``."""

    @model_validator(mode="after")
    def _noise_needs_samples(self) -> ODEInferenceConfig:
        if self.generate_from_noise and self.n_samples is None:
            raise ValueError("generating from noise needs `n_samples`.")
        return self

    def build(self, module: torch.nn.Module) -> ODEInference:
        return ODEInference(module, self)
