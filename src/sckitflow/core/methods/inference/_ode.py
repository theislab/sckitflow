from typing import Any

import torch

from sckitflow.core._data_utils import (
    expand_conditioning,
    get_tensor_dict_from_data,
    prepare_latent_inference,
)
from sckitflow.core._types import (
    PredictionData,
    StepData,
    TNoiseSamplerFn,
    TTimeSamplerFn,
)
from sckitflow.core.methods._base import BaseFlowInferenceProtocol
from sckitflow.core.methods.inference._utils import aggregate_predictions
from sckitflow.core.nn._vf import BaseVelocityField
from sckitflow.core.probability_paths._probability_paths import BaseProbabilityPath
from sckitflow.core.solvers import ODESolver

__all__ = ["ODEInference"]


class ODEInference(BaseFlowInferenceProtocol):
    """Class for handling ODE integrations from an underlying module.

    This class can be used for standard ODE inference with CNFs.
    """

    def __init__(
        self,
        module: BaseVelocityField,
        probability_path: BaseProbabilityPath,
        time_sampler: TTimeSamplerFn,
        noise_sampler: TNoiseSamplerFn | None = None,
        generate_from_noise: bool = False,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
        n_steps: int = 100,
        latent: torch.Tensor | None = None,
        n_samples: int | None = None,
    ) -> None:
        """Initializes the inference class.

        The underlying module should inherit from `BaseVelocityField`; the
        module is expected to implement the `.get_vf_fn` method, to compile the
        velocity field function with the signature `vf_fn(t, xt)` -- this
        requirement is needed for compatibility with `torchdiffeq`.

        Shares the same arguments as `BaseFlowInferenceProtocol`, with some
        additional attributes

        :param solver_kwargs: The keyword arguments used to instantiate ODE solvers.
        :param return_trajectory: Boolean flag indicating wether to return the whole
            trajectory of the simulation, or only the endpoint.
        :param n_steps: The number of discretization steps used to simulate the dynamics.
        :param latent: The optional latent state used to initialize the dynamics from.
        :param n_samples: The number of samples used to run the simulation; it will only be
            used when `generate_from_noise` is `True`.
        """
        # ---- 0. Initialize parent class ----
        super().__init__(
            module,
            probability_path,
            time_sampler,
            noise_sampler=noise_sampler,
            generate_from_noise=generate_from_noise,
            dtype=dtype,
            device_id=device_id,
        )

        # ---- 1. Assign extra attributes ----
        self._solver_kwargs = solver_kwargs
        self._return_trajectory = return_trajectory
        self._n_steps = n_steps
        self._latent = latent
        self._n_samples = n_samples

    def predict(self, step_data: StepData) -> PredictionData:
        # ---- 0. Guard, when generating from noise we need n_samples ----
        if self.generate_from_noise and self.n_samples is None:
            raise ValueError("When generating from noise, you need to provide the number of samples with `n_samples`")
        if self.generate_from_noise and self.noise_sampler is None:
            raise TypeError("When generating from noise, you need to provide a noise_sampler.")

        # ----- 1. Prepare latent (noise) -----
        if self.latent is None:
            latent = prepare_latent_inference(
                step_data["source_state"],
                step_data["target_state"],
                self.noise_sampler,
                n_samples=self.n_samples,
                generate_from_noise=self.generate_from_noise,
            ).to(device=self.device_id, dtype=self.dtype)
        else:
            latent = self.latent.to(device=self.device_id, dtype=self.dtype)

        # ---- 2. Get optional source ----
        source_state_raw = step_data["source_state"]
        source_state = (
            source_state_raw.to(device=self.device_id, dtype=self.dtype) if source_state_raw is not None else None
        )

        # ----- 3. Build conditioning dict -----
        condition_data = get_tensor_dict_from_data(step_data["target_condition_data"])
        group_data = get_tensor_dict_from_data(step_data["target_group_data"])
        condition_dict = {
            **{k: v.to(device=self.device_id, dtype=self.dtype) for k, v in condition_data.items()},
            **{k: v.to(device=self.device_id, dtype=self.dtype) for k, v in group_data.items()},
        }

        # ----- 4. Expand conditioning to match latent dimensions -----
        condition_dict, source_expanded = expand_conditioning(
            latent,
            condition_dict,
            source_state,
        )

        # ----- 5. Configure ODE solver -----
        solver_kwargs = dict(self.solver_kwargs or {})
        solver_kwargs.setdefault("method", "euler")
        method = solver_kwargs.pop("method")

        time_grid = torch.linspace(0.0, 1.0, steps=self.n_steps, device=latent.device, dtype=latent.dtype)
        solver = ODESolver(
            self.module,
            method=method,
            vf_kwargs={"condition_dict": condition_dict, "source": source_expanded},
            device_id=self.device_id,
        )

        # ----- 6. Integrate -----
        predictions = solver.solve(
            latent,
            time_grid,
            solver_kwargs=solver_kwargs,
            return_trajectory=self.return_trajectory,
        )

        # ----- 7. Aggregate predictions (averaging, reshaping) -----
        X, traj, raw_samples = aggregate_predictions(
            predictions=predictions,
            latent_shape=latent.shape,
            return_trajectory=self.return_trajectory,
        )
        return PredictionData(X=X, traj=traj, raw_samples=raw_samples)

    @property
    def solver_kwargs(self) -> dict[str, Any] | None:
        return self._solver_kwargs

    @property
    def return_trajectory(self) -> bool:
        return self._return_trajectory

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def latent(self) -> torch.Tensor | None:
        return self._latent

    @property
    def n_samples(self) -> int | None:
        return self._n_samples
