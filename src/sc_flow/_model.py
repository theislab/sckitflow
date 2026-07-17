"""The FlowMatching model wrapper.

Replaces the legacy SCFlow model. Wraps CellFlow's JAX conditional velocity field
and probability path numerics inside a PyTorch training and prediction loop.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import lightning.pytorch as pl
import numpy as np
import torch

from sc_flow.backends.torch.jaxbridge import (
    CellFlowFMObjective,
    JaxParamModule,
    iter_fm_batches,
    jax_to_torch,
    torch_to_jax,
)
from sc_flow.data import FlowSpec

if TYPE_CHECKING:
    from sc_flow.data._compile_obs import DataInput

__all__ = ["FlowMatching"]

logger = logging.getLogger(__name__)


class FlowMatching:
    """The converged Flow Matching model wrapping CellFlow's JAX numerics inside PyTorch training/inference."""

    def __init__(
        self,
        spec: FlowSpec,
        objective: str = "cellflow",
        architecture: str = "mlp-velocity",
        hidden_dims: tuple[int, ...] = (1024, 1024, 1024),
        condition_embedding_dim: int = 64,
        condition_mode: str = "deterministic",
        regularization: float = 1.0,
        sigma: float = 0.0,
    ):
        self.spec = spec
        self.objective_name = objective
        self.architecture_name = architecture
        self.hidden_dims = hidden_dims
        self.condition_embedding_dim = condition_embedding_dim
        self.condition_mode = condition_mode
        self.regularization = regularization
        self.sigma = sigma

        self.vf = None
        self.probability_path = None
        self.model = None  # JaxParamModule holding PyTorch parameters
        self.objective = None  # CellFlowFMObjective wrapping JAX value_and_grad
        self._condition_fn = None

    def fit(
        self,
        data: DataInput,
        *,
        rep_tables: Mapping[str, Mapping] | None = None,
        batch_size: int = 128,
        n_train_steps: int = 100_000,
        valid_freq: int = 1_000,
        device: str = "cpu",
        lr: float = 1e-4,
        min_runs_per_leaf: int = 0,
    ) -> FlowMatching:
        """Fits the model by compiling data into a Loader and running PL Trainer."""
        # 1. Compile spec & build loader
        loader = self.spec.build_loader(
            data,
            rep_tables=rep_tables,
            batch_size=batch_size,
            min_runs_per_leaf=min_runs_per_leaf,
            to=None,  # yield numpy arrays
        )
        self._condition_fn = loader._cond_fn

        # 2. Infer dimensions
        leaf = next(iter(loader.s.nodes["pert"].weights))
        sample_cond = loader._cond_fn(leaf)
        batch = next(iter(loader))
        D = batch["target"].shape[-1]

        from cellflow._compat import ConstantNoiseFlow
        from cellflow.networks._velocity_field import ConditionalVelocityField

        # 3. Instantiate JAX ConditionalVelocityField & ConstantNoiseFlow
        self.vf = ConditionalVelocityField(
            output_dim=D,
            max_combination_length=sample_cond[next(iter(sample_cond))].shape[1],
            condition_mode=self.condition_mode,
            regularization=self.regularization,
            condition_embedding_dim=self.condition_embedding_dim,
            hidden_dims=self.hidden_dims,
        )
        self.probability_path = ConstantNoiseFlow(sigma=self.sigma)

        # Initialize JAX parameters
        import jax
        import jax.numpy as jnp

        k_init, k_enc = jax.random.split(jax.random.PRNGKey(0))
        cond_init = {k: jnp.ones(v.shape) for k, v in sample_cond.items()}
        params = self.vf.init(
            {"params": k_init, "condition_encoder": k_enc},
            t=jnp.ones((1, 1)),
            x_t=jnp.ones((1, D)),
            cond=cond_init,
            encoder_noise=jnp.ones((1, self.condition_embedding_dim)),
            train=False,
        )["params"]

        # 4. Instantiate JaxParamModule and CellFlowFMObjective
        self.model = JaxParamModule(params)
        self.objective = CellFlowFMObjective(self.vf, self.probability_path, seed=0)

        # 5. Setup data harness
        class _AdaptedIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, loader, cond_emb_dim):
                self.loader = loader
                self.cond_emb_dim = cond_emb_dim

            def __iter__(self):
                yield from iter_fm_batches(self.loader, condition_embedding_dim=self.cond_emb_dim)

        dataset = _AdaptedIterableDataset(loader, self.condition_embedding_dim)
        torch_loader = torch.utils.data.DataLoader(dataset, batch_size=None)

        from sc_flow.backends.torch.training._harness import SCFlowLightningModule

        harness = SCFlowLightningModule(self.model, self.objective, lr=lr)

        trainer = pl.Trainer(
            max_steps=n_train_steps,
            accelerator=device,
            val_check_interval=valid_freq,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(harness, torch_loader)
        return self

    def predict(
        self,
        x: np.ndarray,
        condition: dict[str, np.ndarray] | tuple[Any, ...],
        *,
        device: str = "cpu",
        num_steps: int = 50,
        return_trajectory: bool = False,
    ) -> np.ndarray:
        """Predicts translation using PyTorch ODE solver (torchdiffeq) driven by JAX velocity field."""
        if self.model is None or self.vf is None:
            raise RuntimeError("Model must be fitted before predict() can be called.")

        self.model.to(device)

        if not isinstance(condition, dict) or not all(isinstance(v, np.ndarray) for v in condition.values()):
            if self._condition_fn is None:
                raise RuntimeError("Model must be fitted to resolve leaf conditions.")
            condition = self._condition_fn(condition)

        import jax
        import jax.numpy as jnp

        leaves = [torch_to_jax(p) for p in self.model.param_tensors]
        params_jax = jax.tree_util.tree_unflatten(self.model.treedef, leaves)
        cond_jax = {k: jax.device_put(v) for k, v in condition.items()}
        encoder_noise = jnp.zeros((1, self.condition_embedding_dim))

        # 1. Encode condition once
        cond_embedding, _, _ = self.vf.apply(
            {"params": params_jax}, cond_jax, encoder_noise, train=False, method=self.vf.encode_condition
        )

        # 2. ODE closure for torchdiffeq
        def f(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            t_expanded = t.expand(y.shape[0], 1)
            t_jax = torch_to_jax(t_expanded)
            y_jax = torch_to_jax(y)
            v_jax = self.vf.apply(
                {"params": params_jax},
                t_jax,
                y_jax,
                cond_embedding,
                train=False,
                method=self.vf.velocity_from_embedding,
            )
            return jax_to_torch(v_jax).to(device)

        # 3. Integrate using torchdiffeq
        from torchdiffeq import odeint

        y0 = torch.as_tensor(x, dtype=torch.float32, device=device)
        t_grid = torch.linspace(0.0, 1.0, num_steps, device=device)

        with torch.no_grad():
            trajectory = odeint(f, y0, t_grid, method="euler")

        if return_trajectory:
            return trajectory.cpu().numpy()
        else:
            return trajectory[-1].cpu().numpy()
