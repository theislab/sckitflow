"""The FlowMatching model wrapper.

Torch-native (OT) conditional flow matching. The velocity field, probability path, and loss are all
torch (trained by Lightning); the **only** JAX is the per-minibatch OT coupling
(:func:`~sc_flow.backends.jax.coupling.ot_linear_coupling`), a forward-only resample of the
``(source, target)`` pairing — no autograd crosses into JAX, so there is no DLPack bridge. Prediction
integrates the torch velocity field with ``torchdiffeq``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sc_flow.data import FlowSpec

if TYPE_CHECKING:
    from sc_flow.data._compile_obs import CompiledDims, DataInput

__all__ = ["FlowMatching"]

logger = logging.getLogger(__name__)


class FlowMatching:
    """Torch-native (OT) conditional flow-matching model over a binded data spec."""

    def __init__(
        self,
        spec: FlowSpec,
        *,
        objective: str = "otfm",
        hidden_dims: Sequence[int] = (1024, 1024, 1024),
        decoder_dims: Sequence[int] | None = None,
        time_encoder_dims: Sequence[int] | None = None,
        condition_embedding_dim: int = 64,
        condition_mode: str = "deterministic",
        regularization: float = 1.0,
        sigma: float = 0.0,
        pooling: str = "mean",
        match_method: str = "sinkhorn",
        match_kwargs: Mapping[str, Any] | None = None,
        seed: int = 0,
    ):
        self.spec = spec
        self.objective_name = objective
        self.hidden_dims = tuple(hidden_dims)
        self.decoder_dims = None if decoder_dims is None else tuple(decoder_dims)
        self.time_encoder_dims = None if time_encoder_dims is None else tuple(time_encoder_dims)
        self.condition_embedding_dim = condition_embedding_dim
        self.condition_mode = condition_mode
        self.regularization = regularization
        self.sigma = sigma
        self.pooling = pooling
        self.match_method = match_method
        self.match_kwargs = dict(match_kwargs) if match_kwargs else {}
        self.seed = seed

        self.vf = None  # MLPVelocity (torch nn.Module holding the weights)
        self.model = None  # alias of vf (the trained weights)
        self.probability_path = None
        self.objective = None
        self._condition_fn = None
        self._dims = None

    # --- construction helpers -------------------------------------------------------------------

    def _build_vf(self, dims: CompiledDims) -> torch.nn.Module:
        """Size an ``MLPVelocity`` from :class:`~sc_flow.data.CompiledDims` (no batch pulled)."""
        from sc_flow.backends.torch.nn._vf import MLPVelocity

        cond_input_layers = (
            {
                realm: {"input_dim": int(dim), "output_dim": int(self.condition_embedding_dim)}
                for realm, dim in dims.condition.items()
            }
            if dims.condition
            else None
        )
        vf_kwargs: dict[str, Any] = {
            "state_dim": int(dims.state),
            "conditioning_id": "concat",
            "state_encoder_mlp_kwargs": {"hidden_dims": self.hidden_dims},
        }
        if self.decoder_dims is not None:
            vf_kwargs["vf_decoder_mlp_kwargs"] = {"hidden_dims": self.decoder_dims}
        if self.time_encoder_dims is not None:
            vf_kwargs["time_encoder_mlp_kwargs"] = {"hidden_dims": self.time_encoder_dims}
        if cond_input_layers is not None:
            vf_kwargs.update(
                condition_encoder_input_layers=cond_input_layers,
                condition_encoder_output_dim=int(self.condition_embedding_dim),
                condition_encoder_pooling_mode=self.pooling,
                condition_mode=self.condition_mode,
            )
        if self.objective_name == "genot":
            # GENOT flows noise->target (state space) with the SOURCE cell conditioning the field: enable
            # the source encoder. G1 is same-space — the source cell is the state source, so it is sized by
            # the state dim; the coupling reps (if any) only drive the OT plan. (Cross-space source is G2.)
            vf_kwargs["source_encoder_mlp_kwargs"] = {"input_dim": int(dims.state), "hidden_dims": self.hidden_dims}
        return MLPVelocity(**vf_kwargs)

    def _build_probability_path(self):
        from sc_flow.backends.torch.probability_paths._probability_paths import (
            LinearDiracProbabilityPath,
            LinearGaussianProbabilityPath,
        )

        if self.sigma > 0:
            return LinearGaussianProbabilityPath(sigma=self.sigma, prng=torch.Generator().manual_seed(int(self.seed)))
        return LinearDiracProbabilityPath(sigma=0.0)

    # --- API ------------------------------------------------------------------------------------

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
        """Compile ``data``, build the torch VF + OT-FM objective, and run the Lightning trainer."""
        import lightning.pytorch as pl
        from binded import Loader, SamplerConfig

        # Seed every stochastic source from self.seed for a bit-reproducible run: VF init (torch global,
        # reset here), the binded data order (Scheme seed), the OT plan-sampling + t draw (objective), and
        # any probability-path noise (its prng).
        # TODO(reprod): this only guarantees bit-reproducibility on CPU. A CUDA run still has torch's
        # nondeterministic kernels (atomics in some backward ops) — gate torch.use_deterministic_algorithms
        # (+ CUBLAS_WORKSPACE_CONFIG) behind a `deterministic=True` fit flag when GPU repro is needed.
        torch.manual_seed(int(self.seed))

        # 1. Compile to labels + dims (no cells / no sampler); build a numpy-yielding loader.
        compiled = self.spec.compile(
            data, rep_tables=rep_tables, min_runs_per_leaf=min_runs_per_leaf, seed=self.seed
        )
        self._condition_fn = compiled.condition_fn
        self._dims = compiled.dims
        cfg = SamplerConfig(batch_size=batch_size, chunk_size=1, preload_nchunks=max(1, batch_size), to=None)
        loader = Loader(compiled.scheme, cfg, compiled.condition_fn)

        # 2. Torch velocity field + probability path, sized from compiled.dims.
        self.vf = self._build_vf(compiled.dims)
        self.model = self.vf
        self.probability_path = self._build_probability_path()

        # 3. Objective selected by name (OT coupling in JAX, everything else torch) + shared harness.
        from sc_flow.backends.torch.training._harness import SCFlowLightningModule
        from sc_flow.backends.torch.training._objective import build_objective

        self.objective = build_objective(
            self.objective_name,
            self.probability_path,
            condition_mode=self.condition_mode,
            regularization=self.regularization,
            coupling_locs=compiled.coupling,
            match_method=self.match_method,
            match_kwargs=self.match_kwargs,
            seed=self.seed,
        )
        harness = SCFlowLightningModule(self.vf, self.objective, lr=lr)

        # 4. Wrap the binded loader as an IterableDataset (batches pass through untouched).
        class _BindedIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, loader):
                self._loader = loader

            def __iter__(self):
                yield from self._loader

        torch_loader = torch.utils.data.DataLoader(_BindedIterableDataset(loader), batch_size=None)

        trainer = pl.Trainer(
            max_steps=n_train_steps,
            accelerator=device,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
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
        seed: int | None = None,
    ) -> np.ndarray:
        """Translate ``x`` under ``condition`` by integrating the torch velocity field with torchdiffeq.

        For ``objective="otfm"`` the ODE integrates the cells ``x`` themselves (source → target). For
        ``objective="genot"`` it integrates **from latent noise** (target space) with ``x`` held fixed as
        the source-conditioning input (noise → target | source) — a *generative* translation, so it is
        stochastic; ``seed`` (default :attr:`self.seed`) makes the noise draw reproducible.
        """
        if self.vf is None:
            raise RuntimeError("Model must be fitted before predict() can be called.")
        from torchdiffeq import odeint

        self.vf.to(device)
        self.vf.eval()

        # Resolve a leaf tuple to its condition dict; a ready condition dict is used as-is.
        if not (isinstance(condition, dict) and all(isinstance(v, np.ndarray) for v in condition.values())):
            if self._condition_fn is None:
                raise RuntimeError("Model must be fitted to resolve leaf conditions.")
            condition = self._condition_fn(condition)
        cond_t = {k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=device) for k, v in condition.items()}

        x_arr = np.asarray(x, dtype=np.float32)
        is_genot = self.objective_name == "genot"
        if is_genot:
            # y0 = latent ~ N(0, I) in the target/generated space (dims.state); x conditions the field.
            gen = torch.Generator().manual_seed(int(self.seed if seed is None else seed))
            y0 = torch.randn(x_arr.shape[0], int(self._dims.state), generator=gen).to(device=device)
            source_cells = torch.as_tensor(x_arr, device=device)
        else:
            y0 = torch.as_tensor(x_arr, device=device)
            source_cells = None
        t_grid = torch.linspace(0.0, 1.0, num_steps, device=device)

        def f(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            t_exp = t.reshape(1, 1).expand(y.shape[0], 1)
            return self.vf(t_exp, y, cond_t, source_cells)

        with torch.no_grad():
            trajectory = odeint(f, y0, t_grid, method="euler")

        if return_trajectory:
            return trajectory.cpu().numpy()
        return trajectory[-1].cpu().numpy()

    # --- persistence ------------------------------------------------------------------------------

    _CTOR_FIELDS: tuple[str, ...] = (
        "hidden_dims",
        "decoder_dims",
        "time_encoder_dims",
        "condition_embedding_dim",
        "condition_mode",
        "regularization",
        "sigma",
        "pooling",
        "match_method",
        "match_kwargs",
        "seed",
    )

    def save(self, path: str | Path) -> None:
        """Persist the fitted model so :meth:`predict` works again after :meth:`load`.

        Writes ``path`` as a directory: ``weights.pt`` (the torch ``state_dict``, portable across
        devices) and ``state.pkl`` (cloudpickle — the constructor config, :attr:`spec`, the compiled
        :class:`~sc_flow.data.CompiledDims`, and the fitted ``condition_fn`` closure — the same
        cloudpickle-of-a-closure pattern already used by :mod:`sc_flow.external`). Does **not** persist
        optimizer/trainer state — this is for inference after reload, not resuming ``fit()``.
        """
        import cloudpickle

        if self.vf is None:
            raise RuntimeError("Model must be fitted before save().")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.vf.state_dict(), path / "weights.pt")
        state = {
            "objective": self.objective_name,
            "ctor_kwargs": {name: getattr(self, name) for name in self._CTOR_FIELDS},
            "spec": self.spec,
            "dims": self._dims,
            "condition_fn": self._condition_fn,
        }
        with open(path / "state.pkl", "wb") as f:
            cloudpickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> FlowMatching:
        """Reconstruct a fitted model from :meth:`save`.

        ``predict()`` works immediately; ``fit()`` would start a fresh run (no optimizer/trainer state
        is restored).
        """
        import cloudpickle

        path = Path(path)
        with open(path / "state.pkl", "rb") as f:
            state = cloudpickle.load(f)

        model = cls(spec=state["spec"], objective=state["objective"], **state["ctor_kwargs"])
        model._dims = state["dims"]
        model._condition_fn = state["condition_fn"]
        model.vf = model._build_vf(model._dims)
        model.vf.load_state_dict(torch.load(path / "weights.pt", map_location="cpu"))
        model.model = model.vf
        model.probability_path = model._build_probability_path()
        return model
