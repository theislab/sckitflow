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

from sc_flow.core.data import FlowSpec

if TYPE_CHECKING:
    from sc_flow.core.data._compile_obs import CompiledDims, DataInput

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
        # {metric_name: [mean-over-held-out-conditions per validation pass]}; populated by fit(split_by=...).
        self.metrics_history: dict[str, list[float]] = {}

    # --- construction helpers -------------------------------------------------------------------

    def _build_vf(self, dims: CompiledDims) -> torch.nn.Module:
        """Size an ``MLPVelocity`` from :class:`~sc_flow.core.data.CompiledDims` (no batch pulled)."""
        from sc_flow.flow._set_encoder import SetEncoder
        from sc_flow.flow._vf import MLPVelocity

        vf_kwargs: dict[str, Any] = {
            "state_dim": int(dims.state),
            "combiner": "concat",
            "state_encoder_mlp_kwargs": {"hidden_dims": self.hidden_dims},
        }
        if self.decoder_dims is not None:
            vf_kwargs["vf_decoder_mlp_kwargs"] = {"hidden_dims": self.decoder_dims}
        if self.time_encoder_dims is not None:
            vf_kwargs["time_encoder_mlp_kwargs"] = {"hidden_dims": self.time_encoder_dims}
        if dims.condition:
            vf_kwargs["condition_encoder"] = SetEncoder(
                input_layers={
                    realm: {"input_dim": int(dim), "output_dim": int(self.condition_embedding_dim)}
                    for realm, dim in dims.condition.items()
                },
                output_dim=int(self.condition_embedding_dim),
                pooling_mode=self.pooling,
                condition_mode=self.condition_mode,
            )
        if self.objective_name == "genot":
            # GENOT flows noise->target (state space) with the SOURCE cell conditioning the field: enable
            # the source encoder. G1 is same-space — the source cell is the state source, so it is sized by
            # the state dim; the coupling reps (if any) only drive the OT plan. (Cross-space source is G2.)
            vf_kwargs["source_encoder_mlp_kwargs"] = {"input_dim": int(dims.state), "hidden_dims": self.hidden_dims}
        return MLPVelocity(**vf_kwargs)

    def _build_probability_path(self):
        from sc_flow.flow.probability_paths._probability_paths import (
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
        chunk_size: int = 1,
        preload_nchunks: int | None = None,
        preload_to_gpu: bool | None = None,
        n_train_steps: int = 100_000,
        valid_freq: int = 1_000,
        device: str = "cpu",
        lr: float = 1e-4,
        min_runs_per_leaf: int = 0,
        control_in_memory: bool = False,
        split_by: str | Sequence[str] | None = None,
        split_ratios: Mapping[str, float] | Sequence[float] | None = None,
        val_batch_size: int | None = None,
        n_val_conditions: int | None = None,
        metrics: Sequence[str] = ("r_squared", "e-dist"),
        val_num_steps: int = 50,
        val_max_source_cells: int | None = 2048,
        debug_val: bool = False,
        callbacks: Sequence[Any] | None = None,
    ) -> FlowMatching:
        """Compile ``data``, build the torch VF + OT-FM objective, and run the Lightning trainer.

        With ``split_by`` set, whole conditions (target combinations sharing the ``split_by`` values) are
        held out into a validation split (deterministic in :attr:`seed`); every ``valid_freq`` steps the
        held-out controls are translated under each held-out condition and the ``metrics`` (distribution
        metrics — ``r_squared``, ``e-dist``) are scored, logged as ``val_<metric>_mean`` and appended to
        :attr:`metrics_history`. ``split_by=None`` trains on all conditions with no validation.

        :param chunk_size: Contiguous cells the binded ``Loader`` reads per chunk. ``1`` (default) works
            on any layout but issues ``batch_size`` scattered single-row zarr reads per batch — on the
            Tahoe plates that is the dominant training cost (~2s/batch on Lustre vs a ~7ms GPU step).
            Set to e.g. ``32`` on data grouped into contiguous per-condition runs (Tahoe's grouped plates)
            to read sequentially — cf-train measured ~80x fewer read ops. Must divide ``batch_size`` and
            requires every sampled condition's contiguous run to be ≥ ``chunk_size`` cells.
        :param preload_nchunks: How many chunks the loader prefetches (buffer size). ``None`` picks a
            batch-sized buffer for ``chunk_size=1`` and a large prefetch (``~32`` batches of chunks) when
            ``chunk_size>1`` — the buffer refills synchronously, so a too-small buffer stalls training
            ~400ms on every drain (measured on Tahoe/Lustre).
        :param preload_to_gpu: Keep the loader's read window GPU-resident (needs ``cupy``), so batches
            arrive as GPU tensors with no per-step host→device copy. ``None`` = auto (GPU training uses it
            when cupy is available; CPU training never does). Batches are always torch tensors (``to="torch"``).
        :param split_by: Condition column(s) whose unique combinations are partitioned into train/val
            (a subset of the compiled root columns). ``None`` = no held-out split / no validation.
        :param split_ratios: Train/val fractions — a ``{"train": .., "val": ..}`` mapping or a
            ``(train, val)`` sequence summing to 1.0. Defaults to ``(0.8, 0.2)``.
        :param val_batch_size: Target cells sampled per held-out condition (controls are read in full).
            Defaults to ``batch_size``.
        :param n_val_conditions: How many condition batches to score per validation pass (control
            populations are cycled, each drawing a held-out condition; seeded). Defaults to the number of
            held-out condition combinations.
        :param metrics: Names (in :data:`~sc_flow.core.metrics.METRICS_REGISTRY`) of the
            distribution metrics to score on the held-out split.
        :param val_num_steps: ODE integration steps for the validation translation.
        :param val_max_source_cells: Cap on the control/source population size fed to prediction + metrics
            per validation batch (random subsample; ``None`` disables the cap). binded's ``EvalLoader``
            reads each held-out control population **in full** regardless of ``val_batch_size`` — with
            ``match_context`` pooling controls across many plates/stores that population can reach tens of
            thousands of cells, and both the ODE trajectory and the O(n^2) pairwise-distance metrics
            (e.g. ``EnergyDistance``) scale with it, reliably OOMing at real multi-plate scale.
        :param control_in_memory: Materialize the control (source) population in RAM. In-memory nodes are
            exempt from the ``chunk_size`` run-length rule, so this lets ``chunk_size>1`` work even when the
            controls are fragmented across stores (the target node is still governed by ``min_runs_per_leaf``).
        :param callbacks: Extra Lightning ``Callback``\\s appended to the trainer (e.g. loss logging,
            throughput timing, checkpointing). ``None`` adds none.
        """
        from binded import Loader, SamplerConfig

        from sc_flow._optional import require

        pl = require("lightning.pytorch")

        # Seed every stochastic source from self.seed for a bit-reproducible run: VF init (torch global,
        # reset here), the binded data order (Scheme seed), the OT plan-sampling + t draw (objective), and
        # any probability-path noise (its prng).
        # TODO(reprod): this only guarantees bit-reproducibility on CPU. A CUDA run still has torch's
        # nondeterministic kernels (atomics in some backward ops) — gate torch.use_deterministic_algorithms
        # (+ CUBLAS_WORKSPACE_CONFIG) behind a `deterministic=True` fit flag when GPU repro is needed.
        torch.manual_seed(int(self.seed))

        if chunk_size > 1 and batch_size % chunk_size != 0:
            raise ValueError(f"chunk_size ({chunk_size}) must divide batch_size ({batch_size}).")

        # 1. Compile to labels + dims (no cells / no sampler); optionally hold out whole conditions.
        compiled = self.spec.compile(
            data, rep_tables=rep_tables, min_runs_per_leaf=min_runs_per_leaf,
            control_in_memory=control_in_memory, seed=self.seed,
        )
        self._condition_fn = compiled.condition_fn
        self._dims = compiled.dims
        self.metrics_history = {}

        train_scheme, val_scheme = compiled.scheme, None
        if split_by is not None:
            from binded import split_scheme

            split_by_cols = [split_by] if isinstance(split_by, str) else list(split_by)
            ratios = self._resolve_split_ratios(split_ratios)
            splits = split_scheme(
                compiled.scheme, split_by=split_by_cols, ratios=ratios, random_state=int(self.seed)
            )
            train_scheme, val_scheme = splits["train"], splits["val"]

        preload = self._resolve_preload(batch_size, chunk_size, preload_nchunks)
        # Transport: yield torch tensors and (on GPU) keep the read window GPU-resident so there is no
        # per-step host->device copy and refills happen on-device. `to=None` (annbatch default) is host
        # numpy — a redundant numpy->torch->host->device chain, and it errors outright once cupy is present.
        to_gpu = self._resolve_preload_to_gpu(device, preload_to_gpu)
        cfg = SamplerConfig(batch_size=batch_size, chunk_size=chunk_size, preload_nchunks=preload,
                            to="torch", preload_to_gpu=to_gpu)
        loader = Loader(train_scheme, cfg, compiled.condition_fn)

        # 2. Torch velocity field + probability path, sized from compiled.dims.
        self.vf = self._build_vf(compiled.dims)
        self.model = self.vf
        self.probability_path = self._build_probability_path()

        # 3. Objective selected by name (OT coupling in JAX, everything else torch) + shared harness.
        # Import from sc_flow.flow so the concrete objectives / predictor register before we build by name.
        from sc_flow.core.training import TrainingModule
        from sc_flow.flow import build_objective, build_predictor

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

        val_metrics, val_loader = None, None
        if val_scheme is not None:
            val_metrics, val_loader = self._build_validation(
                val_scheme,
                compiled.condition_fn,
                val_batch_size=val_batch_size or batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                preload_to_gpu=to_gpu,
                n_val_conditions=n_val_conditions,
                metrics=metrics,
                val_num_steps=val_num_steps,
            )

        predictor = build_predictor(
            "ode", is_genot=self.objective_name == "genot", state_dim=int(self._dims.state),
            num_steps=val_num_steps, seed=int(self.seed),
        )
        harness = TrainingModule(
            self.vf, self.objective, lr=lr, val_metrics=val_metrics, predictor=predictor,
            val_max_source_cells=val_max_source_cells, debug_val=debug_val,
        )

        # 4. Wrap the binded loader as an IterableDataset (batches pass through untouched).
        class _BindedIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, loader):
                self._loader = loader

            def __iter__(self):
                yield from self._loader

        torch_loader = torch.utils.data.DataLoader(_BindedIterableDataset(loader), batch_size=None)

        trainer_kwargs: dict[str, Any] = {
            "max_steps": n_train_steps,
            "accelerator": device,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }
        if callbacks:
            trainer_kwargs["callbacks"] = list(callbacks)
        if val_loader is not None:
            # Step-based validation: run the held-out pass every valid_freq training steps (no sanity pass,
            # so metrics_history holds only real validation runs).
            trainer_kwargs["val_check_interval"] = valid_freq
            trainer_kwargs["num_sanity_val_steps"] = 0
        trainer = pl.Trainer(**trainer_kwargs)

        if val_loader is not None:
            trainer.fit(harness, torch_loader, val_loader)
        else:
            trainer.fit(harness, torch_loader)
        self.metrics_history = dict(harness.metrics_history)
        return self

    # --- validation helpers -----------------------------------------------------------------------

    @staticmethod
    def _resolve_preload_to_gpu(device: str, preload_to_gpu: bool | None) -> bool | None:
        """Whether the loader keeps its read window on-GPU. Explicit wins; else never on CPU, auto on GPU.

        ``None`` on GPU lets binded auto-select from cupy availability (GPU-resident batches when cupy is
        present, host otherwise). On CPU training force ``False`` so batches never land on a GPU the model
        isn't on.
        """
        if preload_to_gpu is not None:
            return preload_to_gpu
        return False if str(device) == "cpu" else None

    @staticmethod
    def _resolve_preload(batch_size: int, chunk_size: int, preload_nchunks: int | None) -> int:
        """Prefetch-buffer size in chunks: explicit if given, else a batch (chunk 1) / ~32 batches (chunked).

        The chunked buffer must hold *many* batches: when it drains the loader refills synchronously, and a
        refill of a few chunks stalls training ~400ms (measured on Tahoe/Lustre). A 4-batch buffer stalls
        every ~4 steps (~1.5 steps/s); ~32 batches amortizes refills so steady-state holds ~160 steps/s.
        """
        if preload_nchunks is not None:
            return int(preload_nchunks)
        if chunk_size <= 1:
            return max(1, batch_size)
        return 32 * max(1, batch_size // chunk_size)

    @staticmethod
    def _resolve_split_ratios(split_ratios: Mapping[str, float] | Sequence[float] | None) -> dict[str, float]:
        """Normalize ``split_ratios`` to a ``{"train": .., "val": ..}`` mapping (default ``(0.8, 0.2)``)."""
        if split_ratios is None:
            return {"train": 0.8, "val": 0.2}
        if isinstance(split_ratios, Mapping):
            if {"train", "val"} - set(split_ratios):
                raise ValueError("split_ratios mapping must contain 'train' and 'val' keys.")
            return {"train": float(split_ratios["train"]), "val": float(split_ratios["val"])}
        ratios = tuple(split_ratios)
        if len(ratios) != 2:
            raise ValueError("split_ratios sequence must be (train, val).")
        return {"train": float(ratios[0]), "val": float(ratios[1])}

    def _build_validation(
        self,
        val_scheme: Any,
        condition_fn: Any,
        *,
        val_batch_size: int,
        chunk_size: int = 1,
        preload_nchunks: int | None = None,
        preload_to_gpu: bool | None = None,
        n_val_conditions: int | None,
        metrics: Sequence[str],
        val_num_steps: int,
    ) -> tuple[dict[str, Any], Any]:
        """Build the ``{name: Metric}`` dict + an eval DataLoader (one condition per validation batch)."""
        from binded import EvalLoader, SamplerConfig, split_assignment

        from sc_flow.core.metrics import METRICS_REGISTRY

        unknown = [m for m in metrics if m not in METRICS_REGISTRY]
        if unknown:
            raise KeyError(f"Unknown validation metric(s) {unknown}. Available: {sorted(METRICS_REGISTRY)}.")
        val_metrics = {name: METRICS_REGISTRY[name]() for name in metrics}

        if n_val_conditions is None:
            assignment = split_assignment({"val": val_scheme})
            n_val_conditions = max(int((assignment["split"] == "val").sum()), 1)

        preload = self._resolve_preload(val_batch_size, chunk_size, preload_nchunks)
        cfg = SamplerConfig(batch_size=val_batch_size, chunk_size=chunk_size, preload_nchunks=preload,
                            to="torch", preload_to_gpu=preload_to_gpu)
        eval_loader = EvalLoader(val_scheme, cfg, condition_fn, seed=int(self.seed))

        class _EvalIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, eval_loader, n):
                self._eval_loader = eval_loader
                self._n = n

            def __iter__(self):
                yield from self._eval_loader.iter_conditions(self._n)

        val_loader = torch.utils.data.DataLoader(
            _EvalIterableDataset(eval_loader, n_val_conditions), batch_size=None
        )
        return val_metrics, val_loader

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
        from sc_flow.flow._predict import condition_to_device, integrate_translation

        self.vf.to(device)
        self.vf.eval()

        # Resolve a leaf tuple to its condition dict; a ready condition dict is used as-is.
        if not (isinstance(condition, dict) and all(isinstance(v, np.ndarray) for v in condition.values())):
            if self._condition_fn is None:
                raise RuntimeError("Model must be fitted to resolve leaf conditions.")
            condition = self._condition_fn(condition)
        cond_t = condition_to_device(condition, device)

        trajectory = integrate_translation(
            self.vf,
            np.asarray(x, dtype=np.float32),
            cond_t,
            is_genot=self.objective_name == "genot",
            state_dim=int(self._dims.state),
            num_steps=num_steps,
            seed=int(self.seed if seed is None else seed),
            device=device,
            return_trajectory=return_trajectory,
        )
        return trajectory.cpu().numpy()

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
        :class:`~sc_flow.core.data.CompiledDims`, and the fitted ``condition_fn`` closure — the same
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
