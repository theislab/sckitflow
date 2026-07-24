
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sc_flow.data import FlowSpec

if TYPE_CHECKING:
    from sc_flow.data._compile_obs import CompiledDims, DataInput
    from sc_flow.flow._pooling import PoolingSpec

__all__ = ["FlowMatching", "FlowMatchingConfig"]

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class FlowMatchingConfig:
    """The flow-matching **recipe** — everything that configures a :class:`FlowMatching` other than the
    data (:class:`~sc_flow.data.FlowSpec`) and the fitted weights. A plain, JSON/OmegaConf-friendly object;
    it lives in the flow layer (not the generic ``sc_flow`` core). Build a model with
    ``FlowMatching(spec, config)`` or ``FlowMatching.from_config(spec, mapping_or_config)``.
    """

    pooling: PoolingSpec
    objective: str = "otfm"
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dims: Sequence[int] | None = None
    time_encoder_dims: Sequence[int] | None = None
    condition_embedding_dim: int = 64
    condition_mode: str = "deterministic"
    #: categorical realm -> "embedding" (learned, default) or "onehot" (fixed); a global str or per-realm map.
    condition_encoding: str | Mapping[str, str] = "embedding"
    state_latent_dim: int = 32
    time_latent_dim: int = 16
    source_latent_dim: int = 16
    time_features_id: str | None = None
    num_time_features: int = 256
    max_period: int = 1_000
    regularization: float = 1.0
    sigma: float = 0.0
    match_method: str = "sinkhorn"
    match_kwargs: Mapping[str, Any] | None = None
    seed: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FlowMatchingConfig:
        data = dict(data)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown FlowMatchingConfig field(s): {sorted(unknown)}.")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FlowMatching:

    def __init__(self, spec: FlowSpec, config: FlowMatchingConfig):
        from sc_flow.flow._pooling import validate_pooling_spec

        self.spec = spec
        self.config = config
        # Unpack the recipe into the attributes the internals already use (single source of truth = config).
        self.objective_name = config.objective
        self.hidden_dims = tuple(config.hidden_dims)
        self.decoder_dims = None if config.decoder_dims is None else tuple(config.decoder_dims)
        self.time_encoder_dims = None if config.time_encoder_dims is None else tuple(config.time_encoder_dims)
        self.condition_embedding_dim = config.condition_embedding_dim
        self.condition_mode = config.condition_mode
        self.condition_encoding = config.condition_encoding
        self.state_latent_dim = config.state_latent_dim
        self.time_latent_dim = config.time_latent_dim
        self.source_latent_dim = config.source_latent_dim
        self.time_features_id = config.time_features_id
        self.num_time_features = config.num_time_features
        self.max_period = config.max_period
        self.regularization = config.regularization
        self.sigma = config.sigma
        self.pooling = validate_pooling_spec(config.pooling)
        self.match_method = config.match_method
        self.match_kwargs = dict(config.match_kwargs) if config.match_kwargs else {}
        self.seed = config.seed

        self.vf = None  # MLPVelocity (torch nn.Module holding the weights)
        self.model = None  # alias of vf (the trained weights)
        self.probability_path = None
        self.objective = None
        self._condition_lookup = None
        self._dims = None
        # {metric_name: [mean-over-held-out-conditions per validation pass]}; populated by fit(split_by=...).
        self.metrics_history: dict[str, list[float]] = {}

    @classmethod
    def from_config(cls, spec: FlowSpec, config: FlowMatchingConfig | Mapping[str, Any]) -> FlowMatching:
        """Build from a :class:`FlowMatchingConfig`, or a plain / OmegaConf mapping of its fields."""
        if not isinstance(config, FlowMatchingConfig):
            try:
                from omegaconf import DictConfig, OmegaConf

                if isinstance(config, DictConfig):
                    config = OmegaConf.to_container(config, resolve=True)
            except ImportError:
                pass
            config = FlowMatchingConfig.from_dict(config)
        return cls(spec, config)

    # --- construction helpers -------------------------------------------------------------------

    def _build_vf(self, dims: CompiledDims) -> torch.nn.Module:
        # Dim-materialization only: the recipe (self.config) + compiled dims -> a dim-complete
        # MLPVelocityConfig, then config.build(). No module is constructed here — the config owns build.
        from sc_flow.flow._config import (
            MLPEmbedderConfig,
            MLPVelocityConfig,
            SetEncoderConfig,
            VelocityFieldContext,
        )

        cfg_kwargs: dict[str, Any] = {
            "state_dim": int(dims.state),
            "combiner": {"type": "sc_flow.concat", "version": 1, "config": {}},
            "state_embedder": MLPEmbedderConfig(
                output_dim=int(self.state_latent_dim), mlp_kwargs={"hidden_dims": self.hidden_dims}
            ),
            "time_embedder": MLPEmbedderConfig(
                output_dim=int(self.time_latent_dim),
                mlp_kwargs={"hidden_dims": self.time_encoder_dims} if self.time_encoder_dims is not None else {},
            ),
            "time_features_id": self.time_features_id,
            "num_time_features": self.num_time_features,
            "max_period": self.max_period,
        }
        if self.decoder_dims is not None:
            cfg_kwargs["vf_decoder_mlp_kwargs"] = {"hidden_dims": self.decoder_dims}
        if dims.condition or dims.condition_num_categories:
            embed_dim = int(self.condition_embedding_dim)

            def _categorical_spec(realm: str, n: int) -> dict[str, Any]:
                # "embedding" (learned, default) or "onehot" (fixed) per the condition_encoding knob.
                ce = self.condition_encoding
                kind = ce.get(realm, "embedding") if isinstance(ce, Mapping) else ce
                if kind == "onehot":
                    return {"type": "sc_flow.onehot", "version": 1, "config": {"num_categories": int(n)}}
                if kind != "embedding":
                    raise ValueError(f"condition_encoding for {realm!r} must be 'embedding' or 'onehot', got {kind!r}.")
                return {"type": "sc_flow.embedding", "version": 1,
                        "config": {"num_categories": int(n), "output_dim": embed_dim}}

            # Categorical realms -> embedding/onehot sized by the vocab; feature realms -> an MLP projection
            # of the looked-up vector. The data side emits index / vector accordingly.
            realms: dict[str, dict[str, Any]] = {
                realm: _categorical_spec(realm, n) for realm, n in dims.condition_num_categories.items()
            }
            realms.update(
                {
                    realm: {"type": "sc_flow.feature_mlp", "version": 1,
                            "config": {"input_dim": int(dim), "output_dim": embed_dim, "mlp_kwargs": {}}}
                    for realm, dim in dims.condition.items()
                }
            )
            cfg_kwargs["condition_encoder"] = SetEncoderConfig(
                realms=realms,
                output_dim=embed_dim,
                pooling=self.pooling,
                condition_mode=self.condition_mode,
            )
        if self.objective_name == "genot":
            # GENOT flows noise->target (state space) with the SOURCE cell conditioning the field: enable
            # the source encoder. G1 is same-space — the source cell is the state source, so it is sized by
            # the state dim; the coupling reps (if any) only drive the OT plan. (Cross-space source is G2.)
            cfg_kwargs["source_embedder"] = MLPEmbedderConfig(
                output_dim=int(self.source_latent_dim),
                mlp_kwargs={"input_dim": int(dims.state), "hidden_dims": self.hidden_dims}
            )
        return MLPVelocityConfig(**cfg_kwargs).build(VelocityFieldContext())

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
        metrics: Sequence[str] = ("mean_aggregated_r_squared", "e-dist"),
        val_num_steps: int = 50,
        val_max_source_cells: int | None = 2048,
        callbacks: Sequence[Any] | None = None,
    ) -> FlowMatching:
        from scfit.data import SamplerConfig

        from sc_flow._optional import require
        from sc_flow.data._stream import make_train_loader

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
        self._condition_lookup = compiled.condition_lookup
        self._dims = compiled.dims
        self.metrics_history = {}

        train_scheme, val_scheme = compiled.scheme, None
        if split_by is not None:
            from scfit.data import split_scheme

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
        cfg = SamplerConfig(batch_size=batch_size, chunk_size=chunk_size, preload_nchunks=preload, to="torch")
        loader = make_train_loader(train_scheme, cfg, compiled.condition_lookup, preload_to_gpu=to_gpu)

        # 2. Torch velocity field + probability path, sized from compiled.dims.
        self.vf = self._build_vf(compiled.dims)
        self.model = self.vf
        self.probability_path = self._build_probability_path()

        # 3. Objective selected by name (OT coupling in JAX, everything else torch) + generic harness.
        # sc_flow.TrainingModule is training-only; the perturbation held-out scoring is a Lightning Callback
        # from the flow layer (attached below only when validation is configured).
        from sc_flow.training import TrainingModule
        from sc_flow.flow import ODEPredictor, PerturbationValidationCallback, build_objective

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
                compiled.condition_lookup,
                val_batch_size=val_batch_size or batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                preload_to_gpu=to_gpu,
                n_val_conditions=n_val_conditions,
                metrics=metrics,
                val_num_steps=val_num_steps,
            )

        harness = TrainingModule(self.vf, self.objective, lr=lr)

        # Validation is optional: build the predictor + callback only when a held-out split is configured.
        # The predictor is the same inference seam FlowMatching.predict uses on external data.
        val_callback: PerturbationValidationCallback | None = None
        if val_loader is not None:
            predictor = ODEPredictor(
                is_genot=self.objective_name == "genot", state_dim=int(self._dims.state),
                num_steps=val_num_steps, seed=int(self.seed),
            )
            val_callback = PerturbationValidationCallback(
                predictor=predictor, val_metrics=val_metrics,
                val_max_source_cells=val_max_source_cells,
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
        trainer_callbacks = list(callbacks) if callbacks else []
        if val_callback is not None:
            trainer_callbacks.append(val_callback)
        if trainer_callbacks:
            trainer_kwargs["callbacks"] = trainer_callbacks
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
        self.metrics_history = dict(val_callback.metrics_history) if val_callback is not None else {}
        return self

    # --- validation helpers -----------------------------------------------------------------------

    @staticmethod
    def _resolve_preload_to_gpu(device: str, preload_to_gpu: bool | None) -> bool | None:
        if preload_to_gpu is not None:
            return preload_to_gpu
        return False if str(device) == "cpu" else None

    @staticmethod
    def _resolve_preload(batch_size: int, chunk_size: int, preload_nchunks: int | None) -> int:
        if preload_nchunks is not None:
            return int(preload_nchunks)
        if chunk_size <= 1:
            return max(1, batch_size)
        return 32 * max(1, batch_size // chunk_size)

    @staticmethod
    def _resolve_split_ratios(split_ratios: Mapping[str, float] | Sequence[float] | None) -> dict[str, float]:
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
        condition_lookup: Any,
        *,
        val_batch_size: int,
        chunk_size: int = 1,
        preload_nchunks: int | None = None,
        preload_to_gpu: bool | None = None,
        n_val_conditions: int | None,
        metrics: Sequence[str],
        val_num_steps: int,
    ) -> tuple[dict[str, Any], Any]:
        from scfit.data import SamplerConfig, split_assignment

        from scfit.metrics import METRICS_REGISTRY

        from sc_flow.data._stream import EvalLoader

        unknown = [m for m in metrics if m not in METRICS_REGISTRY]
        if unknown:
            raise KeyError(f"Unknown validation metric(s) {unknown}. Available: {sorted(METRICS_REGISTRY)}.")
        val_metrics = {name: METRICS_REGISTRY[name]() for name in metrics}

        if n_val_conditions is None:
            assignment = split_assignment({"val": val_scheme})
            n_val_conditions = max(int((assignment["split"] == "val").sum()), 1)

        preload = self._resolve_preload(val_batch_size, chunk_size, preload_nchunks)
        cfg = SamplerConfig(batch_size=val_batch_size, chunk_size=chunk_size, preload_nchunks=preload, to="torch")
        eval_loader = EvalLoader(val_scheme, cfg, condition_lookup, seed=int(self.seed))

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
        condition_mask: Mapping[str, np.ndarray] | None = None,
        device: str = "cpu",
        num_steps: int = 50,
        return_aux: bool = False,
        seed: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        if self.vf is None:
            raise RuntimeError("Model must be fitted before predict() can be called.")
        from sc_flow.flow import ODEPredictor

        self.vf.to(device)
        self.vf.eval()

        # Resolve a leaf tuple to its condition dict; a ready condition dict is used as-is.
        if not (isinstance(condition, dict) and all(isinstance(v, np.ndarray) for v in condition.values())):
            if self._condition_lookup is None:
                raise RuntimeError("Model must be fitted to resolve leaf conditions.")
            condition = self._condition_lookup(condition)

        # External numpy is coerced to torch on the model's device at the integrator boundary; the batch
        # contract matches what the eval loader emits, so predict() and validation share one code path.
        batch = {
            # TODO: why the cast here
            "source": np.asarray(x, dtype=np.float32),
            "condition": condition,
            "condition_mask": None if condition_mask is None else dict(condition_mask),
        }
        predictor = ODEPredictor(
            is_genot=self.objective_name == "genot", state_dim=int(self._dims.state),
            num_steps=num_steps, seed=int(self.seed if seed is None else seed),
        )
        if return_aux:
            pred, aux = predictor.predict_with_aux(self.vf, batch)
            aux_np = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in aux.items()}
            return pred.detach().cpu().numpy(), aux_np
        return predictor.predict(self.vf, batch).detach().cpu().numpy()

    # --- persistence ------------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import json

        import cloudpickle
        from safetensors.torch import save_model

        from sc_flow.flow._config import FORMAT_VERSION

        if self.vf is None:
            raise RuntimeError("Model must be fitted before save().")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Portable model: config.json (envelope) + model.safetensors. to_config() raises before any write
        # if the VF holds a runtime-only custom module. save_model dedupes the condition encoder's shared
        # (double-registered) tensors.
        config = {
            "format_version": FORMAT_VERSION,
            "flow_matching": self.config.to_dict(),  # the recipe (single source of truth)
            "architecture": self.vf.to_config().to_spec(),
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))
        save_model(self.vf, str(path / "model.safetensors"))

        # Trusted data sidecar (spec / dims / condition_lookup) — cloudpickle, isolated and documented.
        with open(path / "data_state.pkl", "wb") as f:
            cloudpickle.dump({"spec": self.spec, "dims": self._dims, "condition_lookup": self._condition_lookup}, f)

    @classmethod
    def load(cls, path: str | Path) -> FlowMatching:
        import json

        import cloudpickle
        from safetensors.torch import load_model

        from sc_flow.flow._config import MLPVelocityConfig
        from sc_flow.flow._vf import MLPVelocity

        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        with open(path / "data_state.pkl", "rb") as f:
            data_state = cloudpickle.load(f)

        model = cls(data_state["spec"], FlowMatchingConfig.from_dict(config["flow_matching"]))
        model._dims = data_state["dims"]
        model._condition_lookup = data_state["condition_lookup"]
        model.vf = MLPVelocity.from_config(MLPVelocityConfig.from_spec(config["architecture"]))
        load_model(model.vf, str(path / "model.safetensors"), strict=True)
        model.model = model.vf
        model.probability_path = model._build_probability_path()
        return model
