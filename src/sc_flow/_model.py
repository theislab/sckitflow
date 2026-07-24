"""``FlowMatching`` — a *builder* that turns a recipe into Lightning-native training pieces.

The recipe (a plain OmegaConf/dict, sections ``data`` / ``model`` / ``sampler`` / ``trainer``) fully
describes the run: where the cells are, the covariate contract, the velocity-field hyperparameters, and
the loop knobs. ``FlowMatching(recipe)`` reads the obs once via **scfit**'s readers, sizes the network, and
exposes three Lightning-native things — it does **no** training itself:

* :attr:`module`     — the :class:`scfit.training.TrainingModule` (a ``LightningModule``: model + objective),
* :attr:`datamodule` — a :class:`~lightning.pytorch.LightningDataModule` whose loaders are **scfit**
  :class:`~scfit.data.Stream`\\s wired into a :class:`~scfit.data.Loader` (the data mechanics live entirely
  in scfit — this layer only *describes* the problem as Streams),
* :attr:`callbacks`  — the perturbation held-out validation callback (only when a split is configured).

Training is then plain Lightning::

    fm = FlowMatching(recipe)
    trainer = pl.Trainer(max_steps=..., accelerator="gpu", val_check_interval=..., callbacks=fm.callbacks)
    trainer.fit(fm.module, datamodule=fm.datamodule)

The perturbed cells are the primary Stream (target); the matched controls a linked Stream (source), drawn
from the same ``match_on`` context. Per-stream sampler kwargs let the target stream read in fast contiguous
chunks while the control stream sits in memory at chunk_size=1 (short runs are fine in RAM).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch

__all__ = ["FlowMatching", "FlowMatchingConfig"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Dims:
    """Tensor dimensions read from the data — feeds the velocity-field builder."""

    state: int  # feature width of the streamed state rep
    condition: Mapping[str, int]  # feature realms (lookup vector) -> width; empty for categorical-only
    condition_num_categories: Mapping[str, int]  # categorical realms -> vocab size
    max_comb: int = 1  # condition set/pool length (max cols across realms)


@dataclass(kw_only=True)
class FlowMatchingConfig:
    """The **model** recipe — the velocity-field + objective hyperparameters (the ``model`` recipe section).

    Data/paths/sampler/trainer live in the sibling recipe sections (see :class:`FlowMatching`); this is only
    the model. JSON/OmegaConf-friendly.
    """

    pooling: Any  # PoolingSpec (a discriminated ComponentSpec dict)
    objective: str = "otfm"
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dims: Sequence[int] | None = None
    time_encoder_dims: Sequence[int] | None = None
    condition_embedding_dim: int = 64
    condition_mode: str = "deterministic"
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


def _to_container(cfg: Any) -> Any:
    """Resolve an OmegaConf node to a plain python container (no-op for dict/list/scalars)."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return cfg


def _sample_rep_to_key(sample_rep: str) -> str:
    return "X" if sample_rep == "X" else f"obsm/{sample_rep}"


@dataclass(frozen=True)
class _Problem:
    """Everything read from obs to describe the streams + size the net (the 'compile', via scfit readers)."""

    source: Any  # resolved scfit Container (backed)
    rep_loc: str  # "X" | "obsm/<rep>"
    group_by: tuple[str, ...]  # match context + condition cols
    match_on: tuple[str, ...]  # the control<->perturbed binding columns
    pert_leaves: list[tuple]  # perturbed target leaves (group_by combinations)
    ctrl_leaves: list[tuple]  # control source leaves
    label_lookup: dict[tuple, dict[str, np.ndarray]]  # pert leaf -> {realm: (1, set) categorical index}
    dims: _Dims
    pert_leaf_min_run: dict[tuple, int]  # pert leaf -> shortest contiguous on-disk run (per path); for min_runs_per_leaf
    ctrl_source: Any = None  # optional separate control store (e.g. controls.zarr)


def _min_run_per_leaf(source: Any, group_by: Sequence[str]) -> dict[tuple, int]:
    """Shortest contiguous on-disk run per leaf, computed PER source path.

    annbatch's ``ClassSampler`` reads a sampled leaf as contiguous ``chunk_size`` slices and rejects a run
    shorter than ``chunk_size``; a slice never crosses a backing, so runs are per-path. Computing per path
    (and taking the global min per leaf) therefore never *over*-credits a leaf that is split across plates —
    it can only over-drop, never leave a leaf that annbatch would then reject. Loops over runs (few), not
    rows (many). Obs cols are already in RAM on the backed AnnData(s), so this reads directly from RAM.
    """
    from scfit.data._io import obs_columns

    elems = source if isinstance(source, list) else [source]
    cols = list(group_by)
    min_run: dict[tuple, int] = {}
    for el in elems:
        df = el.obs[cols] if hasattr(el, "obs") else obs_columns(el, cols)
        arr = df.loc[:, cols].to_numpy()  # (n, k) leaf values in stored (path) order
        n = len(arr)
        if n == 0:
            continue
        change = np.ones(n, dtype=bool)  # a new run starts wherever the leaf row differs from its predecessor
        if n > 1:
            change[1:] = (arr[1:] != arr[:-1]).any(axis=1)
        starts = np.flatnonzero(change)
        lengths = np.diff(np.append(starts, n))
        for row, length in zip(arr[starts], lengths):
            leaf, length = tuple(row), int(length)
            if leaf not in min_run or length < min_run[leaf]:
                min_run[leaf] = length
    return min_run


def _read_problem(
    data: Any,
    *,
    sample_rep: str,
    control_key: str,
    perturbation_covariates: Mapping[str, Sequence[str]],
    split_covariates: Sequence[str],
    controls: Any | None = None,
) -> _Problem:
    """Read obs once (via scfit's readers) → the streams' description + the net's dims. No cells materialized.

    ``split_covariates`` are the control↔perturbed match context; ``perturbation_covariates`` map each
    condition realm to its obs column(s). Leaves are grouped over ``match_on + condition cols``; the
    perturbed leaves get a categorical ``label_lookup`` (one integer index per realm column, shape ``(1, set)``
    so the objective broadcasts it over the batch); dims come from the rep header + the per-realm vocab.
    """
    import pandas as pd
    from scfit.data._io import get_from_container, obs_columns, open_source

    match_on = tuple(split_covariates)
    realm_cols = {realm: tuple(cols) for realm, cols in perturbation_covariates.items()}
    cond_cols = tuple(dict.fromkeys(c for cols in realm_cols.values() for c in cols))
    group_by = tuple(dict.fromkeys([*match_on, *cond_cols]))
    rep_loc = _sample_rep_to_key(sample_rep)

    source = open_source(data, keys=[rep_loc], cols=[*group_by, control_key])
    obs = obs_columns(source, [*group_by, control_key])
    for col in group_by:
        if not isinstance(obs[col].dtype, pd.CategoricalDtype):
            obs[col] = obs[col].astype("category")

    def _leaves(frame) -> list[tuple]:
        return [tuple(r) for r in frame.loc[:, list(group_by)].drop_duplicates().to_numpy()]

    if controls is not None:
        ctrl_source = open_source(controls, keys=[rep_loc], cols=[*group_by, control_key])
        ctrl_obs = obs_columns(ctrl_source, [*group_by, control_key])
        for col in group_by:
            if not isinstance(ctrl_obs[col].dtype, pd.CategoricalDtype):
                ctrl_obs[col] = ctrl_obs[col].astype("category")
        ctrl_leaves = _leaves(ctrl_obs)
        ctrl_mask = obs[control_key].to_numpy().astype(bool) if control_key in obs else np.zeros(len(obs), dtype=bool)
        pert_obs = obs.loc[~ctrl_mask] if control_key in obs else obs
        pert_leaves = _leaves(pert_obs)
    else:
        ctrl_source = None
        ctrl_mask = obs[control_key].to_numpy().astype(bool)
        pert_obs = obs.loc[~ctrl_mask]
        pert_leaves = _leaves(pert_obs)
        ctrl_leaves = _leaves(obs.loc[ctrl_mask])



    # Per realm: a categorical vocab over its column values (perturbed cells only — the control token is not
    # a perturbation category); the leaf's index per realm column.
    col_idx = {c: group_by.index(c) for c in group_by}
    vocab: dict[str, dict[str, int]] = {}
    num_categories: dict[str, int] = {}
    for realm, cols in realm_cols.items():
        cats = sorted({str(v) for v in pert_obs[list(cols)].to_numpy().reshape(-1)})
        vocab[realm] = {v: i for i, v in enumerate(cats)}
        num_categories[realm] = len(cats)
    max_comb = max((len(cols) for cols in realm_cols.values()), default=1)

    def _label(leaf: tuple) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for realm, cols in realm_cols.items():
            idx = np.asarray([vocab[realm][str(leaf[col_idx[c]])] for c in cols])
            out[realm] = idx[np.newaxis, :]  # (1, set) — broadcast over the batch by the objective
        return out

    label_lookup = {leaf: _label(leaf) for leaf in pert_leaves}
    state_dim = int(get_from_container(source, rep_loc)[0].shape[1])
    dims = _Dims(state=state_dim, condition={}, condition_num_categories=num_categories, max_comb=max_comb)
    min_run = _min_run_per_leaf(source, group_by)
    pert_leaf_min_run = {leaf: min_run.get(leaf, 0) for leaf in pert_leaves}
    return _Problem(
        source, rep_loc, group_by, match_on, pert_leaves, ctrl_leaves, label_lookup, dims, pert_leaf_min_run,
        ctrl_source=ctrl_source
    )


class _SemanticBatches(torch.utils.data.IterableDataset):
    """Adapt scfit's ``{primary, ctrl, labels}`` batches to the objective's ``{source, target, condition}``.

    The primary (perturbed) stream is the target, the ``ctrl`` link the source; ``labels["primary"]`` is the
    batch's condition. The full per-stream rep dicts pass through as ``source_reps``/``target_reps`` for the
    coupling objectives.
    """

    def __init__(self, loader: Any, rep_loc: str) -> None:
        self._loader = loader
        self._loc = rep_loc

    def __iter__(self):
        for b in self._loader:
            out: dict[str, Any] = {
                "source": b["ctrl"][self._loc],
                "target": b["primary"][self._loc],
                "source_reps": b["ctrl"],
                "target_reps": b["primary"],
            }
            labels = b.get("labels")
            if labels is not None and "primary" in labels:
                out["condition"] = labels["primary"]
            yield out


class FlowMatchingDataModule(pl.LightningDataModule):
    """A ``LightningDataModule`` whose train/val loaders are scfit :class:`~scfit.data.Loader`s (Streams)."""

    def __init__(self, train_loader: Any, val_loader: Any, rep_loc: str) -> None:
        super().__init__()
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._loc = rep_loc

    def train_dataloader(self):
        return torch.utils.data.DataLoader(_SemanticBatches(self._train_loader, self._loc), batch_size=None)

    def val_dataloader(self):
        if self._val_loader is None:
            return None
        return torch.utils.data.DataLoader(_SemanticBatches(self._val_loader, self._loc), batch_size=None)


class FlowMatching:
    """Builder: recipe -> (LightningModule, LightningDataModule, callbacks). See the module docstring."""

    def __init__(self, recipe: Mapping[str, Any]):
        recipe = _to_container(recipe)
        data = dict(recipe.get("data", {}))
        sampler = dict(recipe.get("sampler", {}))
        trainer = dict(recipe.get("trainer", {}))
        self.config = FlowMatchingConfig.from_dict(_to_container(recipe["model"]))
        self._data, self._sampler, self._trainer = data, sampler, trainer

        # Unpack the model recipe into the attributes _build_vf uses (single source of truth = self.config).
        c = self.config
        self.objective_name = c.objective
        self.hidden_dims = tuple(c.hidden_dims)
        self.decoder_dims = None if c.decoder_dims is None else tuple(c.decoder_dims)
        self.time_encoder_dims = None if c.time_encoder_dims is None else tuple(c.time_encoder_dims)
        self.condition_embedding_dim = c.condition_embedding_dim
        self.condition_mode = c.condition_mode
        self.condition_encoding = c.condition_encoding
        self.state_latent_dim = c.state_latent_dim
        self.time_latent_dim = c.time_latent_dim
        self.source_latent_dim = c.source_latent_dim
        self.time_features_id = c.time_features_id
        self.num_time_features = c.num_time_features
        self.max_period = c.max_period
        self.regularization = c.regularization
        self.sigma = c.sigma
        self.match_method = c.match_method
        self.match_kwargs = dict(c.match_kwargs) if c.match_kwargs else {}
        self.seed = int(c.seed)

        from sc_flow.flow._pooling import validate_pooling_spec

        self.pooling = validate_pooling_spec(_to_container(c.pooling))

        torch.manual_seed(self.seed)
        self._build()

    # --- build -----------------------------------------------------------------------------------

    def _build(self) -> None:
        from scfit.metrics import METRICS_REGISTRY

        from sc_flow.flow import ODEPredictor, PerturbationValidationCallback, build_objective

        d = self._data
        source = self._resolve_source(d)
        ctrl_source = self._resolve_source({"data": d["controls"]}) if ("controls" in d and d["controls"]) else None
        self._problem = _read_problem(
            source,
            controls=ctrl_source,
            sample_rep=d.get("sample_rep", "X"),
            control_key=d.get("control_key", "is_control"),
            perturbation_covariates=d.get("perturbation_covariates", {"drug": ["drug"]}),
            split_covariates=d.get("split_covariates", []),
        )
        self._dims = self._problem.dims

        # torch velocity field (weights) + probability path + objective (OT coupling in JAX, rest torch).
        self.vf = self._build_vf(self._dims)
        self.model = self.vf
        self.probability_path = self._build_probability_path()
        self.objective = build_objective(
            self.objective_name, self.probability_path, condition_mode=self.condition_mode,
            regularization=self.regularization, coupling_locs={}, match_method=self.match_method,
            match_kwargs=self.match_kwargs, seed=self.seed,
        )

        # Data: describe the streams (target + matched control), split off held-out conditions if configured.
        train_pert, val_pert = self._split_leaves()
        train_loader = self._build_loader(train_pert)
        val_loader = self._build_loader(val_pert) if val_pert else None

        from sc_flow.training import TrainingModule

        self._module = TrainingModule(self.vf, self.objective, lr=float(self._trainer.get("lr", 1e-4)))
        self._datamodule = FlowMatchingDataModule(train_loader, val_loader, self._problem.rep_loc)

        self._callbacks: list[Any] = []
        if val_loader is not None:
            metrics = self._trainer.get("metrics", ["mean_aggregated_r_squared", "e-dist"])
            unknown = [m for m in metrics if m not in METRICS_REGISTRY]
            if unknown:
                raise KeyError(f"Unknown validation metric(s) {unknown}; available: {sorted(METRICS_REGISTRY)}.")
            predictor = ODEPredictor(
                is_genot=self.objective_name == "genot", state_dim=int(self._dims.state),
                num_steps=int(self._trainer.get("val_num_steps", 50)), seed=self.seed,
            )
            self._callbacks.append(PerturbationValidationCallback(
                predictor=predictor, val_metrics={m: METRICS_REGISTRY[m]() for m in metrics},
                val_max_source_cells=self._trainer.get("val_max_source_cells", 2048),
            ))

    def _resolve_source(self, d: Mapping[str, Any]) -> Any:
        """The recipe carries the data location; a caller may also hand an in-memory object as ``data``."""
        return d["data"] if "data" in d else d["paths"] if "paths" in d else d.get("source")

    def _split_leaves(self) -> tuple[list[tuple], list[tuple]]:
        """Hold out whole conditions by ``split_by`` columns (train, val leaf lists over the perturbed set)."""
        split_by = self._data.get("split_by") or None
        pert = self._problem.pert_leaves
        if not split_by:
            return pert, []
        cols = [split_by] if isinstance(split_by, str) else list(split_by)
        idx = [self._problem.group_by.index(c) for c in cols]
        ratios = self._data.get("split_ratios", {"train": 0.8, "val": 0.2})
        val_frac = float(ratios["val"]) / (float(ratios["train"]) + float(ratios["val"]))
        keys = sorted({tuple(str(lf[i]) for i in idx) for lf in pert})
        rng = np.random.default_rng(self.seed)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_frac)) if len(keys) > 1 else 0
        val_keys = set(keys[:n_val])
        train = [lf for lf in pert if tuple(str(lf[i]) for i in idx) not in val_keys]
        val = [lf for lf in pert if tuple(str(lf[i]) for i in idx) in val_keys]
        logger.info("split_by=%s held out %d/%d condition groups", cols, len(val_keys), len(keys))
        return train, val

    def _build_loader(self, pert_leaves: list[tuple]) -> Any:
        """Describe the target (perturbed) + matched-control Streams over the source and wire the Loader.

        Target: fast contiguous chunked reads. Control: in memory at chunk=1 (short runs OK), matched on the
        context. All data mechanics (matching, run-length, streaming) are scfit's — this only describes.
        """
        from scfit.data import Loader, Stream

        p = self._problem
        s = self._sampler
        batch = int(s.get("batch_size", 1024))
        chunk = int(s.get("chunk_size", 1))
        prefetch = int(s.get("prefetch_factor", 2))
        preload = max(1, batch // chunk) * prefetch

        # Drop target leaves whose shortest contiguous run < min_runs_per_leaf so the annbatch ClassSampler's
        # chunk_size run-length rule holds (a fragmented leaf would else abort loader construction). Only
        # relevant for chunk_size>1; a dropped leaf simply leaves the target selection (scfit excludes weight-0).
        min_run = int(s.get("min_runs_per_leaf", 0))
        if chunk > 1 and min_run > 0:
            kept = [lf for lf in pert_leaves if p.pert_leaf_min_run.get(lf, 0) >= min_run]
            if (dropped := len(pert_leaves) - len(kept)):
                logger.info("min_runs_per_leaf=%d: dropped %d/%d target leaves (shortest run < it; chunk_size=%d)",
                            min_run, dropped, len(pert_leaves), chunk)
            if not kept:
                raise ValueError(f"min_runs_per_leaf={min_run} dropped every target leaf; lower it or chunk_size ({chunk}).")
            pert_leaves = kept

        # control leaves whose context matches a kept perturbed leaf (avoid an orphaned target).
        ctx_pos = [p.group_by.index(c) for c in p.match_on]
        pert_ctx = {tuple(str(lf[i]) for i in ctx_pos) for lf in pert_leaves}
        ctrl_leaves = [lf for lf in p.ctrl_leaves if tuple(str(lf[i]) for i in ctx_pos) in pert_ctx]

        primary = Stream(
            p.source, group_by=p.group_by, rep=p.rep_loc,
            weights={lf: 1.0 for lf in pert_leaves},
            label_lookup={lf: p.label_lookup[lf] for lf in pert_leaves},
            batch_size=batch, chunk_size=chunk, preload_nchunks=preload,
        )
        ctrl = Stream(
            p.ctrl_source or p.source, group_by=p.group_by, rep=p.rep_loc,
            weights={lf: 1.0 for lf in ctrl_leaves}, match_on=p.match_on, in_memory=True,
            batch_size=batch, chunk_size=1, preload_nchunks=preload * 2,  # lean preload (2 batches of cells)
        )

        to_gpu = str(self._trainer.get("device", "cpu")) not in ("cpu", "mps")
        return Loader(primary, {"ctrl": ctrl}, seed=self.seed, to="torch", preload_to_gpu=to_gpu)

    def _build_vf(self, dims: _Dims) -> torch.nn.Module:
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
                ce = self.condition_encoding
                kind = ce.get(realm, "embedding") if isinstance(ce, Mapping) else ce
                if kind == "onehot":
                    return {"type": "sc_flow.onehot", "version": 1, "config": {"num_categories": int(n)}}
                if kind != "embedding":
                    raise ValueError(f"condition_encoding for {realm!r} must be 'embedding' or 'onehot', got {kind!r}.")
                return {"type": "sc_flow.embedding", "version": 1,
                        "config": {"num_categories": int(n), "output_dim": embed_dim}}

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
                realms=realms, output_dim=embed_dim, pooling=self.pooling, condition_mode=self.condition_mode,
            )
        if self.objective_name == "genot":
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
            return LinearGaussianProbabilityPath(sigma=self.sigma, prng=torch.Generator().manual_seed(self.seed))
        return LinearDiracProbabilityPath(sigma=0.0)

    # --- Lightning-native pieces -----------------------------------------------------------------

    @property
    def module(self) -> Any:
        """The ``LightningModule`` (model + objective) to hand to ``Trainer.fit``."""
        return self._module

    @property
    def datamodule(self) -> Any:
        """The ``LightningDataModule`` (scfit Streams → Loader) to hand to ``Trainer.fit``."""
        return self._datamodule

    @property
    def callbacks(self) -> list[Any]:
        """Held-out perturbation-validation callbacks (empty when no split is configured)."""
        return list(self._callbacks)

    # --- persistence -----------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import json

        from safetensors.torch import save_model

        from sc_flow.flow._config import FORMAT_VERSION

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "format_version": FORMAT_VERSION,
            "flow_matching": self.config.to_dict(),
            "architecture": self.vf.to_config().to_spec(),
        }
        (path / "config.json").write_text(json.dumps(config, indent=2, default=str))
        save_model(self.vf, str(path / "model.safetensors"))
