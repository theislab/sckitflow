from __future__ import annotations

import logging
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, overload

import cloudpickle
import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from sckitflow._types import PredictionData
from sckitflow.core._types import StepData, TMatchFn
from sckitflow.core.methods import INFERENCE_PROTOCOLS_REGISTRY, TRAINING_PROTOCOLS_REGISTRY
from sckitflow.core.methods._base import (
    FlowSpecs,
    MatchedTrainingProtocol,
    ProtocolSpecs,
    SupportsInference,
    SupportsTraining,
)
from sckitflow.core.methods._opt import OptimConfig, OptimizationManager
from sckitflow.core.nn._modules import BaseModule
from sckitflow.data._dims_registry import DataDimensionalitiesRegistry
from sckitflow.data._manager import DataManager, DataManagerKwargs
from sckitflow.trainer._callbacks import BaseCallback, TrainingCallbacks
from sckitflow.trainer._trainer import Trainer

if TYPE_CHECKING:
    import torch

    from sckitflow.data._loader import LoaderKwargs

__all__ = ["Model", "ModelBuilder"]


def _build_module(
    module: BaseModule | None = None,
    module_cls: type[BaseModule] | None = None,
    data_dims: DataDimensionalitiesRegistry | None = None,
    module_kwargs: dict[str, Any] | None = None,
) -> BaseModule:
    """Returns an instantiated module from the given inputs.

    Resolution order:
    1. `module` is returned as-is.
    2. `module_cls` is instantiated via its `init_from_dims_registry`
       classmethod, using `data_dims` and `module_kwargs`.
    3. Otherwise, a `ValueError` is raised.
    """
    if module is not None:
        return module

    if module_cls is not None:
        if data_dims is None:
            raise ValueError("When initializing the module with `module_cls`, `data_dims` should be provided.")
        module_kwargs = {} if module_kwargs is None else module_kwargs
        return module_cls.init_from_dims_registry(data_dims, **module_kwargs)

    raise ValueError("At least one of `module` or `module_cls` must be passed.")


def _resolve_specs(
    module: BaseModule,
    dtype: torch.dtype | None = None,
    device_id: str | None = None,
    flow_kwargs: Mapping[str, Any] | None = None,
    is_flow: bool = False,
) -> ProtocolSpecs:
    """Resolve the shared `ProtocolSpecs` (or `FlowSpecs`) for the model.

    Resolution order:
    1. If `flow_kwargs` is provided (even as `{}`), a `FlowSpecs` is built with
       those kwargs on top of the module.
    3. Otherwise, a plain `ProtocolSpecs` is built from the module.
    """
    if is_flow:
        flow_kwargs = {} if flow_kwargs is None else flow_kwargs
        conflicting = {"dtype", "device_id"} & flow_kwargs.keys()
        if conflicting:
            raise ValueError(
                f"`flow_kwargs` must not contain {sorted(conflicting)}; pass them as "
                "top-level `ModelKwargs` so both protocols share the same value."
            )
        return FlowSpecs(module, device_id=device_id, dtype=dtype, **flow_kwargs)

    return ProtocolSpecs(module, device_id=device_id, dtype=dtype)


@overload
def _build_protocol(
    specs: ProtocolSpecs | FlowSpecs,
    mode: Literal["training"],
    protocol_cls: type[SupportsTraining] | None = None,
    protocol_id: str | None = None,
    protocol_kwargs: dict[str, Any] | None = None,
    *,
    allow_none: Literal[False] = False,
) -> SupportsTraining: ...


@overload
def _build_protocol(
    specs: ProtocolSpecs | FlowSpecs,
    mode: Literal["training"],
    protocol_cls: type[SupportsTraining] | None = None,
    protocol_id: str | None = None,
    protocol_kwargs: dict[str, Any] | None = None,
    *,
    allow_none: Literal[True],
) -> SupportsTraining | None: ...


@overload
def _build_protocol(
    specs: ProtocolSpecs | FlowSpecs,
    mode: Literal["inference"],
    protocol_cls: type[SupportsInference] | None = None,
    protocol_id: str | None = None,
    protocol_kwargs: dict[str, Any] | None = None,
    *,
    allow_none: Literal[False] = False,
) -> SupportsInference: ...


@overload
def _build_protocol(
    specs: ProtocolSpecs | FlowSpecs,
    mode: Literal["inference"],
    protocol_cls: type[SupportsInference] | None = None,
    protocol_id: str | None = None,
    protocol_kwargs: dict[str, Any] | None = None,
    *,
    allow_none: Literal[True],
) -> SupportsInference | None: ...


def _build_protocol(
    specs: ProtocolSpecs | FlowSpecs,
    mode: Literal["inference", "training"],
    protocol_cls: type[object] | None = None,
    protocol_id: str | None = None,
    protocol_kwargs: dict[str, Any] | None = None,
    allow_none: bool = False,
) -> SupportsTraining | SupportsInference | None:
    """Returns an instantiated protocol from the given shared `specs`.

    Resolution order:
    1. `protocol_cls` is instantiated with `(specs, **protocol_kwargs)`.
    2. `protocol_id` is looked up in the mode-specific registry and
       instantiated the same way.
    3. When `allow_none` is `True`, `None` is returned.
    4. Otherwise, a `ValueError` is raised.

    The protocol receives the shared specs instance, so training and inference
    protocols constructed with the same `specs` share the underlying module,
    dtype, device, and (when `specs` is a `FlowSpecs`) flow configuration.
    """
    if protocol_cls is not None:
        protocol_kwargs = {} if protocol_kwargs is None else protocol_kwargs
        return protocol_cls(specs, **protocol_kwargs)

    elif protocol_id is not None:
        protocol_kwargs = {} if protocol_kwargs is None else protocol_kwargs

        if mode == "training":
            registry = TRAINING_PROTOCOLS_REGISTRY
        elif mode == "inference":
            registry = INFERENCE_PROTOCOLS_REGISTRY
        else:
            raise ValueError(f"Invalid mode {mode}: set to `training` or `inference`.")

        protocol_cls = registry[protocol_id]
        return protocol_cls(specs, **protocol_kwargs)

    elif allow_none:
        return None

    else:
        raise ValueError(
            "At least one of `protocol`, `protocol_cls` or `protocol_id` "
            "must be passed or `allow_none` should be set to True."
        )


def _get_matched_protocol(
    training_protocol: SupportsTraining,
    match_fn: TMatchFn | None = None,
) -> SupportsTraining:
    """Wrap `training_protocol` with `match_fn` when one is provided.

    - `match_fn is None` → return `training_protocol` unchanged.
    - `match_fn` provided, protocol already matched → unwrap, then re-wrap.
    - `match_fn` provided, protocol unmatched → wrap.

    The returned object always satisfies `SupportsTraining`.
    """
    if match_fn is None:
        return training_protocol
    if isinstance(training_protocol, MatchedTrainingProtocol):
        training_protocol = training_protocol.protocol
    return MatchedTrainingProtocol(training_protocol, match_fn)


class ModelKwargs(TypedDict, total=False):
    """Keyword arguments to initialize the model.

    :param module: The neural module used to instantiate the model.
        When provided, it takes precedence over `module_cls`.
    :param module_cls: A neural module class, inheriting from `BaseModule`;
        it is initialized only when `module` is `None`, using `module_kwargs`
        as keyword arguments. Required when `module` is `None`.
    :param module_kwargs: Optional keyword arguments used to initialize the
        neural module; only used when initializing it from `module_cls`.
    :param protocol_specs: An explicit `ProtocolSpecs` (or `FlowSpecs`) to
        share with both protocols. When provided, `flow_kwargs` is ignored.
        Defaults to `None`.
    :param flow_kwargs: Keyword arguments used to build a `FlowSpecs` on top
        of the module, shared by both protocols. Pass `{}` to build a
        `FlowSpecs` with all defaults, or provide the flow configuration
        (`probability_path`, `time_sampler`, `noise_sampler`,
        `generate_from_noise`) to share it between training and inference.
        When `None`, a plain `ProtocolSpecs` is built instead. Defaults to
        `None`. Values must be picklable for `Model.save` to work (module-level
        functions or bound methods of picklable objects; not lambdas).
    :param training_protocol_cls: A class satisfying `SupportsTraining`, to be
        initialized from the shared specs. When provided, it takes precedence
        over `training_protocol_id`. Initialized with
        `training_protocol_kwargs`.
    :param training_protocol_id: Identifier of a training protocol in
        `TRAINING_PROTOCOLS_REGISTRY`. Used only when `training_protocol_cls`
        is `None`. Initialized from the shared specs with
        `training_protocol_kwargs`.
    :param training_protocol_kwargs: Keyword arguments used to initialize the
        training protocol.
    :param inference_protocol_cls: A class satisfying `SupportsInference`, to
        be initialized from the shared specs. When provided, it takes
        precedence over `inference_protocol_id`. Initialized with
        `inference_protocol_kwargs`.
    :param inference_protocol_id: Identifier of an inference protocol in
        `INFERENCE_PROTOCOLS_REGISTRY`. Used only when
        `inference_protocol_cls` is `None`. Initialized from the shared specs
        with `inference_protocol_kwargs`.
    :param inference_protocol_kwargs: Keyword arguments used to initialize the
        inference protocol.
    :param match_fn: Matching callable applied during training. When provided,
        the training protocol is wrapped in `MatchedTrainingProtocol`, which
        matches `StepData` before `compute_loss`. May be overridden per
        `Model.train` call. Must be picklable for `Model.save` to work.
    """

    module: BaseModule | None
    module_cls: type[BaseModule] | None
    module_kwargs: dict[str, Any] | None
    dtype: torch.dtype | None
    device_id: str | None
    is_flow: bool
    flow_kwargs: dict[str, Any] | None
    training_protocol_cls: type[SupportsTraining] | None
    training_protocol_id: str | None
    training_protocol_kwargs: dict[str, Any] | None
    inference_protocol_cls: type[SupportsInference] | None
    inference_protocol_id: str | None
    inference_protocol_kwargs: dict[str, Any] | None
    match_fn: TMatchFn | None


class ModelBuilder:
    """Two-step builder for a :class:`Model`.

    Step one (:meth:`from_adata`) prepares the *data* side: it initializes the
    :class:`DataManager` from the schema keyword arguments and derives the data
    dimensionalities. Step two (:meth:`build`) attaches the module and the
    training/inference protocols and returns a ready-to-train :class:`Model`.

    Any preprocessing of the state representation (e.g. PCA, normalization)
    must be done by the caller *before* :meth:`from_adata`, and the resulting
    representation passed via the ``sample_rep`` keyword argument.
    """

    def __init__(
        self,
        dm: DataManager,
        data_dims: DataDimensionalitiesRegistry,
    ) -> None:
        """See :meth:`from_adata` for the usual entry point."""
        self._dm = dm
        self._data_dims = data_dims

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        **dm_kwargs: Unpack[DataManagerKwargs],
    ) -> ModelBuilder:
        """Prepare the data side of a model from an annotated data object.

        :param adata: The annotated data object used to fit the schema. Any
            preprocessing of the state representation must already have been
            applied by the caller.
        :type adata: class: `AnnData`

        :param dm_kwargs: Keyword arguments forwarded to :class:`DataManager`.
        """
        dm = DataManager(**dm_kwargs)
        data_dims = dm.get_data_dimensionalities(adata)
        return cls(dm, data_dims)

    @property
    def dm(self) -> DataManager:
        """The fitted data manager."""
        return self._dm

    @property
    def data_dims(self) -> DataDimensionalitiesRegistry:
        """The data dimensionalities derived from the registration data."""
        return self._data_dims

    def build(self, **model_kwargs: Unpack[ModelKwargs]) -> Model:
        """Attach a training protocol, an inference protocol, and a module to the Model."""
        return Model(self._dm, self._data_dims, **model_kwargs)


class Model:
    def __init__(
        self, dm: DataManager, data_dims: DataDimensionalitiesRegistry, **model_kwargs: Unpack[ModelKwargs]
    ) -> None:
        """Initialize a model from a fitted data manager and its dimensionalities.

        Usually constructed through :class:`ModelBuilder` rather than directly.

        :param dm: The fitted data manager describing the data schema.
        :type dm: class: `DataManager`

        :param data_dims: The data dimensionalities derived from the registration data.
        :type data_dims: class: `DataDimensionalitiesRegistry`

        :param model_kwargs: Module and protocol configuration; see
            :class:`ModelKwargs` for the accepted options.
        """
        # ----- Store data manager and dimensionalities -----
        self._dm = dm
        self._dims_registry = data_dims

        # ---- Initialize module ----
        self._module: BaseModule = _build_module(
            module=model_kwargs.get("module"),
            module_cls=model_kwargs.get("module_cls"),
            data_dims=self._dims_registry,
            module_kwargs=model_kwargs.get("module_kwargs"),
        )

        # ---- Resolve the shared specs ----
        # A single `ProtocolSpecs` (or `FlowSpecs`) instance is shared by both
        # the training and the inference protocol, so module, dtype, device, and
        # flow configuration are guaranteed identical between them.
        dtype = model_kwargs.get("dtype")
        device_id = model_kwargs.get("device_id")
        flow_kwargs = model_kwargs.get("flow_kwargs", None)
        is_flow = model_kwargs.get("is_flow", False)
        self._specs: ProtocolSpecs | FlowSpecs = _resolve_specs(
            self._module, device_id=device_id, dtype=dtype, flow_kwargs=flow_kwargs, is_flow=is_flow
        )

        # ----- Initialize protocols -----
        training_protocol = _build_protocol(
            self._specs,
            "training",
            protocol_cls=model_kwargs.get("training_protocol_cls"),
            protocol_id=model_kwargs.get("training_protocol_id"),
            protocol_kwargs=model_kwargs.get("training_protocol_kwargs"),
            allow_none=False,
        )

        match_fn = model_kwargs.get("match_fn")
        self._training_protocol: SupportsTraining = _get_matched_protocol(training_protocol, match_fn=match_fn)

        self._inference_protocol: SupportsInference = _build_protocol(
            self._specs,
            "inference",
            protocol_cls=model_kwargs.get("inference_protocol_cls"),
            protocol_id=model_kwargs.get("inference_protocol_id"),
            protocol_kwargs=model_kwargs.get("inference_protocol_kwargs"),
            allow_none=False,
        )

        # ----- Initialize additional attributes -----
        self._trainer: Trainer | None = None

    @overload
    def _predict_empty(
        self,
        return_raw: Literal[False],
    ) -> AnnData:
        pass

    @overload
    def _predict_empty(
        self,
        return_raw: Literal[True],
    ) -> tuple[AnnData, None]:
        pass

    @overload
    def _aggregate_nodes_pred(
        self,
        all_preds: list[PredictionData],
        all_obs: list[pd.DataFrame],
        all_obsm: dict[str, list[np.ndarray]],
        return_raw: Literal[False],
    ) -> AnnData:
        pass

    @overload
    def _aggregate_nodes_pred(
        self,
        all_preds: list[PredictionData],
        all_obs: list[pd.DataFrame],
        all_obsm: dict[str, list[np.ndarray]],
        return_raw: Literal[True],
    ) -> tuple[AnnData, PredictionData]:
        pass

    @overload
    def predict(
        self,
        adata: AnnData,
        *,
        return_raw: Literal[False] = ...,
        **kwargs,
    ) -> AnnData:
        pass

    @overload
    def predict(
        self,
        adata: AnnData,
        *,
        return_raw: Literal[True],
        **kwargs,
    ) -> tuple[AnnData, PredictionData]:
        pass

    def _predict_empty(self, return_raw: bool) -> AnnData | tuple[AnnData, None]:
        """Returns empty anndata for prediction."""
        empty_adata = AnnData(
            X=np.empty((0, len(self._dims_registry.feature_names))),
            var=pd.DataFrame(index=self._dims_registry.feature_names),
        )
        return empty_adata if not return_raw else (empty_adata, None)

    def _pred_obs_from_leaf(self, group_cols: tuple[str, ...], leaf: tuple, pred_obj: PredictionData) -> pd.DataFrame:
        """Rebuild a group's obs rows from its ``leaf`` (the ``group_by`` value tuple), one per predicted observation."""
        n_pred_obs = pred_obj.X.shape[0] if getattr(pred_obj, "X", None) is not None else 1
        return pd.DataFrame({col: np.repeat(val, n_pred_obs) for col, val in zip(group_cols, leaf, strict=True)})

    def _get_pred_traj(self, pred_obj: PredictionData) -> np.ndarray | None:
        if pred_obj.traj is None:
            return None

        n_obs = pred_obj.X.shape[0]
        traj_np = self._to_numpy(pred_obj.traj)

        if traj_np.ndim == 2 and traj_np.shape[0] == n_obs:
            return traj_np
        elif traj_np.ndim == 3 and traj_np.shape[1] == n_obs:
            return np.transpose(traj_np, (1, 0, 2))
        elif traj_np.ndim == 4 and traj_np.shape[2] == n_obs:
            return np.transpose(traj_np, (2, 0, 1, 3))
        else:
            raise ValueError(
                "Trajectory array has incompatible shape for AnnData.obsm: "
                f"got {traj_np.shape}, expected first dimension to equal "
                f"n_obs ({n_obs}) or, for 3D trajectories, second "
                "dimension to equal n_obs so it can be transposed from "
                "(n_time_steps, n_obs, n_features) to "
                "(n_obs, n_time_steps, n_features)."
            )

    def _get_pred_raw_samples(self, pred_obj: PredictionData) -> np.ndarray | None:
        raw_samples = getattr(pred_obj, "raw_samples", None)
        if raw_samples is None:
            return None

        X = getattr(pred_obj, "X", None)
        if X is None:
            raise ValueError("Prediction object should have the .X attribute.")
        n_obs = X.shape[0]

        samples_np = self._to_numpy(raw_samples)
        if samples_np.ndim == 2 and samples_np.shape[0] == n_obs:
            return samples_np
        elif samples_np.ndim == 3 and samples_np.shape[1] == n_obs:
            return np.transpose(samples_np, (1, 0, 2))
        else:
            raise ValueError(
                "Samples array has incompatible shape for AnnData.obsm: "
                f"got {samples_np.shape}, expected data of shape "
                f"(n_obs, n_features) or (n_samples, n_obs, n_features)"
            )

    def _get_pred_obsm_dict(
        self, step_data: StepData, pred_obj: PredictionData, cont_keys: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        obsm_dict: dict[str, np.ndarray] = {}
        traj = self._get_pred_traj(pred_obj)
        if traj is not None:
            obsm_dict["trajectory"] = traj
        raw_samples = self._get_pred_raw_samples(pred_obj)
        if raw_samples is not None:
            obsm_dict["raw_samples"] = raw_samples

        condition = step_data["target_condition_data"] or {}
        response = step_data["target_response_data"] or {}
        for key in cont_keys:
            if key in condition:
                obsm_dict[key] = self._to_numpy(condition[key])
            elif key in response:
                obsm_dict[key] = self._to_numpy(response[key])
        return obsm_dict

    def _aggregate_nodes_pred(
        self,
        all_preds: list[PredictionData],
        all_obs: list[pd.DataFrame],
        all_obsm: dict[str, list[np.ndarray]],
        return_raw: bool = False,
    ) -> AnnData | tuple[AnnData, PredictionData]:
        merged_pred = type(all_preds[0]).concatenate(all_preds)

        X_np = self._to_numpy(merged_pred.X)

        obs_final = pd.concat(all_obs, axis=0, ignore_index=True)
        obs_final.index = obs_final.index.astype(str)

        obsm_final = {k: np.concatenate(v, axis=0) for k, v in all_obsm.items()}

        pred_adata = AnnData(
            X=X_np, obs=obs_final, var=pd.DataFrame(index=self._dims_registry.feature_names), obsm=obsm_final
        )

        if return_raw:
            return pred_adata, merged_pred

        return pred_adata

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert a torch tensor (or array-like) to a numpy array."""
        if tensor is None:
            return None
        import torch

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def to_device(self, device: str) -> None:
        """Move the underlying PyTorch module and optimizer state to the specified device."""
        import torch

        self._module.to(device)
        if self._trainer is not None and hasattr(self._trainer, "opt_manager"):
            opt = self._trainer.opt_manager.optimizer
            for param_group in opt.param_groups:
                for param in param_group["params"]:
                    if param.device.type != device.split(":")[0]:
                        param.data = param.data.to(device)
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.device.type != device.split(":")[0]:
                        state[k] = v.to(device)

    def train(
        self,
        adata: AnnData,
        *,
        training_protocol_cls: type[SupportsTraining] | None = None,
        training_protocol_id: str | None = None,
        training_protocol_kwargs: dict[str, Any] | None = None,
        inference_protocol_cls: type[SupportsInference] | None = None,
        inference_protocol_id: str | None = None,
        inference_protocol_kwargs: dict[str, Any] | None = None,
        match_fn: TMatchFn | None = None,
        train_split: str = "train",
        control_adata: AnnData | None = None,
        callbacks: TrainingCallbacks | Sequence[BaseCallback] | None = None,
        n_train_steps: int = 100_000,
        valid_freq: int = 1_000,
        pbar_freq: int = 100,
        batch_size: int = 128,
        loader_kwargs: LoaderKwargs | None = None,
        optim_config: OptimConfig | None = None,
        compute_loss_kwargs: dict[str, Any] | None = None,
        val_predict_kwargs: dict[str, Any] | None = None,
        cb_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Trains the model by streaming ``StepData`` batches from scfit-backed loaders.

        :param adata: The annotated data object to stream.
        :param train_split: The split value whose loader drives optimization. Defaults to ``"train"``.
        :param control_adata: Optional separate control (source) pool, shared by every split.
        :param callbacks: Callbacks used during training.
        :param n_train_steps: Number of training steps (streamed batches). Defaults to ``100_000``.
        :param valid_freq: Frequency (in steps) of validation passes. Defaults to ``1_000``.
        :param pbar_freq: Frequency (in steps) of progress-bar refreshes. Defaults to ``100``.
        :param batch_size: Observations per streamed batch. Defaults to ``128``.
        :param loader_kwargs: Options for each streaming loader.
        :param optim_config: Optimizer / scheduler configuration.
        :param training_protocol_cls: Overrides the model's training protocol for this call only.
        :param training_protocol_id: Identifier of a registered training protocol.
        :param training_protocol_kwargs: Keyword arguments forwarded to the per-call training protocol.
        :param inference_protocol_cls: Overrides the model's inference protocol for this call's validation only.
        :param inference_protocol_id: Identifier of a registered inference protocol.
        :param inference_protocol_kwargs: Keyword arguments forwarded to the per-call inference protocol.
        :param match_fn: Overrides the construction-time matcher for this call only. When `None`,
            the instance's training protocol (matched or not) is used as-is.
        :param compute_loss_kwargs: Forwarded to the training protocol's ``compute_loss``.
        :param val_predict_kwargs: Forwarded to the inference protocol's ``predict`` during validation.
        :param cb_kwargs: Forwarded to every callback hook.
        """
        # get training protocol
        training_protocol = _build_protocol(
            self._specs,
            "training",
            protocol_cls=training_protocol_cls,
            protocol_id=training_protocol_id,
            protocol_kwargs=training_protocol_kwargs,
            allow_none=True,
        )
        if training_protocol is None:
            training_protocol = self._training_protocol
        training_protocol: SupportsTraining = _get_matched_protocol(training_protocol, match_fn=match_fn)

        # get inference protocol
        inference_protocol = _build_protocol(
            self._specs,
            "inference",
            protocol_cls=inference_protocol_cls,
            protocol_id=inference_protocol_id,
            protocol_kwargs=inference_protocol_kwargs,
            allow_none=True,
        )
        if inference_protocol is None:
            inference_protocol = self._inference_protocol

        # Build one streaming loader per split.
        loader_kwargs = dict(loader_kwargs or {})
        loader_kwargs.setdefault("to", None)
        loader_kwargs.setdefault("batch_size", batch_size)
        loader_kwargs.setdefault("dtype", training_protocol.dtype)
        loader_kwargs.setdefault("device", training_protocol.device_id)
        loaders = self._dm.get_dataloaders(adata, control_adata=control_adata, **loader_kwargs)
        if train_split not in loaders:
            raise KeyError(
                f"train split {train_split!r} has no loader; available splits: {list(loaders)}. "
                "(A split with only control groups produces no loader.)"
            )

        train_loader = loaders[train_split].set_n_iters(n_train_steps)
        val_loaders = {split: loader for split, loader in loaders.items() if split != train_split}

        opt_manager = OptimizationManager.from_config(self._module, optim_config or OptimConfig())

        self._trainer = Trainer(
            training_protocol, opt_manager, inference_protocol=inference_protocol, callbacks=callbacks
        )

        training_protocol.set_train_mode(True)

        self._trainer.train(
            train_loader,
            val_loaders=val_loaders or None,
            valid_freq=valid_freq,
            pbar_freq=pbar_freq,
            compute_loss_kwargs=compute_loss_kwargs,
            val_predict_kwargs=val_predict_kwargs,
            cb_kwargs=cb_kwargs,
        )

    def predict(
        self,
        adata: AnnData,
        *,
        inference_protocol_cls: type[SupportsInference] | None = None,
        inference_protocol_id: str | None = None,
        inference_protocol_kwargs: dict[str, Any] | None = None,
        return_raw: bool = False,
        max_per_group: int | None = None,
        require_target_state: bool = True,
        control_values_dict: dict[str, str] | None = None,
        matched_keys: Mapping[tuple, tuple] | None = None,
        control_adata: AnnData | None = None,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> AnnData | tuple[AnnData, PredictionData]:
        """Generate flow predictions, one deterministic pass per group via :class:`EvalLoader`.

        :param adata: The input adata containing the metadata for prediction.
        :param inference_protocol_cls: Overrides the model's inference protocol for this call only.
        :param inference_protocol_id: Identifier of a registered inference protocol.
        :param inference_protocol_kwargs: Keyword arguments forwarded to the per-call inference protocol.
        :param return_raw: If ``True``, also return the raw concatenated ``PredictionData``.
        :param max_per_group: Per-group cap on observations.
        :param require_target_state: Whether ``adata`` must carry a target state representation.
        :param control_values_dict: Optional mapping from each condition level to its control value.
        :param matched_keys: Optional ``{source group key: target group key}`` pairs for fixed matching.
        :param control_adata: Optional separate control (source) pool.
        :param predict_kwargs: Forwarded to the inference protocol's ``predict``.
        :return: An AnnData with predictions, or a tuple ``(AnnData, PredictionData)`` if ``return_raw``.
        """
        inference_protocol = _build_protocol(
            self._specs,
            "inference",
            protocol_cls=inference_protocol_cls,
            protocol_id=inference_protocol_id,
            protocol_kwargs=inference_protocol_kwargs,
            allow_none=True,
        )
        if inference_protocol is None:
            inference_protocol = self._inference_protocol

        inference_protocol.set_train_mode(False)
        predict_kwargs = {} if predict_kwargs is None else predict_kwargs

        eval_loader = self._dm.get_eval_loader(
            adata,
            max_per_group=max_per_group,
            require_target_state=require_target_state,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
            control_adata=control_adata,
            to=None,
            dtype=inference_protocol.dtype,
            device=inference_protocol.device_id,
        )

        if len(eval_loader) == 0:
            return self._predict_empty(return_raw)

        all_preds = []
        all_obs = []
        all_obsm = defaultdict(list)
        group_cols = eval_loader.group_cols
        cont_keys = (*eval_loader.cond_cont_keys, *eval_loader.resp_keys)

        for step_data, leaf in tqdm(eval_loader, total=len(eval_loader), desc="Predicting"):
            pred_obj = inference_protocol.predict(step_data, **predict_kwargs)
            all_preds.append(pred_obj)

            all_obs.append(self._pred_obs_from_leaf(group_cols, leaf, pred_obj))

            node_obsm_dict = self._get_pred_obsm_dict(step_data, pred_obj, cont_keys)
            for key, val in node_obsm_dict.items():
                all_obsm[key].append(val)

        return self._aggregate_nodes_pred(all_preds, all_obs, all_obsm, return_raw=return_raw)

    def save(self, filepath: str, allow_overwrite: bool = False) -> None:
        """Save the entire model (including registered data) to a tarball."""
        path = Path(filepath)
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"{filepath} already exists. Use allow_overwrite=True.")
        elif path.exists() and allow_overwrite:
            path.unlink()

        import torch

        self._module.cpu()
        if self._trainer is not None and hasattr(self._trainer, "opt_manager"):
            opt = self._trainer.opt_manager.optimizer
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

        with tarfile.open(filepath, "w:gz") as tar:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                cloudpickle.dump(self, tmp)
                tmp.flush()
                tar.add(tmp.name, arcname="model.pkl")
            Path(tmp.name).unlink()

        logging.info(f"Model saved to {filepath} (moved to CPU).")

    @classmethod
    def load(
        cls,
        filepath: str,
        adata: AnnData | None = None,
        map_location: str | None = None,
        **register_kwargs,
    ) -> Model:
        """Load a saved model from a tarball."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"{filepath} not found.")

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir).resolve()
            with tarfile.open(filepath, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.issym() or member.islnk():
                        raise ValueError(f"Refusing to extract link from archive: {member.name}")

                    member_path = (extract_dir / member.name).resolve()
                    try:
                        member_path.relative_to(extract_dir)
                    except ValueError as e:
                        raise ValueError(
                            f"Refusing to extract archive member outside target directory: {member.name}"
                        ) from e

                    tar.extract(member, tmpdir)

            with open(Path(tmpdir) / "model.pkl", "rb") as f:
                model = cloudpickle.load(f)

        if adata is not None:
            builder = ModelBuilder.from_adata(adata, **register_kwargs)
            model._dm = builder.dm
            model._dims_registry = builder.data_dims

        if map_location is not None:
            model.to_device(map_location)

        return model

    @property
    def dm(self) -> DataManager:
        """Returns the data manager associated to the current instance."""
        return self._dm

    @property
    def is_paired_setting(self) -> bool:
        """Whether the data was registered in a paired setting."""
        return self._dm.control_values_dict is not None or self._dm.matched_keys is not None

    @property
    def module(self) -> BaseModule:
        """Returns the underlying module."""
        return self._module

    @property
    def specs(self) -> ProtocolSpecs:
        """The shared `ProtocolSpecs` (or `FlowSpecs`) instance.

        Both the training and the inference protocol hold this same instance,
        so `specs.probability_path`, `specs.time_sampler`, etc. reflect what
        both protocols see.
        """
        return self._specs

    @property
    def training_protocol(self) -> SupportsTraining:
        """The underlying training protocol (may be a `MatchedTrainingProtocol` wrapper)."""
        return self._training_protocol

    @property
    def inference_protocol(self) -> SupportsInference:
        """The underlying inference protocol."""
        return self._inference_protocol

    @property
    def trainer(self) -> Trainer | None:
        """Returns the trainer used to fit the model."""
        return self._trainer

    @property
    def condition_state_key(self) -> str | None:
        """Return the key used to extract the state from the condition."""
        return self._dm.condition_state_key
