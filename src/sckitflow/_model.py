from __future__ import annotations

import logging
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Unpack, overload

import cloudpickle
import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from sckitflow._types import PredictionData
from sckitflow.core._types import StepData
from sckitflow.core.methods._base import BaseMethod
from sckitflow.core.methods._opt import OptimConfig, OptimizationManager
from sckitflow.data._dims_registry import DataDimensionalitiesRegistry
from sckitflow.data._manager import DataManager, DataManagerKwargs
from sckitflow.trainer._callbacks import BaseCallback, TrainingCallbacks
from sckitflow.trainer._trainer import Trainer

if TYPE_CHECKING:
    # Typing only: importing `_loader` eagerly would pull scfit into `import sckitflow`.
    from sckitflow.data._loader import LoaderKwargs

__all__ = ["Model", "ModelBuilder"]


class ModelBuilder:
    """Two-step builder for a :class:`Model`.

    Step one (:meth:`from_adata`) prepares the *data* side: it initializes the
    :class:`DataManager` from the schema keyword arguments and derives the data
    dimensionalities. Step two (:meth:`build`) attaches the *method* and returns
    a ready-to-train :class:`Model`. Keeping the two concerns in separate calls
    avoids mixing data-schema configuration with method/module configuration, and
    lets you inspect :attr:`data_dims` before choosing method parameters.

    Any preprocessing of the state representation (e.g. PCA, normalization) must
    be done by the caller *before* :meth:`from_adata`, and the resulting
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

        Initializes the :class:`DataManager` from ``dm_kwargs`` and derives the
        data dimensionalities from ``adata``.

        :param adata: The annotated data object used to fit the schema. Any
            preprocessing of the state representation must already have been
            applied by the caller.
        :type adata: class: `AnnData`

        :param dm_kwargs: Keyword arguments forwarded to :class:`DataManager`.
            See :class:`DataManagerKwargs` for the accepted options.
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

    def build(
        self,
        *args,
        method: BaseMethod | None = None,
        method_cls: type[BaseMethod] | None = None,
        method_id: str | None = None,
        **kwargs,
    ) -> Model:
        """Attach a method and return a ready-to-train :class:`Model`.

        Either pass an already-constructed ``method`` instance, or select a
        method via ``method_cls`` / ``method_id`` and let it be constructed
        from the prepared data manager and dimensionalities using ``args`` /
        ``kwargs``.

        :param method: A pre-built method instance. When provided,
            ``method_cls`` / ``method_id`` and any extra ``args`` / ``kwargs``
            are ignored. Defaults to `None`.
        :type method: class: `BaseMethod | None`

        :param method_cls: The method class to instantiate. Mutually exclusive
            with ``method_id``. Defaults to `None`.
        :type method_cls: class: `type[BaseMethod] | None`

        :param method_id: Identifier of a registered method to instantiate.
            Mutually exclusive with ``method_cls``. Defaults to `None`.
        :type method_id: class: `str | None`

        :param args: Extra positional arguments forwarded to the method.
        :param kwargs: Extra keyword arguments forwarded to the method.
        """
        return Model(
            self._dm,
            self._data_dims,
            *args,
            method=method,
            method_cls=method_cls,
            method_id=method_id,
            **kwargs,
        )


class Model:
    def __init__(
        self,
        dm: DataManager,
        data_dims: DataDimensionalitiesRegistry,
        *args,
        method: BaseMethod | None = None,
        method_cls: type[BaseMethod] | None = None,
        method_id: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize a model from a fitted data manager and its dimensionalities.

        Usually constructed through :class:`ModelBuilder` rather than directly.

        :param dm: The fitted data manager describing the data schema.
        :type dm: class: `DataManager`

        :param data_dims: The data dimensionalities derived from the registration
            data (see :meth:`DataManager.get_data_dimensionalities`).
        :type data_dims: class: `DataDimensionalitiesRegistry`

        :param method: A pre-built method instance. When provided,
            ``method_cls`` / ``method_id`` and any extra ``args`` / ``kwargs``
            are ignored. Defaults to `None`.
        :type method: class: `BaseMethod | None`

        :param method_cls: The method class to instantiate. Mutually exclusive
            with ``method_id``. Defaults to `None`.
        :type method_cls: class: `type[BaseMethod] | None`

        :param method_id: Identifier of a registered method to instantiate.
            Mutually exclusive with ``method_cls``. Defaults to `None`.
        :type method_id: class: `str | None`

        :param args: Extra positional arguments forwarded to the method.
        :param kwargs: Extra keyword arguments forwarded to the method.
        """
        # store data manager and dimensionalities
        self._dm = dm
        self._dims_registry = data_dims

        # use the provided method instance when given
        if method is not None:
            self._method: BaseMethod = method
        else:
            # get method cls
            if method_cls is None and method_id is None:
                msg = "At least one of `method`, `method_id` or `method_cls` should be specified."
                raise ValueError(msg)

            # use registry when method not provided
            if method_cls is None:
                from sckitflow.core.methods import METHODS_REGISTRY

                # get method from registry
                if method_id not in METHODS_REGISTRY:
                    msg = f"Method {method_id} not supported, possible options are {list(METHODS_REGISTRY.keys())}."
                    raise KeyError(msg)
                method_cls = METHODS_REGISTRY[method_id]

            # initialize method
            self._method = method_cls(
                self._dims_registry,
                self._dm,
                *args,
                **kwargs,
            )

        # prepare attributes
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
        """Rebuild a group's obs rows from its ``leaf`` (the ``group_by`` value tuple), one per predicted cell.

        The group identity is the ``leaf`` surfaced by :class:`~sckitflow.data._loader.EvalLoader` -- a tuple
        of the categorical group/condition values, ordered as ``group_cols`` -- so no ``ann_df`` round-trip is
        needed. Each value is repeated to match the number of predicted observations.
        """
        n_pred_obs = pred_obj.X.shape[0] if getattr(pred_obj, "X", None) is not None else 1
        return pd.DataFrame({col: np.repeat(val, n_pred_obs) for col, val in zip(group_cols, leaf, strict=True)})

    def _get_pred_traj(self, pred_obj: PredictionData) -> np.ndarray | None:
        # early return if no trajectory
        if pred_obj.traj is None:
            return None

        # get number of observations
        n_obs = pred_obj.X.shape[0]

        # convert trajectory to numpy
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
                "(n_time_steps, n_cells, n_features) to "
                "(n_cells, n_time_steps, n_features)."
            )

    def _get_pred_raw_samples(self, pred_obj: PredictionData) -> np.ndarray | None:
        # ---- Early return if no raw samples present ----
        raw_samples = getattr(pred_obj, "raw_samples", None)
        if raw_samples is None:
            return None

        # ---- Get number of observations from X ----
        X = getattr(pred_obj, "X", None)
        if X is None:
            raise ValueError("Prediction object should have the .X attribute.")
        n_obs = X.shape[0]

        # ---- Convert raw samples to numpy and handle shape ----

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
        # ---- Trajectory and raw samples ----
        obsm_dict: dict[str, np.ndarray] = {}
        traj = self._get_pred_traj(pred_obj)
        if traj is not None:
            obsm_dict["trajectory"] = traj
        raw_samples = self._get_pred_raw_samples(pred_obj)
        if raw_samples is not None:
            obsm_dict["raw_samples"] = raw_samples

        # ---- Continuous condition/response covariates: per-cell reps carried in the StepData dicts ----
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
        # ---- Aggregate predicted states ----
        # Merge predictions using backend‑specific concatenation
        merged_pred = type(all_preds[0]).concatenate(all_preds)

        # ---- Construct prediction adata ----
        # Convert to numpy
        X_np = self._to_numpy(merged_pred.X)

        # Aggregate obs dataframes (fresh unique string index -- groups each carry a 0..n range)
        obs_final = pd.concat(all_obs, axis=0, ignore_index=True)
        obs_final.index = obs_final.index.astype(str)

        # Aggregate obsm array dictionary
        obsm_final = {k: np.concatenate(v, axis=0) for k, v in all_obsm.items()}

        pred_adata = AnnData(
            X=X_np, obs=obs_final, var=pd.DataFrame(index=self._dims_registry.feature_names), obsm=obsm_final
        )

        # ---- Return output ----
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

        self._method._module.to(device)
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
        split_by: str | None = "split",
        train_split: str = "train",
        control_adata: AnnData | None = None,
        callbacks: TrainingCallbacks | Sequence[BaseCallback] | None = None,
        n_train_steps: int = 100_000,
        valid_freq: int = 1_000,
        pbar_freq: int = 100,
        batch_size: int = 128,
        loader_kwargs: LoaderKwargs | None = None,
        optim_config: OptimConfig | None = None,
        train_step_kwargs: dict[str, Any] | None = None,
        val_predict_kwargs: dict[str, Any] | None = None,
        cb_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Trains the model by streaming ``StepData`` batches from scfit-backed loaders.

        One loader is built per split value of ``adata.obs[split_by]`` via
        :meth:`DataManager.get_dataloaders`. The ``train_split`` loader drives the optimizer -- one
        gradient step per streamed batch, cycling epochs -- and every other split becomes a
        validation loader (iterated for one epoch per validation). Selection is by scfit weights over
        the whole ``adata`` (no subset copying), and controls are shared across splits (see
        :meth:`DataManager.get_dataloaders`).

        :param adata: The annotated data object to stream; must carry ``split_by`` in ``.obs``.
        :type adata: class: `AnnData`

        :param split_by: The ``.obs`` column whose values define the splits. Defaults to ``"split"``.
        :type split_by: class: `str`

        :param train_split: The split value whose loader drives optimization. Defaults to ``"train"``.
        :type train_split: class: `str`

        :param control_adata: Optional separate control (source) pool, shared by every split.
        :type control_adata: class: `AnnData | None`

        :param callbacks: Callbacks to be used during training.
        :type callbacks: class: `TrainingCallbacks | Sequence[BaseCallback] | None`

        :param n_train_steps: The number of training steps (streamed batches) to train over.
            Defaults to ``100_000``.
        :type n_train_steps: class: `int`

        :param valid_freq: The frequency (in steps) of the validation passes during training.
            Defaults to ``1_000``.
        :type valid_freq: class: `int`

        :param pbar_freq: The frequency (in steps) of progress-bar description refreshes.
            Defaults to ``100``.
        :type pbar_freq: class: `int`

        :param batch_size: Number of observations per streamed batch. Defaults to ``128``.
        :type batch_size: class: `int`

        :param loader_kwargs: Options for each streaming loader; see :class:`~sckitflow.data.LoaderKwargs`.
            Defaults to ``None``.
        :type loader_kwargs: class: `LoaderKwargs | None`

        :param optim_config: The optimizer / scheduler configuration. Defaults to ``None`` (an
            :class:`~sckitflow.core.methods._opt.OptimConfig` with its own defaults).
        :type optim_config: class: `OptimConfig | None`

        :param train_step_kwargs: Forwarded to :meth:`BaseMethod.train_step` (method-specific).
        :type train_step_kwargs: class: `dict[str, Any] | None`

        :param val_predict_kwargs: Forwarded to :meth:`BaseMethod.predict` during validation -- e.g.
            CFM's ``n_samples``, required when the method generates from noise.
        :type val_predict_kwargs: class: `dict[str, Any] | None`

        :param cb_kwargs: Forwarded to every callback hook.
        :type cb_kwargs: class: `dict[str, Any] | None`
        """
        # Build one streaming loader per split (scfit weights over the whole adata -- no copies).
        loader_kwargs = dict(loader_kwargs or {})
        # `to=None`: keep annbatch's native arrays (numpy on host, cupy on a GPU-resident window) and let
        # the loader map them onto torch itself, without a copy or a device round-trip.
        loader_kwargs.setdefault("to", None)
        loader_kwargs.setdefault("batch_size", batch_size)
        # The loader settles dtype/device as its last stage, so batches reach the method ready to consume.
        loader_kwargs.setdefault("dtype", self._method.dtype)
        loader_kwargs.setdefault("device", self._method.device_id)
        loaders = self._dm.get_dataloaders(adata, split_by=split_by, control_adata=control_adata, **loader_kwargs)
        if train_split not in loaders:
            raise KeyError(
                f"train split {train_split!r} has no loader; available splits: {list(loaders)}. "
                "(A split with only control groups produces no loader.)"
            )

        # Size the train loader to the number of training steps (the train func sets the length); the
        # trainer then just iterates it. The remaining splits are one-pass validation loaders.
        train_loader = loaders[train_split].set_n_iters(n_train_steps)
        val_loaders = {split: loader for split, loader in loaders.items() if split != train_split}

        # prepare optimization manager
        opt_manager = OptimizationManager.from_config(self._method._module, optim_config or OptimConfig())

        # initialize trainer (once; later calls continue from the current step)
        if self._trainer is None:
            self._trainer = Trainer(self._method, opt_manager, callbacks)

        # module in training mode
        self._method.set_train_mode(True)

        # train model
        self._trainer.train(
            train_loader,
            val_loaders=val_loaders or None,
            valid_freq=valid_freq,
            pbar_freq=pbar_freq,
            train_step_kwargs=train_step_kwargs,
            val_predict_kwargs=val_predict_kwargs,
            cb_kwargs=cb_kwargs,
        )

    def predict(
        self,
        adata: AnnData,
        *,
        return_raw: bool = False,
        max_per_group: int | None = None,
        require_target_state: bool = True,
        control_values_dict: dict[str, str] | None = None,
        matched_keys: Mapping[tuple, tuple] | None = None,
        control_adata: AnnData | None = None,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> AnnData | tuple[AnnData, PredictionData]:
        """Generate flow predictions, one deterministic pass per group via :class:`EvalLoader`.

        Every perturbed group is predicted once (or capped by ``max_per_group``), each matched to its
        control leaf; the output ``obs`` is rebuilt from each group's ``leaf`` (its ``group_by`` values).

        :param adata: The input adata containing the metadata for prediction. When
            ``require_target_state`` is ``False``, this only needs ``.obs`` (and optionally ``.obsm`` for
            continuous conditioning) describing the groups to predict for -- no ``.X`` / expression
            ``obsm`` key required.
        :type adata: class: `AnnData`

        :param return_raw: If ``True``, also return the raw concatenated ``PredictionData`` (keeping the
            computation graph alive). Defaults to ``False``.
        :type return_raw: class: `bool`

        :param max_per_group: Per-group cap on cells: ``None`` = every cell, ``N`` = at most N, ``1`` =
            predict once per condition (dedup / metadata-only). Defaults to ``None``.
        :type max_per_group: class: `int | None`

        :param require_target_state: Whether ``adata`` must carry a target state representation (``.X`` or
            the configured ``obsm`` sample representation). Set to ``False`` to predict purely from the
            conditioning metadata. Defaults to ``True``.
        :type require_target_state: class: `bool`

        :param control_values_dict: Optional mapping from each condition level to its control value,
            overriding the instance's for this call (to allow inference over arbitrary control keys).
            Defaults to ``None`` (use the instance's); pass ``{}`` to predict unpaired.
        :type control_values_dict: class: `dict[str, str] | None`

        :param matched_keys: Optional ``{source group key: target group key}`` pairs for fixed matching,
            overriding the instance's for this call (to allow inference over arbitrary pairs). Defaults to
            ``None`` (use the instance's).
        :type matched_keys: class: `Mapping[tuple, tuple] | None`

        :param control_adata: Optional separate control (source) pool, matched on the group columns.
        :type control_adata: class: `AnnData | None`

        :param predict_kwargs: Forwarded to :meth:`BaseMethod.predict` (method-specific) -- e.g. CFM's
            ``n_samples`` / ``n_steps`` / ``return_trajectory``. Defaults to ``None``.
        :type predict_kwargs: class: `dict[str, Any] | None`

        :return: Either an AnnData with predictions, or a tuple ``(AnnData, PredictionData)`` if
            ``return_raw`` is ``True``.
        """
        # Set module to evaluation mode (backend-agnostic)
        self._method.set_train_mode(False)
        predict_kwargs = {} if predict_kwargs is None else predict_kwargs

        eval_loader = self._dm.get_eval_loader(
            adata,
            max_per_group=max_per_group,
            require_target_state=require_target_state,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
            control_adata=control_adata,
            to=None,  # native arrays in, zero-copy to torch in the loader (see `Model.train`)
            dtype=self._method.dtype,
            device=self._method.device_id,
        )

        # early return when nothing to predict
        if len(eval_loader) == 0:
            return self._predict_empty(return_raw)

        # define store
        all_preds = []
        all_obs = []
        all_obsm = defaultdict(list)
        group_cols = eval_loader.group_cols
        cont_keys = (*eval_loader.cond_cont_keys, *eval_loader.resp_keys)

        # Iterate one ready `StepData` per group (with its `leaf` group identity)
        for step_data, leaf in tqdm(eval_loader, total=len(eval_loader), desc="Predicting"):
            # 1. Inference
            pred_obj = self._method.predict(step_data, **predict_kwargs)
            all_preds.append(pred_obj)

            # 2. Construct obs from the group's leaf (no ann_df / container round-trip)
            all_obs.append(self._pred_obs_from_leaf(group_cols, leaf, pred_obj))

            # 3. Construct obsm (continuous covariates ride per-cell in the StepData dicts)
            node_obsm_dict = self._get_pred_obsm_dict(step_data, pred_obj, cont_keys)
            for key, val in node_obsm_dict.items():
                all_obsm[key].append(val)

        return self._aggregate_nodes_pred(all_preds, all_obs, all_obsm, return_raw=return_raw)

    def save(self, filepath: str, allow_overwrite: bool = False) -> None:
        """
        Save the entire model (including registered data) to a tarball.

        :param filepath: Output file path (e.g., 'model.tar.gz').
        :param allow_overwrite: If True, overwrite existing file.
        """
        path = Path(filepath)
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"{filepath} already exists. Use allow_overwrite=True.")
        elif path.exists() and allow_overwrite:
            path.unlink()

        # Move model to CPU before pickling
        import torch

        self._method._module.cpu()
        # Fixed: access optimizer via trainer, not via method
        if self._trainer is not None and hasattr(self._trainer, "opt_manager"):
            opt = self._trainer.opt_manager.optimizer
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

        # Save self as a tarball containing a single pickle file
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
        """
        Load a saved model from a tarball.

        :param filepath: Path to the saved tarball.
        :param adata: Optional AnnData used to rebuild the data manager and
            dimensionalities when the saved model should be re-registered.
        :param map_location: For PyTorch models, map to a device (e.g., 'cuda:0').
        :param register_kwargs: Additional keyword arguments forwarded to
            :class:`DataManager` when ``adata`` is provided.
        :return: Loaded Model instance.
        """
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

        # If an AnnData is provided, rebuild the data manager and dimensionalities
        # from it (overwrites the saved ones).
        if adata is not None:
            builder = ModelBuilder.from_adata(adata, **register_kwargs)
            model._dm = builder.dm
            model._dims_registry = builder.data_dims

        # Move to desired device if requested
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
    def method(self) -> BaseMethod:
        """Returns the underlying method."""
        return self._method

    @property
    def trainer(self) -> Trainer:
        """Returns the trainer used to fit the model."""
        return self._trainer

    @property
    def condition_state_key(self) -> str | None:
        """Return the key used to extract the state from the condition."""
        return self._dm.condition_state_key
