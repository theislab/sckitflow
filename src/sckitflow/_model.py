import logging
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Unpack, overload

import cloudpickle
import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from sckitflow._types import PredictionData
from sckitflow.core._data_utils import extract_step_data
from sckitflow.core._types import StepData
from sckitflow.core.methods._base import BaseMethod
from sckitflow.data.containers._distribution import build_ann_df
from sckitflow.core.methods._opt import OptimConfig
from sckitflow.data._composite import MatchedData
from sckitflow.data._dims_registry import DataDimensionalitiesRegistry
from sckitflow.data._manager import DataManager, DataManagerKwargs
from sckitflow.data.samplers._train import FTrainSampler
from sckitflow.data.samplers._validation import FValidationSampler
from sckitflow.trainer._callbacks import BaseCallback, TrainingCallbacks
from sckitflow.trainer._trainer import Trainer

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
    ) -> "ModelBuilder":
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
    ) -> "Model":
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
        *args,
        return_raw: Literal[False],
        sort: bool = True,
        **kwargs,
    ) -> AnnData:
        pass

    @overload
    def predict(
        self,
        adata: AnnData,
        *args,
        return_raw: Literal[True],
        sort: bool = True,
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

    def _get_pred_obs_df(self, step_data: StepData, pred_obj: PredictionData) -> pd.DataFrame:
        """Gets the observation dataframe for prediction"""
        # 1. Collect observation metadata (rebuilt from the target condition/group data)
        ann_df = build_ann_df(step_data["target_condition_data"], step_data["target_group_data"]).copy()
        cond_df = ann_df.drop_duplicates()

        # 2. Check that the node contains only one condition
        #    or that the are actually columns inside
        n_ann_cols = len(cond_df.columns)
        if n_ann_cols:
            n_unique_conds = cond_df.shape[0]
            if n_unique_conds != 1:
                msg = f"Node should contain unique condition, {n_unique_conds} found."
                raise ValueError(msg)

        # 3. Get number of generated observations
        if hasattr(pred_obj, "X"):
            n_pred_obs = pred_obj.X.shape[0]
        else:
            n_pred_obs = 1

        # 4. Align dataframe and update
        # only when there are columns we need to repeat, otherwise keep as is
        if n_ann_cols:
            df_data = np.repeat(cond_df, n_pred_obs, axis=0)
        else:
            df_data = cond_df
        return pd.DataFrame(df_data, columns=ann_df.columns)

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

    def _get_pred_obsm_dict(self, step_data: StepData, pred_obj: PredictionData) -> dict[str, np.ndarray]:
        # ---- Get trajectory and samples ----
        traj = self._get_pred_traj(pred_obj)
        raw_samples = self._get_pred_raw_samples(pred_obj)

        # ---- Get condition continuous covariates data ----
        condition_data = step_data["target_condition_data"]
        if condition_data is not None and condition_data.has_continuous_covariates:
            condition_continuous_covs = condition_data.continuous_covariates.mapping
        else:
            condition_continuous_covs = {}

        # ---- Get target (response) continuous covariates data ----
        response_data = step_data["target_response_data"]
        if response_data is not None and response_data.has_continuous_covariates:
            response_continuous_covs = response_data.continuous_covariates.mapping
        else:
            response_continuous_covs = {}

        # ---- Construct output ----
        obsm_dict = {}
        if traj is not None:
            obsm_dict["trajectory"] = traj
        if raw_samples is not None:
            obsm_dict["raw_samples"] = raw_samples
        obsm_dict.update(condition_continuous_covs)
        obsm_dict.update(response_continuous_covs)
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

        # Aggregate obs dataframes
        obs_final = pd.concat(all_obs, axis=0)

        # Aggregate obsm array dictionary
        obsm_final = {k: np.concatenate(v, axis=0) for k, v in all_obsm.items()}

        pred_adata = AnnData(
            X=X_np, obs=obs_final, var=pd.DataFrame(index=self._dims_registry.feature_names), obsm=obsm_final
        )

        # ---- Return output ----
        if return_raw:
            return pred_adata, merged_pred

        return pred_adata

    def _node_to_step_data(self, node: MatchedData) -> StepData:
        """Aligns a node and extracts ready :class:`StepData`.

        This is the single boundary where a :class:`MatchedData` node becomes
        :class:`StepData`. Alignment is a node-level operation, so it lives here rather
        than downstream; everything after this point consumes only ``StepData``. Replace
        this with a ``StepData``-native loader to drop the node pipeline from prediction.
        """
        node_aligned = node.align()
        return extract_step_data(node_aligned, device=self._method.device_id, dtype=self._method.dtype)

    def _predict_on_node(self, step_data: StepData, *args, **kwargs) -> PredictionData:
        """Runs inference on a ready :class:`StepData` batch.

        A thin pass-through: the node has already been aligned and turned into
        :class:`StepData` by :meth:`_node_to_step_data`.
        """
        return self._method.predict(step_data, *args, **kwargs)

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
        train_adata: AnnData,
        *args,
        val_adatas_dict: dict[str, AnnData] | None = None,
        callbacks: TrainingCallbacks | Sequence[BaseCallback] | None = None,
        n_train_steps: int = 100_000,
        valid_freq: int = 1_000,
        train_batch_size: int = 128,
        val_max_n_obs: int = 10_000,
        train_sampler_kwargs: dict[str, Any] | None = None,
        val_sampler_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        optim_kwargs: dict[str, Any] | None = None,
        sort: bool = False,
        **kwargs,
    ) -> None:
        """Trains the model on the input adata.

        :param train_adata: The train adata.
        :type train_adata: class: `AnnData`

        :param *args: Positional arguments used to call the `.train` method of the trainer class.
        :type *args: class: `Sequence[Any]`

        :param val_adatas_dict: Dictionary containing the validation adatas.
        :type val_adatas_dict: class: `dict[str, AnnData]`

        :param callbacks: Callbacks to be used during training.
        :type callbacks: class: `TrainingCallbacks | Sequence[BaseCallback] | None`

        :param n_train_steps: The number of training steps to train the model over.
            Defaults to `10_000`
        :type n_train_steps: class: `int`

        :param valid_freq: The frequency of the validation steps during training.
            Defaults to `1_000`
        :type valid_freq: class: `int`

        :param train_batch_size: The number of observations to sample for each node in a batch
            of training data. Defaults to `128`.
        :type train_batch_size: class: `int`

        :param val_max_n_obs: The maximum number of observations to sample for each node in a batch
            of validation data. Defaults to `10_000`.
        :type val_max_n_obs: class: `int`

        :param train_sampler_kwargs: Extra keyword arguments for the training sampler. Defaults to `None`.
        :type train_sampler_kwargs: class: `dict[str, Any] | None`

        :param val_sampler_kwargs: Extra keyword arguments for the validation sampler. Defaults to `None`.
        :type val_sampler_kwargs: class: `dict[str, Any] | None`

        :param optim_kwargs: Keyword arguments to configure for the optimization manager (optimizer, scheduler, etc.). Defaults to `None`.
        :type optim_kwargs: class: `OptimConfig | None`

        :param sort: If ``True``, create a sorted copy of *adata* via
            :meth:`sort_adata` and compile from that copy.  When setting this one
            should keep in mind this will copy the full adata object. When ``False`` (the
            default) the data must already be sorted; a ``ValueError``
            is raised otherwise.
        :type sort: class: `bool`

        :param *kwargs: Keyword arguments used to call the `.train` method of the trainer class.
        :type *kwargs: class: `dict[str, Any]`
        """
        # compile adata
        train_tree = self._dm.compile_adata(
            train_adata,
            sort=sort,
        )
        if val_adatas_dict is not None:
            val_trees_dict = {
                val_id: self._dm.compile_adata(
                    val_adata,
                    sort=sort,
                )
                for val_id, val_adata in val_adatas_dict.items()
            }
        else:
            val_trees_dict = {}

        # create train sampler
        if train_sampler_kwargs is None:
            train_sampler_kwargs = {}
        train_sampler = FTrainSampler(
            train_tree,
            batch_size=train_batch_size,
            **train_sampler_kwargs,
        )

        # create validation samplers
        if val_sampler_kwargs is None:
            val_sampler_kwargs = {}
        val_samplers_dict = {
            val_id: FValidationSampler(val_tree, max_n_obs=val_max_n_obs, **val_sampler_kwargs)
            for val_id, val_tree in val_trees_dict.items()
        }

        # prepare optimization configurations
        if optim_kwargs is None:
            optim_kwargs = {}
        optim_config = OptimConfig(**optim_kwargs)

        # create optimization manager
        from sckitflow.core.methods._opt import OptimizationManager

        opt_manager = OptimizationManager.from_config(self._method._module, optim_config)

        # initialize trainer
        if self._trainer is None:
            self._trainer = Trainer(self._method, opt_manager, callbacks)

        # module in training mode
        self._method.set_train_mode(True)

        # train model
        self._trainer.train(
            train_sampler,
            *args,
            val_samplers_dict=val_samplers_dict,
            n_train_steps=n_train_steps,
            valid_freq=valid_freq,
            **kwargs,
        )

    def predict(
        self,
        adata: AnnData,
        *args,
        return_raw: bool = False,
        sort: bool = True,
        control_values_dict: dict[str, str] | None = None,
        matched_keys: dict[tuple[Any], tuple[Any]] | None = None,
        require_target_state: bool = True,
        **kwargs,
    ) -> AnnData | tuple[AnnData, PredictionData]:
        """
        Generates flow predictions.

        :param adata: The input adata containing the metadata for prediction. When
            `require_target_state` is `False`, this only needs `.obs` (and optionally
            `.obsm` for continuous conditioning) describing the cells/conditions to
            predict for - no `.X` or expression `obsm` key is required.
        :type adata: class: `AnnData`

        :param return_raw: If True, returns the raw concatenated PredictionData
            keeping the computation graph alive. Defaults to `False`.
        :type return_raw: class: `bool`

        :param sort: If ``True``, create a sorted copy of *adata* via
            :meth:`sort_adata` and compile from that copy.  When setting this one
            should keep in mind this will copy the full adata object. When ``False`` (the
            default) the data must already be sorted; a ``ValueError``
            is raised otherwise.
        :type sort: class: `bool`

        :param control_values_dict: Optional dictionary mapping each condition
            level to the corresponding value used to indicate control observations.
            This overrides the homonimous attribute and is needed to allow
            inference over arbitrary control keys at inference time.
            Without this, inference would be bound to the source
            group defined for training. Defaults to `None`,
            in which case the instance attribute will be used.
        :type control_values_dict: class: `dict[str, str] | None`

        :param matched_keys: Optional keys used to identify the source  and
            corresponding target groups in the case of fixed matches.
            This overrides the homonimous attribute and is needed to allow
            inference over arbitrary matched groups at inference time.
            Without this, inference would be bound to the pairs of source
            and target groups defined for training. Defaults to `None`,
            in which case the instance attribute will be used.
        :type matched_keys: class: `dict[tuple[Any], tuple[Any]] | None`

        :param require_target_state: Whether `adata` needs to carry a target state
            representation (`.X` or the configured `obsm` sample representation).
            Set to `False` to predict purely from the conditioning metadata in
            `adata.obs`/`adata.obsm`, without needing target expression data.
            Defaults to `True`.
        :type require_target_state: class: `bool`

        :return: Either an AnnData with predictions, or a tuple (AnnData, PredictionData)
            if `return_raw` is True.
        """
        # Set module to evaluation mode (backend‑agnostic)
        self._method.set_train_mode(False)

        # Compile the data tree
        tree = self._dm.compile_adata(
            adata,
            sort=sort,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
            require_target_state=require_target_state,
        )
        tree_flat: tuple[MatchedData] = tree.flatten()

        # early return
        if not tree_flat:
            return self._predict_empty(return_raw)

        # define store
        all_preds = []
        all_obs = []
        all_obsm = defaultdict(list)

        # Turn nodes into ready `StepData` up front, so the loop below operates purely
        # on `StepData`. `_node_to_step_data` is the single place a node becomes
        # `StepData` -- swap it for a `StepData`-native loader to drop the node pipeline.
        step_data_stream = (self._node_to_step_data(node) for node in tree_flat)

        # Iterate over each ready `StepData`
        for step_data in tqdm(step_data_stream, total=len(tree_flat), desc="Predicting"):
            # 1. Inference
            pred_obj = self._predict_on_node(step_data, *args, **kwargs)
            all_preds.append(pred_obj)

            # 2. Construct obs dataframe (from `StepData`)
            pred_df = self._get_pred_obs_df(step_data, pred_obj)
            all_obs.append(pred_df.copy())

            # 3. Construct obsm (from `StepData`)
            node_obsm_dict = self._get_pred_obsm_dict(step_data, pred_obj)
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
    ) -> "Model":
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
