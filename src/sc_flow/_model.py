import logging
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, overload

import cloudpickle
import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from sc_flow._runtime import raise_runtime_error_on_backend_not_supported
from sc_flow._types import PredictionData
from sc_flow.data._composite import MatchedData
from sc_flow.data._dims_registry import DataDimensionalitiesRegistry
from sc_flow.data._manager import DataManager
from sc_flow.data.samplers._train import FTrainSampler
from sc_flow.data.samplers._validation import FValidationSampler
from sc_flow.methods._methods import BaseMethod
from sc_flow.methods._opt import OptimConfig
from sc_flow.trainer._callbacks import BaseCallback, TrainingCallbacks
from sc_flow.trainer._trainer import Trainer

__all__ = ["SCFlow"]


class SCFlow:
    _dm_cls: DataManager | None = None
    _dims_registry: DataDimensionalitiesRegistry | None = None
    _is_paired_setting_cls: bool = False
    _view_on_condition_space_cls: bool = False
    _condition_state_key_cls: str | None = None

    @classmethod
    def register_adata(
        cls,
        adata: AnnData,
        view_on_condition_space: bool = False,
        condition_state_key: str | None = None,
        **kwargs,
    ) -> None:
        """Registers the input adata as a class attribute using the provided schema settings.

        :param adata: The input adata to register.
        :type adata: class: `AnnData`

        :param view_on_condition_space: Whether to model condiion as states.
            Defaults to `False`.
        :type view_on_condition_space: class: `bool`

        :param condition_state_key: The key for the continuous condition covariates to be viewed as state
            when :param: `view_on_condition_space` is `True`. This argument is ignored otherwise.
            Defaults to `None`.
        :type condition_state_key: `str | None`

        :param **kwargs: Other key-word arguments used to initialize the `DataManager`.
        :type **kwargs: class: `dict[str, Any]`
        """
        # initialize data manager
        cls._dm_cls = DataManager(**kwargs)
        cls._dims_registry = cls._dm_cls.get_data_dimensionalities(
            adata,
            view_on_condition_space=view_on_condition_space,
            condition_state_key=condition_state_key,
            fit_preproc=True,
            apply_transformations=True,
        )
        cls._is_paired_setting_cls = cls._dm_cls.control_values_dict is not None or cls._dm_cls.matched_keys is not None
        cls._view_on_condition_space_cls = view_on_condition_space
        cls._condition_state_key_cls = condition_state_key

    def __init__(
        self,
        *args,
        method_cls: type[BaseMethod] | None = None,
        method_id: str | None = None,
        backend: Literal["jax", "torch"] = "torch",
        **kwargs,
    ) -> None:
        # check that data was prepared
        if self.__class__._dm_cls is None:
            raise RuntimeError(
                f"Data has not been registered with {self.__class__.__name__}. "
                "Please call .register_adata(adata, ...) before initializing the model."
            )

        # register class attributes to instance
        self._dm = self.__class__._dm_cls
        self._dims_registry = self.__class__._dims_registry
        self._is_paired_setting = self.__class__._is_paired_setting_cls
        self._view_on_condition_space = self.__class__._view_on_condition_space_cls
        self._condition_state_key = self.__class__._condition_state_key_cls

        # register backend
        self._backend = backend

        # get method cls
        if method_cls is None and method_id is None:
            msg = "At least one of `method_id` or `method_cls` should be specified."
            raise ValueError(msg)

        # use registry when method not provided
        if method_cls is None:
            # get registry for current backend
            if backend == "torch":
                from sc_flow.backends.torch.methods import METHODS_REGISTRY
            elif backend == "jax":
                from sc_flow.backends.jax.methods import METHODS_REGISTRY
            else:
                from sc_flow._runtime import raise_runtime_error_on_backend_not_supported

                raise_runtime_error_on_backend_not_supported(backend)

            # get method from registry
            if method_id not in METHODS_REGISTRY:
                msg = f"Method {method_id} not supported, possible options are {list(METHODS_REGISTRY.keys())}."
                raise KeyError(msg)
            method_cls = METHODS_REGISTRY[method_id]

        # initialize method
        self._method: BaseMethod = method_cls(
            self._dims_registry,
            self._dm,
            self._is_paired_setting,
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

    def _get_pred_obs_df(self, node: MatchedData, pred_obj: PredictionData) -> pd.DataFrame:
        """Gets the observation dataframe for prediction"""
        # 1. Collect observation metadata
        ann_df = node.target_distr.ann_df.copy()
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

    def _get_pred_obsm_dict(self, node: MatchedData, pred_obj: PredictionData) -> dict[str, np.ndarray]:
        # ---- Get trajectory and samples ----
        traj = self._get_pred_traj(pred_obj)
        raw_samples = self._get_pred_raw_samples(pred_obj)

        # ---- Get condition continuous covariates data ----
        if node.target.has_continuous_condition_covariates:
            condition_continuous_covs = node.target.condition_data.continuous_covariates.mapping
        else:
            condition_continuous_covs = {}

        # ---- Get target continuous covariates data ----
        if node.target.has_continuous_response_covariates:
            response_continuous_covs = node.target.response_data.continuous_covariates.mapping
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

    def _predict_on_node(self, node: MatchedData, *args, **kwargs) -> PredictionData:
        return self._method.predict(node, *args, **kwargs)

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert a backend-specific tensor to a numpy array.

        :param tensor: A tensor from the current backend (e.g., torch.Tensor or jnp.ndarray).
        :return: numpy array
        """
        if tensor is None:
            return None
        if self._backend == "torch":
            import torch

            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
            return np.array(tensor)
        elif self._backend == "jax":
            return np.array(tensor)
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

    def to_device(self, device: str) -> None:
        """Move the underlying PyTorch module and optimizer state to the specified device."""
        if self._backend == "jax":
            raise NotImplementedError("Device moving is currently supported only for the torch backend.")
        elif self._backend == "torch":
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
        else:
            raise_runtime_error_on_backend_not_supported(self._backend)

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
            view_on_condition_space=self._view_on_condition_space,
            condition_state_key=self._condition_state_key,
            apply_transformations=True,
        )
        if val_adatas_dict is not None:
            val_trees_dict = {
                val_id: self._dm.compile_adata(
                    val_adata,
                    sort=sort,
                    view_on_condition_space=self._view_on_condition_space,
                    condition_state_key=self._condition_state_key,
                    apply_transformations=True,
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
        if self._backend == "torch":
            from sc_flow.backends.torch.methods._opt import TorchOptimizationManager

            opt_manager = TorchOptimizationManager.from_config(self._method._module, optim_config)
        elif self._backend == "jax":
            # Placeholder for future JAX support
            raise NotImplementedError("JAX optimization manager not yet implemented.")
        else:
            raise_runtime_error_on_backend_not_supported(self._backend)

        # initialize trainer
        if self._trainer is None:
            self._trainer = Trainer(self._method, opt_manager, callbacks)

        # module in training mode
        self._method.set_train_mode(True)

        # train model
        self._trainer.train(
            train_sampler, *args, val_samplers_dict=val_samplers_dict, n_train_steps=n_train_steps, valid_freq=valid_freq, **kwargs
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
            view_on_condition_space=self._view_on_condition_space,
            condition_state_key=self._condition_state_key,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
            apply_transformations=True,
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

        # Iterate over each node
        for node in tqdm(tree_flat, desc="Predicting"):
            # 0. Align node
            node_aligned = node.align()

            # 1. Inference
            pred_obj = self._predict_on_node(node_aligned, *args, **kwargs)
            all_preds.append(pred_obj)

            # 2. Construct node dataframe
            pred_df = self._get_pred_obs_df(node_aligned, pred_obj)
            all_obs.append(pred_df.copy())

            # 3. Construct node obsm
            node_obsm_dict = self._get_pred_obsm_dict(node_aligned, pred_obj)
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
        if self._backend == "torch":
            import torch

            self._method._module.cpu()
            # Fixed: access optimizer via trainer, not via method
            if self._trainer is not None and hasattr(self._trainer, "opt_manager"):
                opt = self._trainer.opt_manager.optimizer
                for state in opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()

        # unload preprocessing context
        self.dm.unload_preproc()

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
    ) -> "SCFlow":
        """
        Load a saved model from a tarball.

        :param filepath: Path to the saved tarball.
        :param adata: Optional AnnData to re‑register if the saved model does not contain data.
        :param map_location: For PyTorch models, map to a device (e.g., 'cuda:0').
        :param register_kwargs: Additional arguments for register_adata if adata is provided.
        :return: Loaded SCFlow instance.
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

        # If an AnnData is provided, re‑register it (overwrites the saved data manager if any)
        if adata is not None:
            cls.register_adata(adata, **register_kwargs)
            # Replace the instance's data manager with the newly registered one
            model._dm = cls._dm_cls
            model._dims_registry = cls._dims_registry
            model._is_paired_setting = cls._is_paired_setting_cls

        # Move to desired device if using PyTorch
        if model._backend == "torch" and map_location is not None:
            model.to_device(map_location)

        return model

    @property
    def backend(self) -> str:
        """Returns the backend the model was initialized on."""
        return self._backend

    @property
    def dm(self) -> DataManager:
        """Returns the data manager associated to the current instance."""
        return self._dm

    @property
    def is_paired_setting(self) -> bool:
        """Whether the data was registered in a paired setting."""
        return self._is_paired_setting

    @property
    def method(self) -> BaseMethod:
        """Returns the underlying method."""
        return self._method

    @property
    def trainer(self) -> Trainer:
        """Returns the trainer used to fit the model."""
        return self._trainer

    @property
    def view_on_condition_space(self) -> bool:
        """Return whether the model is operating on the condition space."""
        return self._view_on_condition_space

    @property
    def condition_state_key(self) -> str | None:
        """Return the key used to extract the state from the condition."""
        return self._condition_state_key
