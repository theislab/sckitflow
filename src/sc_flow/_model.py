import logging
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cloudpickle
import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from sc_flow._runtime import raise_runtime_error_on_backend_not_supported
from sc_flow.data._composite import MatchedData
from sc_flow.data._dims_registry import DataDimensionalitiesRegistry
from sc_flow.data._manager import DataManager
from sc_flow.data.samplers._train import FTrainSampler
from sc_flow.data.samplers._validation import FValidationSampler
from sc_flow.methods._methods import BaseMethod
from sc_flow.methods._opt import OptimConfig
from sc_flow.trainer._trainer import Trainer

if TYPE_CHECKING:
    from sc_flow.backends.jax._types import PredictionData as JaxPredictionData
    from sc_flow.backends.torch._types import PredictionData as TorchPredictionData

__all__ = ["SCFlow"]


class SCFlow:
    _dm_cls: DataManager | None = None
    _dims_registry: DataDimensionalitiesRegistry | None = None
    _is_paired_setting_cls: bool = False

    @classmethod
    def register_adata(
        cls,
        adata: AnnData,
        **kwargs,
    ) -> None:
        """Registers the input adata as a class attribute using the provided schema settings.

        :param adata: The input adata to register.
        :type adata: class: `AnnData`

        :param **kwargs: Other key-word arguments used to initialize the `DataManager`.
        :type **kwargs: class: `dict[str, Any]`
        """
        # initialize data manager
        cls._dm_cls = DataManager(**kwargs)
        cls._dims_registry = cls._dm_cls.get_data_dimensionalities(adata)
        cls._is_paired_setting_cls = cls._dm_cls.control_values_dict is not None

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
        n_train_steps: int = 100_000,
        valid_freq: int = 1_000,
        train_batch_size: int = 128,
        val_max_n_obs: int = 10_000,
        train_sampler_kwargs: dict[str, Any] | None = None,
        val_sampler_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        optim_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Trains the model on the input adata.

        :param train_adata: The train adata.
        :type train_adata: class: `AnnData`

        :param *args: Positional arguments used to initialize the trainer class.
        :type *args: class: `Sequence[Any]`

        :param val_adatas_dict: Dictionary containing the validation adatas.
        :type val_adatas_dict: class: `dict[str, AnnData]`

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

        :param train_kwargs: Extra keyword arguments for the call to the trainer `.train` method. Defaults to `None`.
        :type train_kwargs: class: `dict[str, Any] | None`

        :param optim_kwargs: Keyword arguments to configure for the optimization manager (optimizer, scheduler, etc.). Defaults to `None`.
        :type optim_kwargs: class: `OptimConfig | None`

        :param *kwargs: Keyword arguments used to initialize the trainer class.
        :type *kwargs: class: `dict[str, Any]`
        """
        # compile adata
        train_tree = self._dm.compile_adata(train_adata)
        if val_adatas_dict is not None:
            val_trees_dict = {
                val_id: self._dm.compile_adata(val_adata) for val_id, val_adata in val_adatas_dict.items()
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

        # initialize trainer (now passing optimization manager)
        self._trainer = Trainer(self._method, opt_manager, *args, **kwargs)

        # module in training mode
        self._method.set_train_mode(True)

        # train model
        if train_kwargs is None:
            train_kwargs = {}
        self._trainer.train(
            train_sampler, val_samplers_dict, n_train_steps=n_train_steps, valid_freq=valid_freq, **train_kwargs
        )

    def predict(
        self,
        adata: AnnData,
        *args,
        return_tensors: bool = False,
        **kwargs,
    ) -> AnnData | tuple[AnnData, "TorchPredictionData | JaxPredictionData"]:
        """
        Generates flow predictions.

        :param adata: The input adata containing the metadata for prediction.
        :param return_tensors: If True, returns the raw concatenated PredictionData
            keeping the computation graph alive. Defaults to `False`.
        :return: Either an AnnData with predictions, or a tuple (AnnData, PredictionData)
            if `return_tensors` is True.
        """
        # Set module to evaluation mode (backend‑agnostic)
        self._method.set_train_mode(False)

        # Compile the data tree
        tree = self._dm.compile_adata(adata)
        tree_flat: tuple[MatchedData] = tree.flatten()

        # define store
        all_preds = []
        all_obs = []

        # Iterate over each node (e.g., cell type / condition group)
        for node in tqdm(tree_flat, desc="Predicting"):
            # 1. Inference – returns backend‑specific PredictionData
            pred_obj = self._method.predict(node, *args, **kwargs)
            all_preds.append(pred_obj)

            # 2. Collect observation metadata
            ann_df = node.target_distr.ann_df
            all_obs.append(ann_df.copy())

        # early return
        if not all_preds:
            # Return empty AnnData
            empty_adata = AnnData(
                X=np.empty((0, len(self._dims_registry.feature_names))),
                var=pd.DataFrame(index=self._dims_registry.feature_names),
            )
            return empty_adata if not return_tensors else (empty_adata, None)

        # 3. Merge predictions using backend‑specific concatenation
        merged_pred = type(all_preds[0]).concatenate(all_preds)

        # 4. Convert to numpy for AnnData construction
        X_np = self._to_numpy(merged_pred.samples)
        traj_np = self._to_numpy(merged_pred.traj) if merged_pred.traj is not None else None

        # 5. Build AnnData object
        obs_final = pd.concat(all_obs, axis=0)
        pred_adata = AnnData(
            X=X_np,
            obs=obs_final,
            var=pd.DataFrame(index=self._dims_registry.feature_names),
        )

        # 6. Store trajectory in obsm if available.
        # Solvers may emit trajectories as (n_time_steps, n_cells, n_features);
        # AnnData.obsm requires the first dimension to be n_obs (= n_cells), so
        # normalize to (n_cells, n_time_steps, n_features) before storing.
        if traj_np is not None:
            if traj_np.shape[0] == pred_adata.n_obs:
                traj_obsm = traj_np
            elif traj_np.ndim == 3 and traj_np.shape[1] == pred_adata.n_obs:
                traj_obsm = np.transpose(traj_np, (1, 0, 2))
            else:
                raise ValueError(
                    "Trajectory array has incompatible shape for AnnData.obsm: "
                    f"got {traj_np.shape}, expected first dimension to equal "
                    f"n_obs ({pred_adata.n_obs}) or, for 3D trajectories, second "
                    "dimension to equal n_obs so it can be transposed from "
                    "(n_time_steps, n_cells, n_features) to "
                    "(n_cells, n_time_steps, n_features)."
                )
            pred_adata.obsm["trajectory"] = traj_obsm

        # 7. Optionally return differentiable tensors (for further computation)
        if return_tensors:
            return pred_adata, merged_pred

        return pred_adata

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
