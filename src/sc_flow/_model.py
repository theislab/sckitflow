from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from sc_flow.data._composite import MatchedData
from sc_flow.data._dims_registry import DataDimensionalitiesRegistry
from sc_flow.data._manager import DataManager
from sc_flow.data.samplers._train import FTrainSampler
from sc_flow.data.samplers._validation import FValidationSampler
from sc_flow.methods._methods import BaseMethod
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

        # initialize trainer
        self._trainer = Trainer(self._method, *args, **kwargs)

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

        # 6. Store trajectory in obsm if available
        if traj_np is not None:
            # Trajectory shape: (n_cells, n_time_steps, n_features)
            pred_adata.obsm["trajectory"] = traj_np

        # 7. Optionally return differentiable tensors (for further computation)
        if return_tensors:
            return pred_adata, merged_pred

        return pred_adata

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
