from typing import Any, Literal

import pandas as pd
import torch
from anndata import AnnData
from tqdm import tqdm

from sc_flow.backends.torch._types import TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sc_flow.backends.torch.coupling._coupling import independent_coupling
from sc_flow.backends.torch.methods import AVAILABLE_METHODS, METHODS_REGISTRY
from sc_flow.backends.torch.methods._base import BaseGenerativeFlow
from sc_flow.backends.torch.methods._utils import PredictionData
from sc_flow.backends.torch.probability_paths import BaseProbabilityPath, LinearDiracProbabilityPath
from sc_flow.data._composite import MatchedData
from sc_flow.data._dims_registry import DataDimensionalitiesRegistry
from sc_flow.data._manager import DataManager
from sc_flow.data.samplers._train import FTrainSampler
from sc_flow.data.samplers._validation import FValidationSampler
from sc_flow.trainer._trainer import Trainer

__all__ = ["SCFlow"]


class SCFlow:
    """Class for handling flow-based generative models with the torch backend.

    :param _dm_cls: The data manager associate to the current class.
        Defaults to `None`. To set it, you need to call the `register_adata`
        class method before instatiating a `SCFlow` object.
    :type DataManager: class: `DataManager | None`

    :param _dims_registry: The data dimensionality registry associate to the current class.
        Defaults to `None`. To set it, you need to call the `register_adata`
        class method before instatiating a `SCFlow` object.
    :type _dims_registry: class: `DataDimensionalitiesRegistry | None`

    :param _is_paired_setting_cls: Whether the data is condigured in a paired mode.
        Defaults to `False`. It will be automatically inferred
        from the schema during the call to the `register_adata`
        class method, before instatiating a `SCFlow` object.
    :type _is_paired_setting_cls: class: `DataManager | None`
    """

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
        method_cls: type[BaseGenerativeFlow] | None = None,
        method_id: AVAILABLE_METHODS | None = None,
        probability_path_cls: type[BaseProbabilityPath] | None = None,
        probability_path_kwargs: dict[str, Any] | None = None,
        match_fn: TMatchFn | None = None,
        noise_sampler: TNoiseSamplerFn | None = None,
        time_sampler: TTimeSamplerFn | None = None,
        generate_from_noise: bool = False,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr: float = 5e-5,
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_step: Literal["train_step", "validation_step"] = "train_step",
        plan_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initializes the model.

        :param *args: Positional arguments used to initialize the method class.
        :type *args: class: `Sequence[Any]`

        :param method_cls: Reference to a `BaseGenerativeFlow` class, used as the
            underlying method. When provided, it will take priority over the method
            set with :param: `method_id`. Defaults to `None`, in which case the
            :param: `method_id` will be required.
        :type method_cls: class: `type[BaseGenerativeFlow] | None`

        :param method_id: String identifier for the current method to instantiate.
            Defaults to `None`.
        :type method_id: class: `AVAILABLE_METHODS`

        :param probability_path_cls: Instance of the conditional probability path
            used to instantiate the flow model. Defaults to `None`.
        :type probability_path_cls: class: `type[BaseProbabilityPath] | None`

        :param match_fn: Function used to match the source and target groups.
            Defaults to `None`.
        :type match_fn: class: `TMatchFn | None`

        :param noise_sampler: Function used to sample random noise during training.
            Defaults to `None`.
        :type noise_sampler: class: `TNoiseSamplerFn | None`

        :param time_sampler: Function used to sample random time indices during training.
            Defaults to `None`.
        :type time_sampler: class: `TNoiseSamplerFn | None`

        :param generate_from_noise: Flag indicating whether the define the interpolation
            from noise rather than from a source distribution. Only used for the paired setting.
            That is, when the `register_adata` class method infers the :attr: `is_paired_setting` as `True`.
            Otherwise, this attribute will be automatically set to `True`, due to the absence of source states.
            Defaults to `False`.
        :type generate_from_noise: class: `bool`

        :param dtype: Data type for the computations. Defaults to `torch.float32`.
        :type dtype: class: `torch.dtype`

        :param device_id: Identifier for the device used for the computations.
            When available, defaults to "cuda". Otherwise is set to "cpu".
        :type device_id: class: `str`

        :param optimizer_cls: Reference to a `torch.optim.Optimizer` class, used to update the model's weights. Defaults to `torch.optim.Adam`.
        :type optimizer_cls: class: `type[torch.optim.Optimizer] | None`

        :param optimizer_kwargs: Keyword arguments used to initialize the optimizer class, defaults to `None`.
        :type optimizer_kwargs: class: `dict[str, Any] | None`

        :param lr: The learning rate used for optimization. Defaults to `5e-5`.
        :type lr: class: `float`

        :param lr_scheduler_cls: Reference to a `torch.optim.lr_scheduler.LRScheduler` class,
            used to adapt the learning rate during optimization. Defaults to `None`.
        :type lr_scheduler_cls: class: `type[torch.optim.lr_scheduler.LRScheduler] | None`

        :param lr_scheduler_kwargs: Keyword arguments used to initialize the learning rate scheduler class, defaults to `None`.
            Only used when :param: `lr_scheduler_cls` is not `None`.
        :type lr_scheduler_kwargs: class: `dict[str, Any] | None`

        :param lr_scheduler_step: Flag indicating whether the lr scheduler step should be performed after a training iteration
            or after a validation pass. Defaults to `"train_step"`. Only used when :param: `lr_scheduler_cls` is not `None`.
        :type lr_scheduler_step: class: `Literal["train_step", "validation_step"]`

        :param plan_kwargs: Additional arguments for the optimization plan.
        :type plan_kwargs: class: `dict[str, Any] | None`

        :param *kwargs: Keyword arguments used to initialize the method class.
        :type *kwargs: class: `dict[str, Any]`
        """
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

        # set defaults
        if match_fn is None:
            match_fn = independent_coupling
        if noise_sampler is None:
            noise_sampler = torch.randn_like
        if time_sampler is None:
            # for consistency models we need two time indices
            if method_id == "fm" and method_cls is None:

                def time_sampler(shape, **kwargs):
                    s = torch.rand(shape, **kwargs)
                    t = torch.rand(shape, **kwargs)
                    return s, t

            # otherwise fall back to one index only
            time_sampler = torch.rand

        # get method cls
        if method_cls is None and method_id is None:
            msg = "At least one of `method_id` or `method_cls` should be specified."
            raise ValueError(msg)

        # use registry when method not provided
        if method_cls is None:
            if method_id not in METHODS_REGISTRY:
                msg = f"Method {method_id} not supported, possible options are {list(METHODS_REGISTRY.keys())}."
                raise KeyError(msg)
            method_cls = METHODS_REGISTRY[method_id]

        # initialize probability path
        if probability_path_cls is None:
            probability_path_cls = LinearDiracProbabilityPath
        if probability_path_kwargs is None:
            probability_path_kwargs = {}
        probability_path = probability_path_cls(**probability_path_kwargs)

        # initialize method
        self._method: BaseGenerativeFlow = method_cls(
            self._dims_registry,
            self._dm,
            self._is_paired_setting,
            *args,
            probability_path=probability_path,
            match_fn=match_fn,
            noise_sampler=noise_sampler,
            time_sampler=time_sampler,
            generate_from_noise=generate_from_noise,
            dtype=dtype,
            device_id=device_id,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_step=lr_scheduler_step,
            plan_kwargs=plan_kwargs,
            **kwargs,
        )

        # prepare attributes
        self._trainer: Trainer | None = None

    def train(
        self,
        train_adata: AnnData,
        *args,
        val_adatas_dict: dict[str, AnnData] | None = None,
        n_train_steps: int = 100_000,
        valid_freq: int = 1_000,
        train_batch_size: int = 128,
        train_n_nodes: int = 1,
        train_replace_samples: bool = False,
        train_replace_nodes: bool = False,
        train_use_nodes_weights: bool = True,
        val_max_n_obs: int = 10_000,
        val_n_nodes: int = 1,
        val_replace_samples: bool = False,
        val_replace_nodes: bool = False,
        val_use_nodes_weights: bool = True,
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

        :param train_n_nodes: The number of nodes to sample for each batch of training data.
            Defaults to :constant sc_flow._constants.DEFAULT_N_GROUPS:.
        :type train_n_nodes: class: `int`

        :param train_replace_samples: Whether to sample training observations with replacement
            from each node. Defaults to `False`.
        :type train_replace_samples: class: `bool`

        :param train_replace_nodes: Whether to sample trainining nodes with replacement
            from the tree. Defaults to `False`.
        :type train_replace_nodes: class: `bool`

        :param train_use_nodes_weights: Whether to use nodes weights in order to sample training data.
            Each node weight is computed as its relative frequency over the whole tree.
            In order to compute the relative frequency of a node of matched distributions,
            only the target one is considered. Defaults to `True`.
        :type train_use_nodes_weights: class: `bool`

        :param val_max_n_obs: The maximum number of observations to sample for each node in a batch
            of validation data. Defaults to `10_000`.
        :type val_max_n_obs: class: `int`

        :param val_n_nodes: The number of nodes to sample for each batch of validation data.
            Defaults to :constant sc_flow._constants.DEFAULT_N_GROUPS:.
        :type val_n_nodes: class: `int`

        :param val_replace_samples: Whether to sample validation observations with replacement
            from each node. Defaults to `False`.
        :type val_replace_samples: class: `bool`

        :param val_replace_nodes: Whether to sample validation nodes with replacement
            from the tree. Defaults to `False`.
        :type val_replace_nodes: class: `bool`

        :param val_use_nodes_weights: Whether to use nodes weights in order to sample validation data.
            Each node weight is computed as its relative frequency over the whole tree.
            In order to compute the relative frequency of a node of matched distributions,
            only the target one is considered. Defaults to `True`.
        :type val_use_nodes_weights: class: `bool`

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

        # create samplers
        train_sampler = FTrainSampler(
            train_tree,
            batch_size=train_batch_size,
            n_nodes=train_n_nodes,
            replace_samples=train_replace_samples,
            replace_nodes=train_replace_nodes,
            use_nodes_weights=train_use_nodes_weights,
        )
        val_samplers_dict = {
            val_id: FValidationSampler(
                val_tree,
                max_n_obs=val_max_n_obs,
                n_nodes=val_n_nodes,
                replace_samples=val_replace_samples,
                replace_nodes=val_replace_nodes,
                use_nodes_weights=val_use_nodes_weights,
            )
            for val_id, val_tree in val_trees_dict.items()
        }

        # initialize trainer
        self._trainer = Trainer(self._method, *args, **kwargs)

        # module in training mode
        self._method.module.train()

        # train model
        self._trainer.train(
            train_sampler,
            val_samplers_dict,
            n_train_steps=n_train_steps,
            valid_freq=valid_freq,
        )

    def predict(
        self, adata: AnnData, *args, return_tensors: bool = False, **kwargs
    ) -> AnnData | tuple[AnnData, PredictionData]:
        """
        Generates flow predictions.

        :param adata: The input adata containing the metadata for prediction.
        :type adata: class: `AnnData`

        :param return_tensors: If True, returns the raw concatenated PredictionData
            keeping the computation graph alive. Defaults to `False`.
        :type return_tensors: class: `bool`
        """
        # prepare module and data tree
        self._method.module.eval()
        tree = self._dm.compile_adata(adata)
        tree_flat: tuple[MatchedData] = tree.flatten()

        # store for samples and metadata
        all_samples = []
        all_trajs = []
        all_obs = []

        # iterate over each node
        pbar = tqdm(tree_flat, desc="Predicting")
        for node in pbar:
            # 1. Inference (Differentiable)
            pred_obj: PredictionData = self._method.predict(node, *args, **kwargs)

            # 2. Collect Tensors (keep graph)
            all_samples.append(pred_obj.samples)
            if pred_obj.traj is not None:
                all_trajs.append(pred_obj.traj)

            # 3. Metadata
            ann_df = node.source_distr.ann_df if node.source_distr else node.target_distr.ann_df
            all_obs.append(ann_df.copy())

        # --- Construct AnnData (Detached from graph) ---
        X_final = torch.cat(all_samples, dim=0).detach().cpu().numpy()
        obs_final = pd.concat(all_obs, axis=0)

        pred_adata = AnnData(
            X=X_final,
            obs=obs_final,
            var=pd.DataFrame(index=self._dims_registry.feature_names),
        )

        # 4. Store Trajectory in obsm
        if all_trajs:
            # Concatenate on the cell dimension (dim=1 for torchdiffeq [T, N, D])
            # We want [N, T, D] for AnnData alignment
            full_traj = torch.cat(all_trajs, dim=1).transpose(0, 1).detach().cpu().numpy()
            pred_adata.obsm["trajectory"] = full_traj

        # optionally return differentiable tensors
        if return_tensors:
            # Concatenate tensors to return a single Differentiable object
            differentiable_output = PredictionData(
                samples=torch.cat(all_samples, dim=0), traj=torch.cat(all_trajs, dim=1) if all_trajs else None
            )
            return pred_adata, differentiable_output

        # clear memory
        del all_samples, all_trajs
        return pred_adata

    @property
    def dm(self) -> DataManager:
        """Returns the data manager associated to the current instance."""
        return self._dm

    @property
    def is_paired_setting(self) -> bool:
        """Whether the data was registered in a paired setting."""
        return self._is_paired_setting

    @property
    def method(self) -> BaseGenerativeFlow:
        """Returns the underlying method."""
        return self._method

    @property
    def trainer(self) -> Trainer:
        """Returns the trainer used to fit the model."""
        return self._trainer
