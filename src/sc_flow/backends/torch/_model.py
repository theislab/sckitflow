from typing import Any, Literal

import torch
from anndata import AnnData

from sc_flow.backends.torch._types import TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sc_flow.backends.torch.methods import AVAILABLE_METHODS, METHODS_REGISTRY
from sc_flow.backends.torch.methods._base import BaseGenerativeFlow
from sc_flow.backends.torch.probability_paths import BaseProbabilityPath
from sc_flow.backends.torch.solvers import BaseSolver
from sc_flow.data._dims_registry import DataDimensionalitiesRegistry
from sc_flow.data._manager import DataManager
from sc_flow.data.samplers._train import FTrainSampler
from sc_flow.data.samplers._validation import FValidationSampler
from sc_flow.trainer._trainer import Trainer

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
        """"""  # noqa
        # initialize data manager
        cls._dm_cls = DataManager(**kwargs)
        cls._dims_registry = cls._dm_cls.get_data_dimensionalities(adata)
        cls._is_paired_setting_cls = cls._dm_cls.control_values_dict is not None

    def __init__(
        self,
        *args,
        method_cls: BaseGenerativeFlow | None = None,
        method_id: AVAILABLE_METHODS | None = None,
        probability_path: BaseProbabilityPath | None = None,
        match_fn: TMatchFn | None = None,
        noise_sampler: TNoiseSamplerFn | None = None,
        time_sampler: TTimeSamplerFn | None = None,
        generate_from_noise: bool = False,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
        solver_cls: type[BaseSolver] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr: float = 5e-5,
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_step: Literal["train_step", "validation_step"] = "train_step",
        plan_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """"""  # noqa

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

        # initialize method
        self._method = method_cls(
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
            solver_cls=solver_cls,
            solver_kwargs=solver_kwargs,
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
        n_train_steps: int = 10_000,
        valid_freq: int = 1_000,
        val_adatas_dict: dict[str, AnnData] | None = None,
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
    ) -> None:
        """"""  # noqa
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
        self._trainer = Trainer(self._method)

        # train model
        self._trainer.train(
            train_sampler,
            val_samplers_dict,
            n_train_steps=n_train_steps,
            valid_freq=valid_freq,
        )

    def predict(self):
        """"""  # noqa
        raise NotImplementedError

    @property
    def dm(self) -> DataManager:
        """"""  # noqa
        return self._dm

    @property
    def is_paired_setting(self) -> DataManager:
        """"""  # noqa
        return self._is_paired_setting

    @property
    def method(self) -> DataManager:
        """"""  # noqa
        return self._method

    @property
    def trainer(self) -> Trainer:
        """"""  # noqa
        return self._trainer
