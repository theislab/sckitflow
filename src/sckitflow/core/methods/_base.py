import abc
from typing import Any, TypeVar

import lightning.pytorch as pl
import torch

from sckitflow.core._data_utils import extract_step_data
from sckitflow.core._types import PredictionData, StepData, TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sckitflow.core.methods._opt import OptimConfig
from sckitflow.core.nn._modules import BaseModule
from sckitflow.core.probability_paths import BaseProbabilityPath
from sckitflow.core.solvers import BaseSolver
from sckitflow.data._composite import MatchedDistributions
from sckitflow.data._dims_registry import DataDimensionalitiesRegistry
from sckitflow.data._manager import DataManager

__all__ = ["BaseMethod", "GenerativeFlow"]

T = TypeVar("T")


class BaseMethod(pl.LightningModule, abc.ABC):
    """Base class for methods, and the :class:`~lightning.pytorch.LightningModule` that trains them.

    A *node* -- one :class:`MatchedDistributions` of matched source/target observations --
    is the unit of training: one node in, one optimizer step out. Lightning drives the
    loop, so subclasses only implement :meth:`compute_loss` (the loss) and :meth:`infer`
    (inference); ``zero_grad``/``backward``/``step``, device placement, train/eval mode
    and progress reporting are all handled for them.
    """

    _module_cls: type[BaseModule] | None = None

    def __init__(
        self,
        dims_registry: DataDimensionalitiesRegistry,
        dm: DataManager,
        *args,
        dtype: torch.dtype = torch.float32,
        device_id: str | torch.device | None = None,
        optim_config: OptimConfig | None = None,
        val_predict_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initializes the method and the module it trains.

        :param dims_registry: The data dimensionalities used to size the module.
        :type dims_registry: class: `DataDimensionalitiesRegistry`

        :param dm: The fitted data manager describing the data schema.
        :type dm: class: `DataManager`

        :param dtype: Floating point type the module and the extracted batches use.
            Defaults to `torch.float32`.
        :type dtype: class: `torch.dtype`

        :param device_id: (Optional) Device to pin the method to. Leave as `None` -- the
            default -- when training: the trainer's accelerator decides placement and
            :attr: `device` follows it. Set it only to use the method standalone, outside
            a trainer.
        :type device_id: class: `str | torch.device | None`

        :param optim_config: (Optional) Optimizer/scheduler configuration used by
            :meth:`configure_optimizers`. Defaults to `None`, in which case a default
            :class:`OptimConfig` is used. Usually set by :meth:`Model.train`.
        :type optim_config: class: `OptimConfig | None`

        :param val_predict_kwargs: (Optional) Keyword arguments forwarded to
            :meth:`predict` during validation. Defaults to `None`. Needed when prediction
            requires arguments the sampler cannot supply, e.g. ``n_samples`` for methods
            that generate from noise.
        :type val_predict_kwargs: class: `dict[str, Any] | None`
        """
        super().__init__()

        # initialize attributes
        self._dims_registry = dims_registry
        self._dm = dm
        self.optim_config = OptimConfig() if optim_config is None else optim_config
        self.val_predict_kwargs = {} if val_predict_kwargs is None else val_predict_kwargs

        # check module is passed
        if self._module_cls is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define a `_module_cls` class attribute.")

        # initialize module with dimensionality registry
        self._module = self._module_cls.init_from_dims_registry(self._dims_registry, *args, **kwargs)

        # `.to` keeps `self.dtype`/`self.device` in step, which is what batch extraction
        # reads -- assigning the underlying attributes directly would not.
        self.to(dtype)
        if device_id is not None:
            self.to(device_id)

    # -------------------------------------------------------------------------
    # Subclass contract
    # -------------------------------------------------------------------------
    @abc.abstractmethod
    def compute_loss(
        self,
        step_data: StepData,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Computes the training loss for one node, plus the metrics to log."""

    @abc.abstractmethod
    def infer(
        self,
        step_data: StepData,
        *args,
        **kwargs,
    ) -> PredictionData: ...

    @staticmethod
    def _safe_subscript_obj(data: T | None, idx: Any | None) -> T | None:  # TODO: Probably remove from here
        if data is None:
            return None
        if idx is None:
            return data
        return data[idx]

    def _match_observations(
        self,
        step_data: StepData,
    ) -> StepData:
        return step_data

    def _train_step_forward(
        self,
        step_data: StepData,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        step_data = self._match_observations(step_data)
        return self.compute_loss(
            step_data,
            *args,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Lightning hooks
    # -------------------------------------------------------------------------
    def transfer_batch_to_device(
        self,
        batch: MatchedDistributions,
        device: torch.device,
        dataloader_idx: int,
    ) -> MatchedDistributions:
        """Leaves the node on the host instead of moving it to ``device``.

        A node is a tree of frozen dataclasses wrapping numpy arrays, which Lightning's
        default recursive transfer refuses to walk. :func:`extract_step_data` does the
        tensor conversion and device placement per step instead, so there is nothing to
        move here.
        """
        return batch

    def training_step(self, node: MatchedDistributions, batch_idx: int) -> torch.Tensor:
        """Runs one training step over a single node.

        Returning the loss is the whole contract: Lightning runs the backward pass and
        steps the optimizer and scheduler.
        """
        loss, logs = self.train_step(node)
        # `on_step` rather than `on_epoch`: the training stream is unbounded, so there is
        # only ever one epoch and per-step values are the only meaningful granularity.
        # `batch_size` is passed explicitly because Lightning cannot infer it from a node.
        self.log_dict(
            logs,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=getattr(node, "n_target_obs", 1),
        )
        return loss

    def validation_step(
        self,
        node: MatchedDistributions,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        """Predicts on a single validation node.

        Returns the raw predictions and targets rather than reducing them here, so metric
        callbacks can accumulate across nodes before computing.
        """
        preds = self.predict(node, **self.val_predict_kwargs)
        target = node.target_distr.state_data
        return {
            "predictions": preds.X,
            "targets": None if target is None else target.X,
        }

    def configure_optimizers(self) -> dict[str, Any]:
        """Builds the optimizer, and optional scheduler, from :attr: `optim_config`."""
        return self.optim_config.resolve(self.parameters())

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def train_step(
        self,
        matched_distr: MatchedDistributions,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Computes the loss and metrics for one node of matched distributions.

        Kept separate from :meth:`training_step` so methods registered with
        ``category="general"`` can override the whole step while Lightning still owns the
        loop.

        :param matched_distr: Input `MatchedDistributions` object.
        :type matched_distr: class: `MatchedDistributions`
        """
        step_data = extract_step_data(matched_distr, device=self.device, dtype=self.dtype)
        return self._train_step_forward(step_data, *args, **kwargs)

    def predict(
        self,
        data: MatchedDistributions | StepData,
        *args,
        no_grad: bool = True,
        **kwargs,
    ) -> PredictionData:
        """Prediction on node."""
        # extract step data and prepare latent state
        if isinstance(data, MatchedDistributions):
            data = extract_step_data(data, device=self.device, dtype=self.dtype)
        if not isinstance(data, StepData):
            raise ValueError(f"Data is of the wrong type, expected `StepData`, but {type(data)} found.")

        # optionally stop gradients
        if no_grad:
            with torch.no_grad():
                return self.infer(
                    data,
                    *args,
                    **kwargs,
                )
        else:
            return self.infer(
                data,
                *args,
                **kwargs,
            )

    @property
    def module(self) -> BaseModule:
        return self._module

    @property
    def dm(self) -> DataManager | None:
        return self._dm

    @property
    def dims_registry(self) -> DataDimensionalitiesRegistry | None:
        return self._dims_registry

    @property
    def is_paired_setting(self) -> bool:
        return self._dm.control_values_dict is not None or self._dm.matched_keys is not None


class GenerativeFlow(BaseMethod):
    _default_solver_cls: type[BaseSolver] | None = None

    def __init__(
        self,
        dims_registry: DataDimensionalitiesRegistry,
        dm: DataManager,
        *args,
        probability_path: BaseProbabilityPath | None = None,
        match_fn: TMatchFn | None = None,
        noise_sampler: TNoiseSamplerFn | None = None,
        time_sampler: TTimeSamplerFn | None = None,
        generate_from_noise: bool = False,
        **kwargs,
    ) -> None:
        # initialize parent class
        super().__init__(dims_registry, dm, *args, **kwargs)

        # set attributes
        self._probability_path = probability_path
        self._match_fn = match_fn
        self._noise_sampler = noise_sampler
        self._time_sampler = time_sampler

        # automatically fall back to noise generation when
        # no control values are provided
        if not self.is_paired_setting:
            generate_from_noise = True
        self._generate_from_noise = generate_from_noise

    def _call_match_fn_safe(
        self,
        source_lin: torch.Tensor | None,
        source_quad: torch.Tensor | None,
        target_lin: torch.Tensor | None,
        target_quad: torch.Tensor | None,
    ):
        # case 0: no source, do nothing
        if source_lin is None and source_quad is None:
            src_idxs = None
            tgt_idxs = None
            return src_idxs, tgt_idxs

        # case 1: source, match groups
        src_idxs, tgt_idxs = self._match_fn(
            source_lin=source_lin,
            target_lin=target_lin,
            source_quad=source_quad,
            target_quad=target_quad,
        )
        return src_idxs, tgt_idxs

    def _match_observations(
        self,
        step_data: StepData,
    ) -> StepData:
        # Get matching indices
        src_idxs, tgt_idxs = self._call_match_fn_safe(
            step_data.source_coupling_lin,
            step_data.source_coupling_quad,
            step_data.target_coupling_lin,
            step_data.target_coupling_quad,
        )

        # Case: no source distribution → return step_data unchanged (or with source=None)
        if src_idxs is None and tgt_idxs is None:
            # Already no source; keep target as is
            return step_data

        # Slice source side
        source_state = self._safe_subscript_obj(step_data.source_state, src_idxs)
        source_condition_data = self._safe_subscript_obj(step_data.source_condition_data, src_idxs)
        source_group_data = self._safe_subscript_obj(step_data.source_group_data, src_idxs)
        source_coupling_lin = self._safe_subscript_obj(step_data.source_coupling_lin, src_idxs)
        source_coupling_quad = self._safe_subscript_obj(step_data.source_coupling_quad, src_idxs)

        # Slice target side
        target_state = self._safe_subscript_obj(step_data.target_state, tgt_idxs)
        target_condition_data = self._safe_subscript_obj(step_data.target_condition_data, tgt_idxs)
        target_group_data = self._safe_subscript_obj(step_data.target_group_data, tgt_idxs)
        target_coupling_lin = self._safe_subscript_obj(step_data.target_coupling_lin, tgt_idxs)
        target_coupling_quad = self._safe_subscript_obj(step_data.target_coupling_quad, tgt_idxs)

        # Return new StepData with matched slices
        return StepData(
            target_state=target_state,
            target_coupling_lin=target_coupling_lin,
            target_coupling_quad=target_coupling_quad,
            target_condition_data=target_condition_data,
            target_group_data=target_group_data,
            source_state=source_state,
            source_coupling_lin=source_coupling_lin,
            source_coupling_quad=source_coupling_quad,
            source_condition_data=source_condition_data,
            source_group_data=source_group_data,
        )

    @abc.abstractmethod
    def infer(
        self,
        step_data: StepData,
        *args,
        solver_cls: type[BaseSolver] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
        n_steps: int = 100,
        latent: torch.Tensor | None = None,
        n_samples: int | None = None,
        **kwargs,
    ) -> PredictionData: ...

    @property
    def generate_from_noise(self) -> bool:
        return self._generate_from_noise

    @property
    def probability_path(self) -> BaseProbabilityPath | None:
        return self._probability_path

    @property
    def match_fn(self) -> TMatchFn | None:
        return self._match_fn

    @property
    def noise_sampler(self) -> TNoiseSamplerFn | None:
        return self._noise_sampler

    @property
    def time_sampler(self) -> TTimeSamplerFn | None:
        return self._time_sampler
