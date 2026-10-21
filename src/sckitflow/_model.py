"""A model as its spec plus weights, in the Hugging Face Hub layout.

:class:`ModelConfig` is what a model *is*: the data schema it reads, its module, how it is trained and how it
predicts. How one training run goes (optimizer, seeds, batch size, steps, the split, where the data lives) is
the experiment's business and stays out, so a saved model loads with sckitflow alone.

A saved model is :mod:`scfit.pretrained`'s folder: ``config.json`` holds ``ModelConfig.to_spec()`` and
``module/model.safetensors`` the module's ``state_dict``. Nothing is pickled. Share it with
``huggingface_hub.HfApi().upload_folder(repo_id=..., folder_path=path)`` and load it back by its repo id. A
model reads back under the package versions that saved it.

.. code-block:: python

    spec = ModelConfig(data=DataManagerConfig(...), module=MLPVelocityConfig(), training=CFMTrainingConfig())
    model = spec.build(adata)
    datamodule = FlowDataModule(model.manager, model.dims, adata, batch_size=256)
    plan = TrainingPlan(model.training_method, torch.optim.Adam(model.module.parameters()))
    lightning.Trainer(max_steps=1000).fit(plan, datamodule=datamodule)
    model.save_pretrained("model")

    model = Model.from_pretrained("model", adata)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from scfit.pretrained import load_pretrained, save_pretrained
from scfit.registry import Component, component

from sckitflow.core.methods._base import (
    InferenceMethodConfig,
    SupportsInference,
    SupportsTraining,
    TrainingMethodConfig,
)
from sckitflow.core.nn._config import ModuleConfig
from sckitflow.data._manager import DataManager, DataManagerConfig

if TYPE_CHECKING:
    import torch

    from sckitflow.data._dims import DataDimensions
    from sckitflow.data._utils import AnnDataLike

__all__ = ["Model", "ModelConfig"]


class Model(NamedTuple):
    """A built model: the spec, its schema and the data's dims, the module and its methods."""

    spec: ModelConfig
    manager: DataManager
    dims: DataDimensions
    module: torch.nn.Module
    training_method: SupportsTraining
    inference_method: SupportsInference | None

    def save_pretrained(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Writes the spec and the module's weights to the folder ``path``.

        :raises FileExistsError: If ``path`` holds a model and ``overwrite`` is `False`.
        :raises scfit.registry.PortabilityError: If the spec holds a live object, before anything is written.
        """
        save_pretrained(path, self.spec, {"module": self.module.state_dict()}, overwrite=overwrite)

    @classmethod
    def from_pretrained(cls, path_or_repo_id: str | Path, adata: AnnDataLike, *, revision: str | None = None) -> Model:
        """Rebuilds a model from a folder or a Hub repo id, sized by ``adata`` under the saved schema.

        :param revision: The Hub revision to load, for a repo id.
        """
        spec, states = load_pretrained(path_or_repo_id, revision=revision)
        if not isinstance(spec, ModelConfig):
            raise TypeError(f"{path_or_repo_id} holds a {type(spec).__name__}, not a model.")
        if "module" not in states:
            raise ValueError(f"{path_or_repo_id} holds no module weights (no module/model.safetensors).")
        model = spec.build(adata)
        model.module.load_state_dict(states["module"])
        return model


@component("model", builds=Model)
class ModelConfig(Component):
    """Everything portable about a model: what ``config.json`` holds."""

    data: DataManagerConfig
    """The schema, saying what each observation is, which flow into which, and where the split is read from."""
    module: ModuleConfig
    """Builds the module from the data's dims."""
    training: TrainingMethodConfig
    inference: InferenceMethodConfig | None = None

    def build(self, adata: AnnDataLike) -> Model:
        """The model, its module sized by ``adata`` under :attr:`data`."""
        manager = self.data.build()
        dims = manager.get_data_dimensionalities(adata)
        module = self.module.build(dims)
        return Model(
            self,
            manager,
            dims,
            module,
            self.training.build(module, dims=dims),
            self.inference.build(module, dims=dims) if self.inference is not None else None,
        )
