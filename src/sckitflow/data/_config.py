"""Portable config for the data side of a run.

Every field is JSON-serializable, so the schema travels as a spec rather than a
pickled :class:`DataManager`. ``build(adata, rng=...)`` returns the ready
:class:`~sckitflow.data.FlowDataModule`; the ``AnnData`` is never in the spec.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from anndata import AnnData
from pydantic import PositiveInt, field_validator
from scfit.registry import Component, component

from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._manager import DataManagerConfig
from sckitflow.data.splitters._base import Splitter

__all__ = ["FlowDataModuleConfig"]


@component("data_module.flow", builds=FlowDataModule)
class FlowDataModuleConfig(Component):
    """The schema and the streaming options a run reads its batches with.

    The splitter is not a field: it has its own seed, so the run builds it
    beside this config and passes it to :meth:`build`.
    """

    manager: DataManagerConfig = DataManagerConfig()
    """The schema: what each observation is, which flow into which, and which are held out."""

    # --- streaming ---
    train_split: str = "train"
    n_train_steps: PositiveInt = 100_000
    batch_size: PositiveInt = 128
    dtype: Literal["float16", "bfloat16", "float32", "float64"] = "float32"
    """Name of a ``torch`` dtype. A `torch.dtype` is not JSON."""
    loader_kwargs: dict[str, Any] = {}
    """Forwarded to every loader. No ``seed``: the schedule is drawn from ``rng``."""

    @field_validator("loader_kwargs")
    @classmethod
    def _no_seed(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "seed" in kwargs:
            raise ValueError("loader_kwargs must not set `seed`; the schedule is drawn from `rng`.")
        return kwargs

    def build(self, adata: AnnData, *, rng: np.random.Generator, splitter: Splitter | None = None) -> FlowDataModule:
        """Builds the data manager and returns the data module over ``adata``.

        :param adata: The `AnnData` to derive dimensionalities from and stream.
        :param rng: The loaders' sampling schedule is seeded from it.
        :param splitter: Applied to ``adata`` before streaming; exclusive with ``manager.split_by``.
        """
        return FlowDataModule.from_adata(
            adata,
            self.manager.build(splitter=splitter),
            train_split=self.train_split,
            n_train_steps=self.n_train_steps,
            batch_size=self.batch_size,
            dtype=getattr(torch, self.dtype),
            loader_kwargs={**self.loader_kwargs, "seed": int(rng.integers(2**63))},
        )
