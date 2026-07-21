"""The Lightning harness trains a torch objective; weights live on torch."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from sc_flow.backends.torch.training import SCFlowLightningModule, TorchLinearFMObjective

D = 2


def _trainer(cb):
    return pl.Trainer(
        max_steps=40,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[cb],
    )


class _LossRecorder(pl.Callback):
    def __init__(self):
        self.losses: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.losses.append(float(outputs["loss"]))


class _TorchVNet(torch.nn.Module):
    def __init__(self, d=D, h=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d + 1, h), torch.nn.SiLU(), torch.nn.Linear(h, d))

    def forward(self, t, x):
        return self.net(torch.cat([x, t], dim=-1))


def test_torch_objective_trains_through_harness():
    torch.manual_seed(0)
    n = 64
    src = torch.zeros(n, D)
    tgt = torch.ones(n, D) * 2.0  # learn the constant translation +2

    class _DS(IterableDataset):
        def __iter__(self):
            for _ in range(40):
                yield {"source": src, "target": tgt}

    module = SCFlowLightningModule(_TorchVNet(), TorchLinearFMObjective(), lr=1e-2)
    assert all(isinstance(p, torch.nn.Parameter) for p in module.model.parameters())  # weights on torch

    rec = _LossRecorder()
    _trainer(rec).fit(module, DataLoader(_DS(), batch_size=None))
    assert rec.losses[-1] < rec.losses[0]
