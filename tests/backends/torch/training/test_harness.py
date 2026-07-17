"""One harness, two objectives (torch-compute and JAX-compute), weights on torch."""

from __future__ import annotations

import numpy as np
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


def test_jax_objective_trains_through_same_harness():
    pytest.importorskip("jax")
    pytest.importorskip("cellflow")
    import jax
    import jax.numpy as jnp
    from cellflow._compat import ConstantNoiseFlow
    from cellflow.networks._velocity_field import ConditionalVelocityField

    from sc_flow.backends.torch.jaxbridge._objective import CellFlowFMObjective, JaxParamModule

    emb, max_comb, cond_dim, n = 6, 2, 4, 16
    vf = ConditionalVelocityField(
        output_dim=D,
        max_combination_length=max_comb,
        condition_mode="deterministic",
        regularization=1.0,
        condition_embedding_dim=emb,
        hidden_dims=(16, 16),
        decoder_dims=(16, 16),
        time_encoder_dims=(16, 16),
    )
    pp = ConstantNoiseFlow(sigma=0.0)
    ki, ke = jax.random.split(jax.random.PRNGKey(0))
    params = vf.init(
        {"params": ki, "condition_encoder": ke},
        t=jnp.ones((1, 1)),
        x_t=jnp.ones((1, D)),
        cond={"drug": jnp.ones((1, max_comb, cond_dim))},
        encoder_noise=jnp.ones((1, emb)),
        train=False,
    )["params"]

    model = JaxParamModule(params)
    objective = CellFlowFMObjective(vf, pp, seed=0)

    # weights are torch nn.Parameters — the optimizer's single source of truth
    assert len(model.param_tensors) > 0
    assert all(isinstance(p, torch.nn.Parameter) for p in model.parameters())

    rng = np.random.default_rng(0)
    src = rng.standard_normal((n, D)).astype(np.float32)
    tgt = (src + 2.0).astype(np.float32)
    import jax.numpy as _jnp

    cond = _jnp.asarray(rng.standard_normal((n, max_comb, cond_dim)).astype(np.float32))

    class _DS(IterableDataset):
        def __iter__(self):
            for i in range(40):
                yield {
                    "time": torch.from_numpy(np.random.default_rng(i).uniform(size=(n, 1)).astype(np.float32)),
                    "source": torch.from_numpy(src),
                    "target": torch.from_numpy(tgt),
                    "encoder_noise": torch.from_numpy(
                        np.random.default_rng(i + 99).standard_normal((n, emb)).astype(np.float32)
                    ),
                    "conditions": {"drug": cond},
                }

    module = SCFlowLightningModule(model, objective, lr=1e-2)
    rec = _LossRecorder()
    _trainer(rec).fit(module, DataLoader(_DS(), batch_size=None))
    assert rec.losses[-1] < rec.losses[0]  # JAX-computed loss went down via the torch optimizer
