"""Equivalence + smoke tests for the JAX-compute / torch-optimize CellFlow bridge.

The bridge must reproduce pure-JAX CellFlow's loss and gradient *exactly* (it is
literally the same JAX computation, only mirrored through DLPack), and a full
Lightning loop must actually update the torch-owned parameters and reduce the loss.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("cellflow")

import jax
import jax.numpy as jnp
import lightning.pytorch as pl
import torch
from cellflow._compat import ConstantNoiseFlow
from cellflow.networks._velocity_field import ConditionalVelocityField
from torch.utils.data import DataLoader, IterableDataset

from sc_flow.backends.torch.jaxbridge import (
    CellFlowJaxModule,
    JaxLossFunction,
    jax_to_torch,
    make_fm_value_and_grad,
    torch_to_jax,
)

N, D, EMB, MAX_COMB, COND_DIM = 8, 5, 6, 2, 4


def _build_vf(condition_mode: str) -> ConditionalVelocityField:
    return ConditionalVelocityField(
        output_dim=D,
        max_combination_length=MAX_COMB,
        condition_mode=condition_mode,
        regularization=1.0,
        condition_embedding_dim=EMB,
        hidden_dims=(16, 16),
        decoder_dims=(16, 16),
        time_encoder_dims=(16, 16),
    )


def _init_params(vf: ConditionalVelocityField, seed: int = 0):
    k_init, k_enc = jax.random.split(jax.random.PRNGKey(seed))
    return vf.init(
        {"params": k_init, "condition_encoder": k_enc},
        t=jnp.ones((1, 1)),
        x_t=jnp.ones((1, D)),
        cond={"drug": jnp.ones((1, MAX_COMB, COND_DIM))},
        encoder_noise=jnp.ones((1, EMB)),
        train=False,
    )["params"]


def _fixed_batch(seed: int = 42):
    ks = jax.random.split(jax.random.PRNGKey(seed), 5)
    return {
        "time": jax.random.uniform(ks[0], (N, 1)),
        "source": jax.random.normal(ks[1], (N, D)),
        "target": jax.random.normal(ks[2], (N, D)),
        "encoder_noise": jax.random.normal(ks[3], (N, EMB)),
        "conditions": {"drug": jax.random.normal(ks[4], (N, MAX_COMB, COND_DIM))},
    }


def test_dlpack_roundtrip_zero_copy():
    """torch -> jax -> torch preserves values (shared-buffer DLPack path)."""
    x = torch.randn(4, 3, dtype=torch.float32)
    back = jax_to_torch(torch_to_jax(x))
    assert torch.allclose(back, x)


@pytest.mark.parametrize("condition_mode", ["deterministic", "stochastic"])
def test_bridge_matches_pure_jax(condition_mode: str):
    """A single step's loss and gradients match pure-JAX CellFlow to tolerance."""
    vf = _build_vf(condition_mode)
    pp = ConstantNoiseFlow(sigma=0.0)
    params = _init_params(vf)
    batch = _fixed_batch()
    rng = jax.random.PRNGKey(123)

    # Pure-JAX reference: value_and_grad of CellFlow's OT-FM loss.
    vg = make_fm_value_and_grad(vf, pp, vf.condition_mode, vf.regularization)
    loss_jax, grads_jax = vg(
        params, batch["time"], batch["source"], batch["target"], batch["conditions"], batch["encoder_noise"], rng
    )
    grad_leaves_jax = jax.tree_util.tree_leaves(grads_jax)

    # torch bridge: same params (torch-owned), same batch, same rng.
    module = CellFlowJaxModule(vf, pp, params)
    loss_t = JaxLossFunction.apply(
        lambda p: vg(
            p, batch["time"], batch["source"], batch["target"], batch["conditions"], batch["encoder_noise"], rng
        ),
        module._treedef,
        *module._params,
    )
    loss_t.backward()

    assert loss_t.detach().item() == pytest.approx(float(loss_jax), abs=1e-6)

    grad_leaves_torch = [p.grad for p in module._params]
    assert len(grad_leaves_torch) == len(grad_leaves_jax)
    for gj, gt in zip(grad_leaves_jax, grad_leaves_torch, strict=True):
        gj_np = np.asarray(gj)
        gt_np = gt.detach().cpu().numpy()
        assert gt_np.shape == gj_np.shape
        assert np.max(np.abs(gj_np - gt_np)) < 1e-6


def test_lightning_training_updates_params_and_reduces_loss():
    """A full Lightning loop drives the torch optimizer and lowers the JAX loss."""
    vf = _build_vf("deterministic")
    pp = ConstantNoiseFlow(sigma=0.0)
    params = _init_params(vf)

    rng = np.random.default_rng(0)
    src = rng.standard_normal((N, D)).astype(np.float32)
    tgt = (src + 2.0).astype(np.float32)  # learnable constant shift
    cond = jnp.asarray(rng.standard_normal((N, MAX_COMB, COND_DIM)).astype(np.float32))

    class _DS(IterableDataset):
        def __init__(self, steps: int):
            self.steps = steps

        def __iter__(self):
            for i in range(self.steps):
                yield {
                    "time": torch.from_numpy(np.random.default_rng(i).uniform(size=(N, 1)).astype(np.float32)),
                    "source": torch.from_numpy(src),
                    "target": torch.from_numpy(tgt),
                    "encoder_noise": torch.from_numpy(
                        np.random.default_rng(i + 100).standard_normal((N, EMB)).astype(np.float32)
                    ),
                    "conditions": {"drug": cond},
                }

    module = CellFlowJaxModule(vf, pp, params, lr=1e-2, seed=0)
    p0 = module._params[0].detach().clone()

    losses: list[float] = []

    class _Cb(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            losses.append(float(outputs["loss"]))

    trainer = pl.Trainer(
        max_steps=30,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[_Cb()],
    )
    trainer.fit(module, DataLoader(_DS(30), batch_size=None))

    p1 = module._params[0].detach().clone()
    assert (p0 - p1).abs().max() > 0  # torch optimizer moved the params
    assert losses[-1] < losses[0]  # JAX loss went down


def test_adapter_and_objective_end_to_end():
    """Verify that FlowSpec -> Loader -> iter_fm_batches -> CellFlowFMObjective -> JaxParamModule trains."""
    import anndata as ad
    import pandas as pd

    from sc_flow.backends.torch.jaxbridge import CellFlowFMObjective, JaxParamModule, iter_fm_batches
    from sc_flow.backends.torch.training._harness import SCFlowLightningModule
    from sc_flow.data import FlowSpec
    from sc_flow.data._encoders import lookup
    from sc_flow.data.schemas import ConditionDataSchema, StateDataSchema

    rng = np.random.default_rng(0)
    n = 64
    drugs = ["drug_a", "drug_b"]
    obs = pd.DataFrame(
        {
            "cell_type": rng.choice(["cl_a", "cl_b"], n),
            "drug1": rng.choice(drugs, n),
            "control": rng.choice([True, False], n),
        }
    )
    obs.loc[rng.choice(n, n // 4, replace=False), "drug1"] = "control"
    obs["control"] = obs["drug1"] == "control"
    for c in obs.columns:
        obs[c] = obs[c].astype("category") if c in ("drug1", "cell_type") else obs[c]

    adata = ad.AnnData(X=rng.random((n, D)).astype(np.float32), obs=obs)
    adata.uns["drug"] = {d: rng.standard_normal((1, COND_DIM)).astype(np.float32) for d in obs["drug1"].cat.categories}

    spec = FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
    )
    loader = spec.build_loader(adata, batch_size=8, to=None)

    adapted_loader = iter_fm_batches(loader, condition_embedding_dim=EMB, seed=42)
    # The loader is infinite, so fetch a finite list of batches
    batches = [next(adapted_loader) for _ in range(5)]
    batch = batches[0]

    assert set(batch) == {"source", "target", "time", "encoder_noise", "conditions"}
    assert batch["source"].shape == (8, D)
    assert batch["target"].shape == (8, D)
    assert batch["time"].shape == (8, 1)
    assert batch["encoder_noise"].shape == (8, EMB)
    assert "drug" in batch["conditions"]
    assert batch["conditions"]["drug"].shape == (1, 1, COND_DIM)

    vf = _build_vf("deterministic")
    pp = ConstantNoiseFlow(sigma=0.0)
    params = _init_params(vf)

    model = JaxParamModule(params)
    objective = CellFlowFMObjective(vf, pp, seed=42)

    loss, logs = objective.compute_loss(model, batch)
    assert loss is not None
    assert "loss" in logs
    assert loss.item() > 0

    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()

    class _AdaptedDataset(IterableDataset):
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            yield from self.batches

    harness = SCFlowLightningModule(model, objective, lr=1e-2)
    trainer = pl.Trainer(
        max_steps=5,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(harness, DataLoader(_AdaptedDataset(batches), batch_size=None))
