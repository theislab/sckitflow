import torch

from sc_flow.concept import (
    NUM_SPECIAL,
    ContrastiveHead,
    ContrastiveModel,
    ContrastiveObjective,
    GeneEncoderConfig,
)
from sc_flow.training import OBJECTIVE_REGISTRY, build_objective


def test_registered_in_core_registry():
    assert "concept-clip" in OBJECTIVE_REGISTRY
    assert isinstance(build_objective("concept-clip"), ContrastiveObjective)


def _model(n_tokens: int):
    backbone = GeneEncoderConfig(
        n_tokens=n_tokens, dim_model=32, n_layers=2, n_heads=4, dim_feedforward=64, dropout=0.0, max_rank=32
    ).build()
    return ContrastiveModel(backbone, ContrastiveHead())  # logit_scale lives on the head now


def _identifiable_batch(batch: int = 16, seq: int = 8, n_genes: int = 64, seed: int = 0):
    """Two views per cell sharing ONE identifying gene (random position) amid fresh noise."""
    g = torch.Generator().manual_seed(seed)
    n_tokens = NUM_SPECIAL + n_genes
    identity = NUM_SPECIAL + torch.arange(batch)
    noise_lo = NUM_SPECIAL + batch

    def view() -> torch.Tensor:
        t = torch.randint(noise_lo, n_tokens, (batch, seq), generator=g)
        pos = torch.randint(0, seq, (batch,), generator=g)
        t[torch.arange(batch), pos] = identity
        return t

    return view(), view(), n_tokens


def test_loss_finite_and_logs():
    v1, v2, n_tokens = _identifiable_batch()
    loss, logs = ContrastiveObjective().compute_loss(_model(n_tokens), {"tokens_1": v1, "tokens_2": v2})
    assert torch.isfinite(loss) and loss.ndim == 0
    assert set(logs) >= {"loss", "logit_scale", "retrieval_acc"}


def test_contrastive_learns_cell_identity():
    torch.manual_seed(0)
    v1, v2, n_tokens = _identifiable_batch()
    model = _model(n_tokens)
    obj = ContrastiveObjective()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # trains backbone + head (logit_scale)

    first_loss = None
    logs: dict = {}
    for _ in range(300):
        opt.zero_grad()
        loss, logs = obj.compute_loss(model, {"tokens_1": v1, "tokens_2": v2})
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(loss.detach())

    assert float(loss.detach()) < first_loss
    assert float(logs["retrieval_acc"]) >= 0.9
