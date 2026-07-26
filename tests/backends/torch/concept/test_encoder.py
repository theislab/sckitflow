import torch

from sc_flow.concept import PAD_TOKEN, GeneEncoderConfig


def _small_encoder(n_tokens: int = 40):
    return GeneEncoderConfig(
        n_tokens=n_tokens, dim_model=32, n_layers=2, n_heads=4, dim_feedforward=64, dropout=0.0, max_rank=64
    ).build()


def test_forward_shape_and_finite():
    enc = _small_encoder().eval()
    tokens = torch.randint(PAD_TOKEN + 1, 40, (5, 7))
    out = enc(tokens)
    assert out.shape == (5, 32)
    assert torch.isfinite(out).all()


def test_config_is_a_portable_component():
    from sc_flow._registry import parse

    cfg = GeneEncoderConfig(n_tokens=40, dim_model=16, activation="relu")  # activation is a string id
    spec = cfg.to_spec()
    assert spec["type"] == "sc_flow.gene_encoder" and spec["version"] == 1
    back = parse(spec)  # dispatched by the registry
    assert back == cfg
    assert back.build().dim_model == 16


def test_padding_is_invisible():
    # A masked-and-padded sequence must give the same CLS as the unpadded one.
    enc = _small_encoder().eval()
    real = torch.tensor([[3, 4, 5, 6]])
    cls_unpadded = enc(real)
    padded = torch.tensor([[3, 4, 5, 6, PAD_TOKEN, PAD_TOKEN]])
    mask = torch.tensor([[False, False, False, False, True, True]])
    cls_padded = enc(padded, mask)
    assert torch.allclose(cls_unpadded, cls_padded, atol=1e-4)


def test_gradients_flow():
    enc = _small_encoder()
    enc(torch.randint(1, 40, (4, 6))).sum().backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
