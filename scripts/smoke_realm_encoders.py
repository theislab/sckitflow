"""Smoke: the per-realm condition-encoder registry (model-side index|feature -> embedding).

Verifies the discriminated-spec family that replaces the old fixed compile-time encoding:
  * ``sc_flow.embedding`` : (batch, set) int  -> (batch, set, output_dim)   [learned]
  * ``sc_flow.onehot``    : (batch, set) int  -> (batch, set, num_categories) [fixed]
  * ``sc_flow.feature_mlp``: (batch, set, in) -> (batch, set, output_dim)     [learned]
  * embedding + onehot consume the SAME data-side input (an index) — only feature_mlp takes a vector,
    which is why adding an LLM encoder (index-consuming) needs no data-side change.
  * realm_output_dim() reports the post-encoder width without building.

    python scripts/smoke_realm_encoders.py
"""

from __future__ import annotations

import torch

from sc_flow.flow._realm_encoders import (
    REALM_ENCODER_REGISTRY,
    build_realm_encoder,
    realm_output_dim,
    validate_realm_encoder_spec,
)

B, S = 3, 2  # batch, set (combinatorial slots)


def _spec(type_id, **config):
    return {"type": type_id, "version": 1, "config": config}


def main() -> None:
    assert REALM_ENCODER_REGISTRY.types == ["sc_flow.embedding", "sc_flow.feature_mlp", "sc_flow.onehot"]

    # learned embedding: index -> vector
    emb_spec = _spec("sc_flow.embedding", num_categories=10, output_dim=8)
    emb = build_realm_encoder(emb_spec)
    idx = torch.randint(0, 10, (B, S))
    assert emb(idx).shape == (B, S, 8)
    assert realm_output_dim(emb_spec) == 8
    print("[ok] embedding: (B,S) int -> (B,S,8), learned weights:", sum(p.numel() for p in emb.parameters()) > 0)

    # fixed one-hot: same index input, output_dim == num_categories, no weights
    oh_spec = _spec("sc_flow.onehot", num_categories=5)
    oh = build_realm_encoder(oh_spec)
    assert oh(torch.randint(0, 5, (B, S))).shape == (B, S, 5)
    assert realm_output_dim(oh_spec) == 5
    assert sum(p.numel() for p in oh.parameters()) == 0
    print("[ok] onehot: (B,S) int -> (B,S,5), weightless; same input kind as embedding")

    # feature projection: vector -> vector (learned)
    fm_spec = _spec("sc_flow.feature_mlp", input_dim=16, output_dim=8, mlp_kwargs={"hidden_dims": [12]})
    fm = build_realm_encoder(fm_spec)
    assert fm(torch.randn(B, S, 16)).shape == (B, S, 8)
    assert realm_output_dim(fm_spec) == 8
    print("[ok] feature_mlp: (B,S,16) -> (B,S,8), learned")

    # validation is strict + canonical (round-trips through the registry)
    canon = validate_realm_encoder_spec(emb_spec)
    assert canon == emb_spec, canon
    try:
        validate_realm_encoder_spec(_spec("sc_flow.embedding", num_categories=0, output_dim=8))
        raise AssertionError("should reject num_categories=0")
    except ValueError:
        pass
    print("[ok] spec validation strict + canonical")

    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
