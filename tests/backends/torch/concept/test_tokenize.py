import numpy as np
import torch

from sc_flow.concept import PAD_TOKEN, GeneVocab, TwoViewCollate, rank_encode


def test_rank_encode_orders_by_expression_and_truncates():
    tokens = np.array([10, 11, 12])
    counts = np.array([1.0, 5.0, 3.0])
    out = rank_encode(tokens, counts, max_tokens=2)
    assert out.tolist() == [11, 12]  # highest-count genes first, top-2 kept
    assert rank_encode(np.array([], dtype=np.int64), np.array([]), 5).tolist() == []


def _fixture():
    vocab = GeneVocab(["G0", "G1", "G2", "G3", "G4"])  # tokens 2..6
    var_ids = ["G2", "GX", "G0", "G4", "G1", "GY"]  # GX, GY are out-of-vocab
    var_token = vocab.align(var_ids)  # [4, -1, 2, 6, 3, -1]
    return vocab, var_token


def test_collate_shapes_dtypes_and_excludes_unmapped_and_zero():
    vocab, var_token = _fixture()
    # cell0 expresses G2=5, GX=9 (unmapped), G0=3, G4=0 (zero), G1=1, GY=2 (unmapped)
    # -> in-vocab expressed genes are {G2:tok4, G0:tok2, G1:tok3}
    # cell1 is all-zero -> no genes
    x = np.array([[5, 9, 3, 0, 1, 2], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    batch = TwoViewCollate(var_token, max_tokens=10, seed=0)(x)

    assert batch["tokens_1"].dtype == torch.int64 and batch["pad_mask_1"].dtype == torch.bool
    assert batch["tokens_1"].shape[0] == batch["tokens_2"].shape[0] == 2

    def nonpad(tok_row, mask_row):
        return set(tok_row[~mask_row].tolist())

    v1 = nonpad(batch["tokens_1"][0], batch["pad_mask_1"][0])
    v2 = nonpad(batch["tokens_2"][0], batch["pad_mask_2"][0])
    assert v1 & v2 == set()  # two views are disjoint
    assert v1 | v2 == {2, 3, 4}  # all (and only) in-vocab expressed genes; token 6 (G4, zero) excluded

    # the all-zero cell is entirely padding in both views
    assert bool(batch["pad_mask_1"][1].all()) and bool(batch["pad_mask_2"][1].all())


def test_padding_positions_hold_pad_token():
    _, var_token = _fixture()
    x = np.array([[5, 9, 3, 0, 1, 2], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    batch = TwoViewCollate(var_token, max_tokens=10, seed=1)(x)
    for tk, mk in (("tokens_1", "pad_mask_1"), ("tokens_2", "pad_mask_2")):
        tokens, mask = batch[tk], batch[mk]
        assert torch.all(tokens[mask] == PAD_TOKEN)  # padded slots hold PAD
        assert torch.all(tokens[~mask] != PAD_TOKEN)  # real slots never do


def test_nvars_mismatch_raises():
    _, var_token = _fixture()
    collate = TwoViewCollate(var_token)
    import pytest

    with pytest.raises(ValueError, match="n_vars mismatch"):
        collate(np.zeros((2, 3), dtype=np.float32))
