import numpy as np

from sc_flow.concept import CLS_TOKEN, NUM_SPECIAL, PAD_TOKEN, GeneVocab


def test_token_layout():
    v = GeneVocab(["ENSG1", "ENSG2", "ENSG3"])
    assert (PAD_TOKEN, CLS_TOKEN, NUM_SPECIAL) == (0, 1, 2)
    assert v.n_genes == 3
    assert v.n_tokens == 3 + NUM_SPECIAL == len(v)
    # genes are numbered from NUM_SPECIAL in vocabulary order
    assert v.token_of("ENSG1") == NUM_SPECIAL
    assert v.token_of("ENSG3") == NUM_SPECIAL + 2
    assert v.token_of("ENSG_absent") == -1


def test_align_drops_unmapped_and_follows_var_order():
    v = GeneVocab(["ENSG1", "ENSG2", "ENSG3"])
    tok = v.align(["ENSG3", "ENSG_absent", "ENSG1"])
    assert tok.dtype == np.int64
    assert tok.tolist() == [v.token_of("ENSG3"), -1, v.token_of("ENSG1")]


def test_version_stripping_and_case_insensitivity():
    v = GeneVocab(["ENSG00000139618.15", "ensg00000141510"])
    assert v.token_of("ENSG00000139618") == NUM_SPECIAL  # query without version matches
    assert v.token_of("ENSG00000141510.3") == NUM_SPECIAL + 1  # query with version matches
    assert v.token_of("Ensg00000139618.99") == NUM_SPECIAL  # case + differing version


def test_dedupe_preserves_first_order():
    v = GeneVocab(["ENSG1", "ENSG1", "ENSG2"])
    assert v.n_genes == 2
    assert v.gene_ids == ("ENSG1", "ENSG2")


def test_strip_version_off_is_exact():
    v = GeneVocab(["ENSG1.2"], strip_version=False)
    assert v.token_of("ENSG1.2") == NUM_SPECIAL
    assert v.token_of("ENSG1") == -1
