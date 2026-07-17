"""Parity: sc_flow's obs-only ``compile_obs`` vs cellflow's ``build_annbatch_training``.

Oracle = cellflow's streaming-prepare (itself pinned to the in-memory ``DataManager``).
We assert our composed-schema, labels-only compile produces the same dagloader topology
(pert/ctrl leaves + ``Bind.common``) and the same per-leaf condition embeddings across
categorical single / combination / one-hot conditions and tiled sample covariates.

The ``builds``/``embeds`` flags gate cases not yet supported (as ``xfail``); currently all
listed cases pass. Continuous "paired" covariates (per-cell ``obsm``) are out of scope here
by design — they're streamed as extra Node keys, not a ``condition_fn`` lookup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cellflow")
pytest.importorskip("annbatch")
pytest.importorskip("dagloader")

import anndata as ad
from cellflow.data._annbatch import build_annbatch_training

from sc_flow.data._compile_obs import compile_obs
from sc_flow.data.schemas._condition_data_schema import ConditionDataSchema
from sc_flow.data.schemas._groups_data_schema import GroupsDataSchema
from sc_flow.data.schemas._state_data_schema import StateDataSchema


def _make_adata(seed: int = 0) -> ad.AnnData:
    """cellflow-compatible toy: obs covariates + uns embedding tables (drug, cell_type)."""
    rng = np.random.default_rng(seed)
    n = 400
    drugs = ["drug_a", "drug_b", "drug_c"]
    obs = pd.DataFrame(
        {
            "cell_type": rng.choice(["cl_a", "cl_b", "cl_c"], n),
            "drug1": rng.choice(drugs, n),
            "drug2": rng.choice(drugs, n),
            "dosage_a": rng.choice([10.0, 100.0, 1000.0], n),
        }
    )
    ctrl = rng.choice(n, n // 10, replace=False)
    for c in ("drug1", "drug2"):
        obs.loc[ctrl, c] = "control"
    for c in obs.columns:
        obs[c] = obs[c].astype("category")
    obs["control"] = obs["drug1"] == "control"
    adata = ad.AnnData(X=rng.random((n, 12)).astype(np.float32), obs=obs)
    adata.uns["drug"] = {d: rng.standard_normal((1, 5)).astype(np.float32) for d in obs["drug1"].cat.categories}
    adata.uns["cell_type"] = {c: rng.standard_normal((1, 3)).astype(np.float32) for c in obs["cell_type"].cat.categories}
    return adata


# (pert, reps, split, samp, samp_reps, builds, embeds, reason)
#   builds = compile_obs can construct the schemas at all
#   embeds = the per-leaf condition embedding matches cellflow
SPECS = [
    pytest.param({"drug": ["drug1"]}, {"drug": "drug"}, ["cell_type"], [], {}, True, True, "", id="single-drug-rep-split"),
    pytest.param({"drug": ["drug1"]}, {"drug": "drug"}, [], [], {}, True, True, "", id="single-drug-rep-nosplit"),
    pytest.param({"drug": ["drug1"]}, {}, ["cell_type"], [], {}, True, True, "", id="single-drug-onehot"),
    # single multi-column level (a drug combination) — extract_reps stacks the slots.
    # cellflow requires all perturbation levels to share one column count, so this is the
    # general combination case; no cross-level padding is possible (unequal lengths are rejected).
    pytest.param({"drug": ["drug1", "drug2"]}, {"drug": "drug"}, ["cell_type"], [], {}, True, True, "", id="combo-len2"),
    pytest.param(
        {"drug": ["drug1"]}, {"drug": "drug"}, [], ["cell_type"], {"cell_type": "cell_type"}, True, True, "",
        id="sample-covar",
    ),
]


def _build_ref(adata, pert, reps, split, samp, samp_reps):
    return build_annbatch_training(
        data=adata,
        sample_rep="X",
        control_key="control",
        perturbation_covariates=pert,
        perturbation_covariate_reps=reps or None,
        split_covariates=split,
        sample_covariates=samp,
        sample_covariate_reps=samp_reps or None,
        rep_dict=adata.uns,
    )


def _compile_ours(adata, pert, reps, split, samp, samp_reps):
    # cellflow infers one-hot from a missing rep; our schema requires it declared — so make it
    # explicit for every perturbation level that lacks a rep.
    encoding = {level: "one-hot" for level in pert if level not in reps}
    return compile_obs(
        adata,
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions=pert, conditions_reps=reps, conditions_encoding=encoding),
        groups=GroupsDataSchema(groups=samp, groups_reps=samp_reps) if samp else None,
        control_key="control",
        match_context=split,
    )


@pytest.mark.parametrize(("pert", "reps", "split", "samp", "samp_reps", "builds", "embeds", "reason"), SPECS)
class TestCompileObsParity:
    def test_topology_parity(self, pert, reps, split, samp, samp_reps, builds, embeds, reason):
        """binded layer: pert/ctrl leaves + Bind.common match cellflow's scheme."""
        if not builds:
            pytest.xfail(reason)
        adata = _make_adata()
        ref = _build_ref(adata, pert, reps, split, samp, samp_reps)
        ours = _compile_ours(adata, pert, reps, split, samp, samp_reps)

        def leaves(node):
            return {tuple(map(str, k)) for k in node.weights}

        assert leaves(ours.scheme.nodes["pert"]) == leaves(ref.scheme.nodes["pert"])
        assert leaves(ours.scheme.nodes["ctrl"]) == leaves(ref.scheme.nodes["ctrl"])
        assert ours.scheme.binds[0].common == ref.scheme.binds[0].common

    def test_condition_embedding_parity(self, pert, reps, split, samp, samp_reps, builds, embeds, reason):
        """encoder layer: our condition_fn(leaf) == cellflow's, per group, per leaf."""
        if not (builds and embeds):
            pytest.xfail(reason)
        adata = _make_adata()
        ref = _build_ref(adata, pert, reps, split, samp, samp_reps)
        ours = _compile_ours(adata, pert, reps, split, samp, samp_reps)

        for leaf in ref.scheme.nodes["pert"].weights:  # same cols ordering both sides
            ours_emb = ours.condition_fn(leaf)
            ref_emb = ref.condition_fn(leaf)
            assert set(ours_emb) == set(ref_emb)
            for group in ref_emb:
                np.testing.assert_array_equal(ours_emb[group], ref_emb[group])
