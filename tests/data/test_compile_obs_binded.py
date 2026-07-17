"""End-to-end: ``compile_obs`` (labels only) → ``binded.Loader`` → a streamed batch.

Proves the data layer speaks binded's vocab and holds no cell arrays: ``compile_obs`` reads
``obs``/``uns`` only and returns a :class:`binded.Scheme` + a per-leaf ``condition_fn``; binded
streams ``{source, target, condition}`` from the source ``adata`` at iteration time.

Skips where binded (and its annbatch ``BoundClassSampler`` fork) is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("binded")  # needs the annbatch BoundClassSampler fork too

import anndata as ad

from sc_flow.data import compile_obs
from sc_flow.data._encoders import lookup
from sc_flow.data.schemas import ConditionDataSchema, CovariatesDataSchema, StateDataSchema


def _make_adata(seed: int = 0) -> ad.AnnData:
    """Tiny in-memory toy: obs covariates + uns embedding tables (drug, cell_type)."""
    rng = np.random.default_rng(seed)
    n = 400
    drugs = ["drug_a", "drug_b", "drug_c"]
    obs = pd.DataFrame(
        {
            "cell_type": rng.choice(["cl_a", "cl_b", "cl_c"], n),
            "drug1": rng.choice(drugs, n),
        }
    )
    ctrl = rng.choice(n, n // 10, replace=False)
    obs.loc[ctrl, "drug1"] = "control"
    for c in obs.columns:
        obs[c] = obs[c].astype("category")
    obs["control"] = obs["drug1"] == "control"
    adata = ad.AnnData(X=rng.random((n, 12)).astype(np.float32), obs=obs)
    adata.uns["drug"] = {d: rng.standard_normal((1, 5)).astype(np.float32) for d in obs["drug1"].cat.categories}
    adata.uns["cell_type"] = {c: rng.standard_normal((1, 3)).astype(np.float32) for c in obs["cell_type"].cat.categories}
    return adata


def _compile(adata: ad.AnnData):
    return compile_obs(
        adata,
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        covariates=CovariatesDataSchema(covariate_encoders={"cell_type": lookup("cell_type")}),
        control_key="control",
        match_context=["cell_type"],
    )


def test_compile_obs_produces_binded_scheme():
    """compile_obs returns a binded Scheme (pert/ctrl nodes bound on the split context)."""
    from binded import Scheme

    compiled = _compile(_make_adata())
    assert isinstance(compiled.scheme, Scheme)
    assert compiled.cols == ("cell_type", "drug1")
    assert set(compiled.scheme.nodes) == {"pert", "ctrl"}
    assert compiled.scheme.root == "pert"
    assert compiled.scheme.binds[0].common == ("cell_type",)
    # the rep the root streams is the state's sample_rep as a binded loc-string
    assert compiled.scheme.nodes["pert"].keys == ("X",)


def test_condition_fn_is_per_leaf_not_dataset_wide():
    """condition_fn maps ONE leaf → its reps dict; it never materializes a dataset-wide array."""
    compiled = _compile(_make_adata())
    leaf = next(iter(compiled.scheme.nodes["pert"].weights))  # a (cell_type, drug1) tuple
    reps = compiled.condition_fn(leaf)
    assert set(reps) == {"drug", "cell_type"}
    # one value per leaf: leading dim is 1 (single condition), not n_obs
    assert reps["drug"].shape[0] == 1
    assert reps["cell_type"].shape[0] == 1
    assert reps["drug"].shape[-1] == 5  # the drug embedding dim from uns


def test_condition_fn_matches_raw_uns_lookup():
    """The unified Lookup encoder reproduces the raw ``uns[key][value]`` embedding, per leaf."""
    adata = _make_adata()
    compiled = _compile(adata)
    assert compiled.cols == ("cell_type", "drug1")  # leaf tuple order
    for leaf in compiled.scheme.nodes["pert"].weights:
        cell_type, drug = leaf
        reps = compiled.condition_fn(leaf)
        np.testing.assert_array_equal(reps["drug"][0, 0], np.asarray(adata.uns["drug"][drug]).reshape(-1))
        np.testing.assert_array_equal(
            reps["cell_type"][0, 0], np.asarray(adata.uns["cell_type"][cell_type]).reshape(-1)
        )


def test_loader_streams_source_target_condition():
    """binded.Loader consumes the compiled scheme + condition_fn and yields aligned batches."""
    from binded import Loader, SamplerConfig

    adata = _make_adata()
    compiled = _compile(adata)
    cfg = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8, to=None)
    loader = Loader(compiled.scheme, cfg, compiled.condition_fn)

    batch = next(iter(loader))
    assert set(batch) >= {"source", "target", "condition"}
    assert np.asarray(batch["target"]).shape == (8, adata.n_vars)
    assert np.asarray(batch["source"]).shape == (8, adata.n_vars)
    cond = batch["condition"]
    assert isinstance(cond, dict) and set(cond) == {"drug", "cell_type"}


def test_array_holding_containers_are_gone():
    """The dataset-wide array holders were removed; only label containers survive."""
    from sc_flow.data import containers

    assert containers.__all__ == ["BaseData", "CategoricalData"]
    with pytest.raises(ImportError):
        from sc_flow.data.containers import StateData  # noqa: F401
    with pytest.raises(ImportError):
        from sc_flow.data.containers import MixedTypeData  # noqa: F401
