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

from sc_flow.data import FlowSpec, compile_obs
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
    adata.uns["cell_type"] = {
        c: rng.standard_normal((1, 3)).astype(np.float32) for c in obs["cell_type"].cat.categories
    }
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


def _fragmented_adata() -> ad.AnnData:
    """A perturbed leaf split into two contiguous runs (40 then 10) so total != shortest-run.

    Layout (disk order): (cl_a, drug_a)×40, (cl_a, drug_b)×40, (cl_a, drug_a)×10, controls×40. So drug_a
    has runs [40, 10] (total 50, shortest 10) and drug_b has one run [40].
    """
    rng = np.random.default_rng(0)
    rows: list[tuple[str, str]] = []
    for cl, dr, n in [("cl_a", "drug_a", 40), ("cl_a", "drug_b", 40), ("cl_a", "drug_a", 10),
                      ("cl_a", "control", 40)]:
        rows += [(cl, dr)] * n
    obs = pd.DataFrame(rows, columns=["cell_type", "drug1"])
    obs["control"] = obs["drug1"] == "control"
    for c in ("cell_type", "drug1"):
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=rng.random((len(obs), 6)).astype(np.float32), obs=obs)
    adata.uns["drug"] = {d: rng.standard_normal((1, 4)).astype(np.float32) for d in obs["drug1"].cat.categories}
    return adata


def _pert_leaves(adata: ad.AnnData, min_runs: int) -> set:
    compiled = compile_obs(
        adata,
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
        min_runs_per_leaf=min_runs,
    )
    return set(compiled.scheme.nodes["pert"].weights)


def test_min_runs_per_leaf_filters_on_shortest_run_not_total():
    """min_runs_per_leaf excludes a leaf by its SHORTEST contiguous run, not its total count.

    drug_a is fragmented into runs [40, 10] (total 50); at threshold 16 it must be dropped (its 10-run is
    < 16) even though its total 50 >= 16 — the old total-count filter wrongly kept it, which then broke
    chunk_size=16 downstream.
    """
    adata = _fragmented_adata()
    both = {("cl_a", "drug_a"), ("cl_a", "drug_b")}
    assert _pert_leaves(adata, 0) == both           # no filtering
    assert _pert_leaves(adata, 8) == both            # drug_a shortest run 10 >= 8 -> kept
    assert _pert_leaves(adata, 16) == {("cl_a", "drug_b")}  # drug_a 10-run < 16 -> dropped (total 50 ignored)


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


def test_coupling_reps_stream_as_aligned_node_keys():
    """A coupling ref in a different space than the state rep streams as an extra aligned Node key."""
    from binded import Loader, SamplerConfig

    from sc_flow.data.schemas import CouplingDataSchema

    adata = _make_adata()
    rng = np.random.default_rng(1)
    adata.obsm["X_lin"] = rng.random((adata.n_obs, 6)).astype(np.float32)  # coupling space != state "X"
    compiled = compile_obs(
        adata,
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        covariates=CovariatesDataSchema(covariate_encoders={"cell_type": lookup("cell_type")}),
        coupling=CouplingDataSchema(src_lin="X_lin", tgt_lin="X_lin"),
        control_key="control",
        match_context=["cell_type"],
    )
    # role -> streamed loc recorded, and both nodes carry the extra aligned key
    assert compiled.coupling == {"src_lin": "obsm/X_lin", "tgt_lin": "obsm/X_lin"}
    assert compiled.scheme.nodes["pert"].keys == ("X", "obsm/X_lin")
    assert compiled.scheme.nodes["ctrl"].keys == ("X", "obsm/X_lin")

    cfg = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8, to=None)
    batch = next(iter(Loader(compiled.scheme, cfg, compiled.condition_fn)))
    assert "obsm/X_lin" in batch["target_reps"] and "obsm/X_lin" in batch["source_reps"]
    assert np.asarray(batch["target_reps"]["obsm/X_lin"]).shape == (8, 6)
    assert np.asarray(batch["source_reps"]["obsm/X_lin"]).shape == (8, 6)


def test_coupling_same_space_as_state_is_not_duplicated():
    """When a coupling ref IS the state rep, it is not streamed twice — source/target are reused."""
    from sc_flow.data.schemas import CouplingDataSchema

    adata = _make_adata()
    compiled = compile_obs(
        adata,
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        coupling=CouplingDataSchema(src_lin="X", tgt_lin="X"),  # == state rep
        control_key="control",
        match_context=["cell_type"],
    )
    assert compiled.coupling == {"src_lin": "X", "tgt_lin": "X"}
    assert compiled.scheme.nodes["pert"].keys == ("X",)  # deduped — no extra stream
    assert compiled.scheme.nodes["ctrl"].keys == ("X",)


def _spec() -> FlowSpec:
    return FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        covariates=CovariatesDataSchema(covariate_encoders={"cell_type": lookup("cell_type")}),
        control_key="control",
        match_context=["cell_type"],
    )


def test_flowspec_build_loader_from_zarr_path_with_explicit_rep_tables(tmp_path):
    """FlowSpec.build_loader streams from a zarr PATH; lookup tables are passed via explicit rep_tables."""
    adata = _make_adata()
    path = tmp_path / "adata.zarr"
    adata.write_zarr(path)

    loader = _spec().build_loader(str(path), rep_tables=adata.uns, batch_size=8, to=None)
    batch = next(iter(loader))
    assert set(batch) >= {"source", "target", "condition"}
    assert np.asarray(batch["target"]).shape == (8, adata.n_vars)
    assert isinstance(batch["condition"], dict) and set(batch["condition"]) == {"drug", "cell_type"}


def test_out_of_core_scheme_matches_in_memory(tmp_path):
    """Compiling from a zarr PATH (out-of-core) yields the same nodes + weights as in-memory."""
    adata = _make_adata()
    path = tmp_path / "adata.zarr"
    adata.write_zarr(path)

    def w(compiled, node):
        return {tuple(map(str, k)): float(v) for k, v in compiled.scheme.nodes[node].weights.items()}

    kw = {
        "state": StateDataSchema(sample_rep="X"),
        "condition": ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        "covariates": CovariatesDataSchema(covariate_encoders={"cell_type": lookup("cell_type")}),
        "control_key": "control",
        "match_context": ["cell_type"],
    }
    mem = compile_obs(adata, **kw)  # in-memory (rep_tables defaults to .uns)
    disk = compile_obs(str(path), rep_tables=adata.uns, **kw)  # out-of-core (explicit rep_tables)

    assert mem.cols == disk.cols
    assert w(mem, "pert") == w(disk, "pert")
    assert w(mem, "ctrl") == w(disk, "ctrl")
    assert mem.scheme.nodes["pert"].keys == disk.scheme.nodes["pert"].keys


def test_zarr_path_lookup_without_rep_tables_raises(tmp_path):
    """A path has no in-memory .uns, so a lookup encoder without explicit rep_tables fails loudly."""
    adata = _make_adata()
    path = tmp_path / "adata.zarr"
    adata.write_zarr(path)
    with pytest.raises(KeyError, match="drug"):
        _spec().build_loader(str(path), batch_size=8, to=None)  # no rep_tables → Lookup.fit can't bind


def test_control_path_uses_separate_source():
    """With control_path, controls come from a separate source, matched to targets by match_context only."""
    adata = _make_adata()
    targets = adata[~adata.obs["control"].to_numpy()].copy()
    controls = adata[adata.obs["control"].to_numpy()].copy()
    compiled = compile_obs(
        targets,
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        covariates=CovariatesDataSchema(covariate_encoders={"cell_type": lookup("cell_type")}),
        control_key="control",
        match_context=["cell_type"],
        rep_tables=adata.uns,
        control_path=controls,
    )
    assert set(compiled.scheme.sources) == {"data", "control"}
    assert compiled.scheme.nodes["pert"].source == "data"
    assert compiled.scheme.nodes["ctrl"].source == "control"
    assert compiled.scheme.nodes["ctrl"].cols == ("cell_type",)  # match_context only


def test_min_runs_per_leaf_drops_small_target_leaves_only():
    """min_runs_per_leaf zero-weights small TARGET leaves; controls are never filtered.

    Uses grouped (sorted) obs so each leaf is a single contiguous run — there shortest-run == total, so a
    median-of-totals threshold drops the smaller half (the per-run filter reduces to the count filter).
    """
    import collections

    adata = _make_adata()
    # group into contiguous per-leaf runs (a single grouped store); random order would give run-length 1.
    order = adata.obs.sort_values(["cell_type", "drug1"], kind="stable").index
    adata = adata[order].copy()
    kw = {
        "state": StateDataSchema(sample_rep="X"),
        "condition": ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        "control_key": "control",
        "match_context": ["cell_type"],
    }
    # a threshold between the smallest and largest target-leaf count drops some (not all) target leaves
    tgt = adata.obs.loc[~adata.obs["control"].to_numpy(), ["cell_type", "drug1"]]
    counts = sorted(collections.Counter(map(tuple, tgt.to_numpy())).values())
    threshold = counts[len(counts) // 2]  # median → drops the smaller half

    base = compile_obs(adata, **kw)  # default 0 → keep all
    filt = compile_obs(adata, **kw, min_runs_per_leaf=threshold)

    n_all = len(base.scheme.nodes["pert"].weights)
    n_kept = len(filt.scheme.nodes["pert"].weights)
    assert 0 < n_kept < n_all  # partial drop
    # controls untouched
    assert filt.scheme.nodes["ctrl"].weights == base.scheme.nodes["ctrl"].weights


def test_control_path_requires_match_context():
    adata = _make_adata()
    with pytest.raises(ValueError, match="match_context"):
        compile_obs(
            adata,
            state=StateDataSchema(sample_rep="X"),
            condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
            control_key="control",
            match_context=[],
            rep_tables=adata.uns,
            control_path=adata,
        )


def test_array_holding_containers_are_gone():
    """The dataset-wide array holders were removed; only label containers survive."""
    from sc_flow.data import containers

    assert containers.__all__ == ["BaseData", "CategoricalData"]
    with pytest.raises(ImportError):
        from sc_flow.data.containers import StateData  # noqa: F401
    with pytest.raises(ImportError):
        from sc_flow.data.containers import MixedTypeData  # noqa: F401
