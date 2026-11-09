"""sc_flow.group_share leaf weighting, and lookup_fallback (rep-table vector when a value has one, else a learned twin row)."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from sc_flow._model import _read_problem, _SemanticBatches
from sc_flow._weighting import build_weighting


def test_group_share_splits_shares_within_groups():
    """Named groups get their share spread evenly over their leaves; every other leaf shares ``rest``."""
    leaves = [("tahoe", "d1"), ("tahoe", "d2"), ("sciplex", "d1"), ("gse", "d3")]
    spec = {"type": "sc_flow.group_share", "version": 1, "config": {"by": ["dataset"], "shares": {"tahoe": 0.7}, "rest": 0.3}}
    w = build_weighting(spec, leaves=leaves, group_by=("dataset", "drug"))
    assert w[("tahoe", "d1")] == pytest.approx(0.35) and w[("tahoe", "d2")] == pytest.approx(0.35)
    assert w[("sciplex", "d1")] == pytest.approx(0.15) and w[("gse", "d3")] == pytest.approx(0.15)
    assert sum(w.values()) == pytest.approx(1.0)


@pytest.fixture
def store(tmp_path):
    """One line, two drugs with UniPert vectors (d1, control), one gene without (G1), controls."""
    rng = np.random.default_rng(0)
    p1 = ["d1"] * 30 + ["G1"] * 30 + ["control"] * 40
    obs = pd.DataFrame({"cell_line_ds": "ds:L1", "perturbation_1": p1, "perturbation_2": "control",
                        "is_control": [p == "control" for p in p1]}, index=[f"c{i}" for i in range(len(p1))])
    for c in ("cell_line_ds", "perturbation_1", "perturbation_2"):
        obs[c] = obs[c].astype("category")
    uni = pd.DataFrame(rng.normal(size=(2, 4)).astype(np.float32), index=["d1", "control"], columns=list("abcd"))
    a = ad.AnnData(X=np.zeros((len(p1), 3), np.float32), obs=obs, obsm={"X_rep": rng.normal(size=(len(p1), 5)).astype(np.float32)},
                   uns={"unipert_emb": uni})
    path = str(tmp_path / "s.zarr")
    a.write_zarr(path)
    return path


def _problem(path, fallback):
    return _read_problem(path, sample_rep="X_rep", control_key="is_control",
                         perturbation_covariates={"drug": ["perturbation_1", "perturbation_2"]}, split_covariates=["cell_line_ds"],
                         perturbation_covariate_reps={"drug": "unipert_emb"}, lookup_fallback=fallback)


def test_missing_vector_still_errors_without_fallback(store):
    """Default behaviour is unchanged: a value with no rep vector is an error."""
    with pytest.raises(KeyError, match="no vector"):
        _problem(store, ())


def test_lookup_fallback_routes_each_slot(store):
    """The twin is a categorical realm over the same columns; each slot pools exactly one of (vector, twin row)."""
    p = _problem(store, ["drug"])
    assert "drug_lookup" in p.dims.condition_num_categories and p.dims.condition["drug"] == 4
    leaf = {lf[1]: lf for lf in p.pert_leaves}
    gene, drug = p.condition_masks[leaf["G1"]], p.condition_masks[leaf["d1"]]
    assert gene["drug"].tolist() == [[False, True]] and gene["drug_lookup"].tolist() == [[True, False]]
    assert drug["drug"].tolist() == [[True, True]] and drug["drug_lookup"].tolist() == [[False, False]]
    assert np.all(p.label_lookup[leaf["G1"]]["drug"][0, 0] == 0)


def test_semantic_batches_attach_masks():
    """A batch's mask is found by its label dict's identity and travels as ``condition_mask``."""
    lab = {"drug": np.zeros((1, 2))}
    mask = {"drug": np.ones((1, 2), dtype=bool)}
    t = torch.zeros(4, 5)
    batch = {"primary": {"X_rep": t}, "ctrl": {"X_rep": t}, "labels": {"primary": lab}}
    out = next(iter(_SemanticBatches([batch], "X_rep", {id(lab): mask})))
    assert out["condition"] is lab and out["condition_mask"] is mask
    assert "condition_mask" not in next(iter(_SemanticBatches([batch], "X_rep")))
