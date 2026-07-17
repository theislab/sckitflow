import numpy as np
import pandas as pd
import pytest
import anndata as ad
from sc_flow import FlowMatching
from sc_flow.data import FlowSpec
from sc_flow.data.schemas import ConditionDataSchema, StateDataSchema
from sc_flow.data._encoders import lookup

def test_flow_matching_fit_and_predict():
    """Verify that FlowMatching fits on a toy adata and translates cells."""
    rng = np.random.default_rng(0)
    n = 64
    d = 5
    cond_dim = 4
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

    adata = ad.AnnData(X=rng.random((n, d)).astype(np.float32), obs=obs)
    adata.uns["drug"] = {d: rng.standard_normal((1, cond_dim)).astype(np.float32) for d in obs["drug1"].cat.categories}

    spec = FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
    )

    model = FlowMatching(
        spec=spec,
        condition_embedding_dim=8,
        hidden_dims=(16, 16),
        condition_mode="deterministic",
    )

    # Fit model for 5 steps
    model.fit(
        adata,
        rep_tables=adata.uns,
        batch_size=8,
        n_train_steps=5,
        device="cpu",
    )

    assert model.model is not None
    assert model.vf is not None

    # Predict translation
    x_source = rng.random((10, d)).astype(np.float32)
    # Target condition leaf: (cell_type, drug1)
    leaf = ("cl_a", "drug_a")
    x_pred = model.predict(x_source, leaf, device="cpu", num_steps=5)
    
    assert x_pred.shape == (10, d)
    assert not np.isnan(x_pred).any()
