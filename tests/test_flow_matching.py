import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from sc_flow import FlowMatching
from sc_flow.data import FlowSpec
from sc_flow.data._encoders import lookup
from sc_flow.data.schemas import ConditionDataSchema, CouplingDataSchema, StateDataSchema


def _toy_adata(n=64, d=5, cond_dim=4, pca=None, seed=0):
    """Toy adata: X + a categorical drug (with an uns embedding table) + control flag."""
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {
            "cell_type": rng.choice(["cl_a", "cl_b"], n),
            "drug1": rng.choice(["drug_a", "drug_b"], n),
            "control": rng.choice([True, False], n),
        }
    )
    obs.loc[rng.choice(n, n // 4, replace=False), "drug1"] = "control"
    obs["control"] = obs["drug1"] == "control"
    for c in ("drug1", "cell_type"):
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=rng.random((n, d)).astype(np.float32), obs=obs)
    if pca is not None:
        adata.obsm["X_pca"] = rng.standard_normal((n, pca)).astype(np.float32)
    adata.uns["drug"] = {
        dd: rng.standard_normal((1, cond_dim)).astype(np.float32) for dd in obs["drug1"].cat.categories
    }
    return adata, rng


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


@pytest.mark.parametrize("match_method", ["sinkhorn", "independent"])
def test_flow_matching_coupling_rep_and_baselines(match_method):
    """OT-match on a coupling rep (``obsm/X_pca``) distinct from the state rep (``X``), + baseline.

    Exercises the ``source_reps``/``target_reps`` branch of the OT step and confirms
    :attr:`CompiledData.dims` carries state + per-realm condition + coupling dims.
    """
    d, pca = 6, 4
    adata, rng = _toy_adata(n=80, d=d, cond_dim=3, pca=pca)
    spec = FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
        coupling=CouplingDataSchema(src_lin="X_pca", tgt_lin="X_pca"),  # OT on X_pca, flow on X
    )

    # dims computed from headers + one condition_fn lookup — no sampler pulled.
    dims = spec.compile(adata, rep_tables=adata.uns).dims
    assert dims.state == d
    assert dims.condition == {"drug": 3}
    assert dims.coupling == {"src_lin": pca, "tgt_lin": pca}

    model = FlowMatching(spec=spec, condition_embedding_dim=8, hidden_dims=(16, 16), match_method=match_method)
    model.fit(adata, rep_tables=adata.uns, batch_size=8, n_train_steps=4, device="cpu")

    x_pred = model.predict(rng.random((7, d)).astype(np.float32), ("cl_a", "drug_a"), device="cpu", num_steps=5)
    assert x_pred.shape == (7, d)
    assert np.isfinite(x_pred).all()


def test_flow_matching_bit_reproducible():
    """A fixed ``seed`` makes fit bit-reproducible: identical weights + identical predictions.

    Covers every stochastic source (VF init, data order, OT plan-sampling, t draw); a different seed
    must diverge.
    """
    d = 5
    adata, _ = _toy_adata(n=64, d=d, cond_dim=4, seed=1)
    spec = FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
    )
    x_probe = np.arange(6 * d, dtype=np.float32).reshape(6, d)  # fixed probe input

    def run(seed):
        model = FlowMatching(spec=spec, condition_embedding_dim=8, hidden_dims=(16, 16), seed=seed)
        model.fit(adata, rep_tables=adata.uns, batch_size=8, n_train_steps=6, device="cpu")
        return model.vf.state_dict(), model.predict(x_probe, ("cl_a", "drug_a"), device="cpu", num_steps=5)

    sd_a, pred_a = run(0)
    sd_b, pred_b = run(0)
    sd_c, pred_c = run(1)

    # same seed -> bit-identical weights and predictions
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"param {k} differs across identical-seed runs"
    assert np.array_equal(pred_a, pred_b)
    # different seed -> different training -> different predictions
    assert not np.array_equal(pred_a, pred_c)


@pytest.mark.parametrize("with_coupling", [False, True])
def test_genot_fit_and_predict(with_coupling):
    """GENOT (``objective="genot"``): flow noise→target conditioned on the source cell.

    Same-space (no coupling schema, source = state) and cross-rep (OT on ``obsm/X_pca``). Asserts the VF
    is built with a source encoder, and that the (stochastic) generative predict is seed-reproducible.
    """
    d, pca = 6, 4
    adata, rng = _toy_adata(n=80, d=d, cond_dim=3, pca=(pca if with_coupling else None))
    coupling = CouplingDataSchema(src_lin="X_pca", tgt_lin="X_pca") if with_coupling else None
    spec = FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
        coupling=coupling,
    )
    model = FlowMatching(spec=spec, objective="genot", condition_embedding_dim=8, hidden_dims=(16, 16))
    model.fit(adata, rep_tables=adata.uns, batch_size=8, n_train_steps=4, device="cpu")
    assert model.vf.use_source_encoder  # GENOT built the source-conditioning encoder

    x = rng.random((7, d)).astype(np.float32)
    p0 = model.predict(x, ("cl_a", "drug_a"), device="cpu", num_steps=5, seed=0)
    p1 = model.predict(x, ("cl_a", "drug_a"), device="cpu", num_steps=5, seed=0)
    p2 = model.predict(x, ("cl_a", "drug_a"), device="cpu", num_steps=5, seed=1)
    assert p0.shape == (7, d)
    assert np.isfinite(p0).all()
    assert np.array_equal(p0, p1)  # generative predict is reproducible given a seed
    assert not np.array_equal(p0, p2)  # a different noise seed -> a different sample


def _conditional_shift_adata(*, n_per=128, d=6, delta=5.0, seed=0):
    """Cells where each drug applies a known **opposite** shift, so a trained model must use the condition.

    control ~ N(mu_ct, 0.3); perturbed(drug) = control-like + shift(drug), shift(drug_a)=+delta,
    shift(drug_b)=-delta, per cell_type context. Returns (adata, controls_by_ct, shift).
    """
    rng = np.random.default_rng(seed)
    cts = {"cl_a": 0.0, "cl_b": 2.0}
    shift = {"drug_a": +delta, "drug_b": -delta}
    xs, ct_col, dr_col, ctrl_col = [], [], [], []
    controls_by_ct = {}
    for ct, mu in cts.items():
        ctrl = rng.normal(mu, 0.3, (n_per, d)).astype(np.float32)
        controls_by_ct[ct] = ctrl
        xs.append(ctrl)
        ct_col += [ct] * n_per
        dr_col += ["control"] * n_per
        ctrl_col += [True] * n_per
        for dr, s in shift.items():
            xs.append((rng.normal(mu, 0.3, (n_per, d)) + s).astype(np.float32))
            ct_col += [ct] * n_per
            dr_col += [dr] * n_per
            ctrl_col += [False] * n_per
    obs = pd.DataFrame({"cell_type": ct_col, "drug1": dr_col, "control": ctrl_col})
    for c in ("cell_type", "drug1"):
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.concatenate(xs, axis=0), obs=obs)
    # distinct (random but fixed) embedding per drug so the condition encoder can separate them
    adata.uns["drug"] = {dd: rng.standard_normal((1, 4)).astype(np.float32) for dd in obs["drug1"].cat.categories}
    return adata, controls_by_ct, shift


@pytest.mark.parametrize("objective", ["otfm", "genot"])
def test_model_learns_conditional_translation(objective):
    """End-to-end learning check: after training, predict moves control cells in the drug's direction.

    Not a unit test — it actually trains and asserts the model learned a *condition-dependent* translation
    (opposite shifts for the two drugs), which requires the coupling, the flow, and the condition encoder to
    all work together.
    """
    d, delta = 6, 5.0
    adata, controls, shift = _conditional_shift_adata(n_per=128, d=d, delta=delta)
    spec = FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
    )
    model = FlowMatching(spec=spec, objective=objective, condition_embedding_dim=16, hidden_dims=(64, 64))
    model.fit(adata, rep_tables=adata.uns, batch_size=64, n_train_steps=200, lr=1e-2, device="cpu")

    x_ctrl = controls["cl_a"]
    move_a = float((model.predict(x_ctrl, ("cl_a", "drug_a"), num_steps=20, seed=0) - x_ctrl).mean())
    move_b = float((model.predict(x_ctrl, ("cl_a", "drug_b"), num_steps=20, seed=0) - x_ctrl).mean())

    # learned the right direction/sign for each drug (true shifts are +delta / -delta) ...
    assert move_a > 0.4 * delta, f"{objective}: drug_a move {move_a:.2f} not toward +{delta}"
    assert move_b < -0.4 * delta, f"{objective}: drug_b move {move_b:.2f} not toward -{delta}"
    # ... and the condition actually matters (the two drugs pull apart).
    assert move_a - move_b > delta, f"{objective}: drugs not separated ({move_a:.2f} vs {move_b:.2f})"


def test_genot_quadratic_coupling_fit_predict():
    """GENOT-Q (G2): quadratic/GW coupling — match cells by structure on ``obsm/X_pca``, generate ``X``."""
    d, pca = 6, 4
    adata, rng = _toy_adata(n=80, d=d, cond_dim=3, pca=pca)
    spec = FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
        coupling=CouplingDataSchema(src_quad="X_pca", tgt_quad="X_pca"),  # quadratic GW matching
    )
    compiled = spec.compile(adata, rep_tables=adata.uns)
    assert compiled.dims.coupling == {"src_quad": pca, "tgt_quad": pca}
    assert spec.coupling.is_quadratic

    model = FlowMatching(spec=spec, objective="genot", condition_embedding_dim=8, hidden_dims=(16, 16))
    model.fit(adata, rep_tables=adata.uns, batch_size=8, n_train_steps=4, device="cpu")
    assert model.objective._quad  # took the quadratic/GW coupling branch
    assert model.vf.use_source_encoder

    x = rng.random((7, d)).astype(np.float32)
    p0 = model.predict(x, ("cl_a", "drug_a"), device="cpu", num_steps=5, seed=0)
    p1 = model.predict(x, ("cl_a", "drug_a"), device="cpu", num_steps=5, seed=0)
    assert p0.shape == (7, d)
    assert np.isfinite(p0).all()
    assert np.array_equal(p0, p1)


def _shift_spec():
    return FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
    )


def test_stochastic_condition_encoder_learns():
    """condition_mode='stochastic': a VAE condition encoder (mean+logvar, reparameterized, KL-regularized).

    Asserts the encoder actually gained a variance head and that the KL-regularized model still learns the
    condition-dependent translation.
    """
    d, delta = 6, 5.0
    adata, controls, _ = _conditional_shift_adata(n_per=128, d=d, delta=delta, seed=3)
    model = FlowMatching(
        spec=_shift_spec(), objective="otfm", condition_mode="stochastic", condition_embedding_dim=16, hidden_dims=(64, 64)
    )
    model.fit(adata, rep_tables=adata.uns, batch_size=64, n_train_steps=200, lr=1e-2, device="cpu")
    assert model.vf.is_stochastic  # the encoder is variational (has the logvar head)

    x = controls["cl_a"]
    move_a = float((model.predict(x, ("cl_a", "drug_a"), num_steps=20, seed=0) - x).mean())
    move_b = float((model.predict(x, ("cl_a", "drug_b"), num_steps=20, seed=0) - x).mean())
    assert move_a > 0.3 * delta, f"stochastic CE: drug_a move {move_a:.2f} not toward +{delta}"
    assert move_b < -0.3 * delta, f"stochastic CE: drug_b move {move_b:.2f} not toward -{delta}"
    assert move_a - move_b > delta, f"stochastic CE: drugs not separated ({move_a:.2f} vs {move_b:.2f})"


def test_stochastic_condition_encoder_reproducible():
    """The stochastic path (reparameterization noise) is bit-reproducible given a seed."""
    adata, _, _ = _conditional_shift_adata(n_per=32, d=5, seed=4)

    def run():
        m = FlowMatching(spec=_shift_spec(), condition_mode="stochastic", condition_embedding_dim=8, hidden_dims=(16, 16))
        m.fit(adata, rep_tables=adata.uns, batch_size=16, n_train_steps=8, device="cpu")
        return m.vf.state_dict()

    sd_a, sd_b = run(), run()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"stochastic param {k} differs across identical-seed runs"
