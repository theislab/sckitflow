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


def test_predict_helpers_accept_device_tensors():
    """_as_f32 / condition_to_device / integrate_translation accept torch tensors already on a device.

    Guards the GPU validation path: Lightning moves the val batch onto the accelerator, so the coercion
    must NOT do np.asarray on those tensors (that raises "can't convert cuda tensor to numpy"). Exercised
    on CPU tensors here — the code path (isinstance torch.Tensor branch) is device-agnostic.
    """
    from sc_flow.backends.torch.training._predict import _as_f32, condition_to_device

    t = torch.ones(3, 4, dtype=torch.float64)  # a tensor, not numpy
    out = _as_f32(t, torch.device("cpu"))
    assert out.dtype == torch.float32 and out.shape == (3, 4)
    cond = condition_to_device({"drug": torch.zeros(3, 5)}, torch.device("cpu"))
    assert cond["drug"].dtype == torch.float32
    # numpy still works too (predict() path passes numpy)
    assert _as_f32(np.ones((2, 2), np.float64), torch.device("cpu")).dtype == torch.float32


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


@pytest.mark.parametrize("objective", ["otfm", "genot"])
def test_save_load_roundtrip(tmp_path, objective):
    """save()/load(): predict() after reload matches predict() before saving, bit-for-bit."""
    d = 5
    adata, rng = _toy_adata(n=64, d=d, cond_dim=4)
    spec = FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
    )
    model = FlowMatching(spec=spec, objective=objective, condition_embedding_dim=8, hidden_dims=(16, 16))
    model.fit(adata, rep_tables=adata.uns, batch_size=8, n_train_steps=5, device="cpu")

    x = rng.random((6, d)).astype(np.float32)
    leaf = ("cl_a", "drug_a")
    pred_before = model.predict(x, leaf, device="cpu", num_steps=5, seed=0)

    save_dir = tmp_path / "model"
    model.save(save_dir)
    assert (save_dir / "weights.pt").exists()
    assert (save_dir / "state.pkl").exists()

    reloaded = FlowMatching.load(save_dir)
    assert reloaded.objective_name == objective
    assert reloaded.vf.state_dict().keys() == model.vf.state_dict().keys()
    for k, v in model.vf.state_dict().items():
        assert torch.equal(v, reloaded.vf.state_dict()[k]), f"param {k} differs after reload"

    # reloaded model never called .fit(); predict must still work via the persisted condition_fn + dims.
    pred_after = reloaded.predict(x, leaf, device="cpu", num_steps=5, seed=0)
    assert np.array_equal(pred_before, pred_after)

    # also resolves a raw condition dict (not just a leaf tuple) and a fresh leaf, without erroring.
    other = reloaded.predict(x, ("cl_b", "drug_b"), device="cpu", num_steps=5, seed=0)
    assert np.isfinite(other).all()


def test_save_before_fit_raises(tmp_path):
    """save() on an unfitted model raises rather than silently writing nothing useful."""
    spec = FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
    )
    model = FlowMatching(spec=spec)
    with pytest.raises(RuntimeError, match="fitted"):
        model.save(tmp_path / "model")


# --- validation loop + held-out split ------------------------------------------------------------


def test_r_squared_metric():
    """RSquared: per-condition R² between predicted and target feature-wise means, averaged over updates."""
    from sc_flow.backends.torch.metrics import RSquared

    torch.manual_seed(0)
    # feature-varying target means (baseline differs per feature) so the R² denominator is non-degenerate.
    baseline = torch.tensor([0.0, 3.0, -2.0, 5.0])
    target = baseline + 0.01 * torch.randn(200, 4)
    m = RSquared()
    m.update(target.clone(), target)  # a (near-)perfect prediction → R² ≈ 1
    assert float(m.compute()) > 0.99

    # two conditions, each a constant ±0.5 offset from the true feature means → identical per-feature R²,
    # so the averaged compute() equals the single-condition hand-computed value.
    m.reset()
    ss_res = float(((baseline - (baseline + 0.5)) ** 2).sum())
    ss_tot = float(((baseline - baseline.mean()) ** 2).sum())
    for shift in (0.5, -0.5):
        m.update((baseline + shift).unsqueeze(0), baseline.unsqueeze(0))
    assert float(m.compute()) == pytest.approx(1.0 - ss_res / ss_tot, rel=1e-5)


def _multi_drug_adata(*, n_per=96, d=6, n_drugs=4, seed=0):
    """Cells with per-feature baselines (so R² is well-behaved) + several drugs to split over."""
    rng = np.random.default_rng(seed)
    baseline = rng.normal(0.0, 4.0, d).astype(np.float32)  # feature-varying means
    cts = {"cl_a": 0.0, "cl_b": 2.0}
    drugs = [f"drug_{i}" for i in range(n_drugs)]
    shifts = {dr: float(s) for dr, s in zip(drugs, np.linspace(-3, 3, n_drugs), strict=True)}
    xs, ct_col, dr_col, ctrl_col = [], [], [], []
    for ct, mu in cts.items():
        xs.append((baseline + mu + rng.normal(0, 0.3, (n_per, d))).astype(np.float32))
        ct_col += [ct] * n_per
        dr_col += ["control"] * n_per
        ctrl_col += [True] * n_per
        for dr in drugs:
            xs.append((baseline + mu + shifts[dr] + rng.normal(0, 0.3, (n_per, d))).astype(np.float32))
            ct_col += [ct] * n_per
            dr_col += [dr] * n_per
            ctrl_col += [False] * n_per
    obs = pd.DataFrame({"cell_type": ct_col, "drug1": dr_col, "control": ctrl_col})
    for c in ("cell_type", "drug1"):
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.concatenate(xs, 0), obs=obs)
    adata.uns["drug"] = {dd: rng.standard_normal((1, 4)).astype(np.float32) for dd in obs["drug1"].cat.categories}
    return adata


@pytest.mark.parametrize("objective", ["otfm", "genot"])
def test_fit_validation_loop(objective):
    """fit(split_by=...): holds out whole drugs, scores distribution metrics on them, records history.

    Asserts the plumbing (both flows): metrics_history carries the requested metrics, one finite value per
    validation pass, and the whole train+validate run is bit-reproducible under a fixed seed.
    """
    adata = _multi_drug_adata()
    spec = _shift_spec()
    kw = {
        "rep_tables": adata.uns, "batch_size": 48, "n_train_steps": 60, "valid_freq": 25, "lr": 1e-2,
        "device": "cpu", "split_by": "drug1", "split_ratios": (0.5, 0.5),
        "metrics": ("r_squared", "e-dist"), "val_num_steps": 10,
    }

    def run():
        m = FlowMatching(spec=spec, objective=objective, condition_embedding_dim=8, hidden_dims=(32, 32), seed=0)
        m.fit(adata, **kw)
        return m

    model = run()
    hist = model.metrics_history
    assert set(hist) == {"r_squared", "e-dist"}
    # validation fires every 25 steps over 60 steps → passes at 25 and 50.
    assert len(hist["r_squared"]) == 2
    assert len(hist["e-dist"]) == len(hist["r_squared"])
    assert all(np.isfinite(v) for vals in hist.values() for v in vals)

    # same seed → identical validation history (train + eval both deterministic on CPU)
    assert run().metrics_history == hist


def test_fit_split_holds_out_whole_conditions():
    """split_by held out whole drugs: the val leaves are disjoint from the train leaves (no cell leakage)."""
    from binded import split_assignment, split_scheme

    adata = _multi_drug_adata(n_drugs=4)
    compiled = _shift_spec().compile(adata, rep_tables=adata.uns, seed=0)
    splits = split_scheme(compiled.scheme, split_by=["drug1"], ratios={"train": 0.5, "val": 0.5}, random_state=0)
    assign = split_assignment(splits)
    train_drugs = set(assign.loc[assign["split"] == "train", "drug1"])
    val_drugs = set(assign.loc[assign["split"] == "val", "drug1"])
    assert train_drugs and val_drugs
    assert train_drugs.isdisjoint(val_drugs)  # whole conditions held out, not random cells


def test_split_ratios_forms_and_validation():
    """_resolve_split_ratios accepts a (train, val) sequence or a mapping, and rejects malformed input."""
    resolve = FlowMatching._resolve_split_ratios
    assert resolve(None) == {"train": 0.8, "val": 0.2}
    assert resolve((0.7, 0.3)) == {"train": 0.7, "val": 0.3}
    assert resolve({"train": 0.6, "val": 0.4}) == {"train": 0.6, "val": 0.4}
    with pytest.raises(ValueError, match="train.*val|val.*train"):
        resolve({"train": 0.8})
    with pytest.raises(ValueError, match="train, val"):
        resolve((0.5, 0.3, 0.2))


@pytest.mark.parametrize("chunk_size", [1, 8])
def test_fit_chunked_reads_train_and_predict(chunk_size):
    """fit() accepts chunk_size>1 on grouped data (contiguous per-condition runs) — the loader fast path.

    ``_multi_drug_adata`` lays cells out in contiguous per-(cell_type, drug) blocks, so a chunked read is
    valid. chunk_size changes only the read pattern (sequential vs scattered) — this checks the fast path
    fits and predicts finite outputs without error (not bit-equivalence: chunked reads change sample
    order, so weights differ from the scattered path).
    """
    adata = _multi_drug_adata(n_per=96, d=6)
    model = FlowMatching(spec=_shift_spec(), condition_embedding_dim=8, hidden_dims=(16, 16))
    model.fit(
        adata, rep_tables=adata.uns, batch_size=32, chunk_size=chunk_size, n_train_steps=6, device="cpu"
    )
    pred = model.predict(np.zeros((5, 6), np.float32), ("cl_a", "drug_0"), device="cpu", num_steps=4)
    assert pred.shape == (5, 6) and np.isfinite(pred).all()


def test_fit_resolve_preload_buffer():
    """_resolve_preload: explicit wins; chunk_size=1 → batch-sized; chunked → a many-batch prefetch."""
    assert FlowMatching._resolve_preload(256, 1, 999) == 999
    assert FlowMatching._resolve_preload(256, 1, None) == 256
    # chunked default buffers ~32 batches of chunks so periodic loader refills don't stall training.
    assert FlowMatching._resolve_preload(1024, 32, None) == 32 * (1024 // 32)  # = 1024 chunks


def test_fit_resolve_preload_to_gpu():
    """Transport: explicit wins; CPU training never GPU-preloads; GPU training defers to binded auto (None)."""
    assert FlowMatching._resolve_preload_to_gpu("cpu", True) is True     # explicit wins
    assert FlowMatching._resolve_preload_to_gpu("cuda", False) is False  # explicit wins
    assert FlowMatching._resolve_preload_to_gpu("cpu", None) is False    # never GPU-preload on CPU
    assert FlowMatching._resolve_preload_to_gpu("cuda", None) is None    # GPU: binded auto (cupy if present)


def test_fit_unknown_metric_raises():
    """A validation metric name not in METRICS_REGISTRY fails fast (before training)."""
    adata = _multi_drug_adata()
    model = FlowMatching(spec=_shift_spec(), condition_embedding_dim=8, hidden_dims=(16, 16))
    with pytest.raises(KeyError, match="not-a-metric"):
        model.fit(
            adata, rep_tables=adata.uns, batch_size=16, n_train_steps=2,
            split_by="drug1", metrics=("not-a-metric",),
        )
