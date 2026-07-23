"""Minimal end-to-end pipeline smoke, driven entirely by an OmegaConf config.

Exercises the condition contract end-to-end on tiny dummy AnnData (no jax: otfm + independent coupling):
  * a **categorical** condition realm (``categorical()`` -> int index) encoded model-side, with the
    ``condition_encoding`` knob selecting **onehot** for it (default is a learned embedding);
  * a **feature covariate** realm (``lookup()`` -> looked-up vector, tiled across slots -> feature_mlp),
    exercising the covariate migration onto the same index/feature contract as conditions.

The gate for the condition-contract rewrite — must stay green.

    python scripts/smoke_pipeline.py
"""

from __future__ import annotations

from omegaconf import OmegaConf

from sc_flow import FlowMatching
from sc_flow.data import FlowSpec
from sc_flow.data._encoders import build_encoder
from sc_flow.data.schemas import ConditionDataSchema, CovariatesDataSchema, StateDataSchema
from sc_flow.data.sim import get_dummy_adata

CONFIG = """
data:
  n_obs_pert: 200
  n_obs_ctrl: 100
  n_genes: 50
spec:
  state_rep: X_repr
  control_key: is_control
  conditions:
    drug: [drugA, drugB]
  condition_encoders:
    drug: categorical          # -> int index; model owns the encoding
  covariates:
    koA: lookup:ko             # -> looked-up feature vector (tiled), model MLP-projects it
  match_context: [source_split]
model:
  objective: otfm
  match_method: independent
  condition_encoding: {drug: onehot}   # the knob: drug -> onehot (else embedding); feature realms -> mlp
  pooling: {type: sc_flow.mean, version: 1, config: {}}
  hidden_dims: [16]
  condition_embedding_dim: 8
  state_latent_dim: 8
  time_latent_dim: 4
  source_latent_dim: 4
  num_time_features: 16
fit:
  batch_size: 32
  n_train_steps: 3
  device: cpu
"""


def build(cfg) -> tuple[FlowMatching, object]:
    adata = get_dummy_adata(
        n_obs_pert=cfg.data.n_obs_pert, n_obs_ctrl=cfg.data.n_obs_ctrl, n_genes=cfg.data.n_genes
    )
    covariates = None
    if cfg.spec.get("covariates"):
        covariates = CovariatesDataSchema(
            covariate_encoders={c: build_encoder(e) for c, e in cfg.spec.covariates.items()}
        )
    spec = FlowSpec(
        state=StateDataSchema(sample_rep=cfg.spec.state_rep),
        condition=ConditionDataSchema(
            conditions=OmegaConf.to_container(cfg.spec.conditions, resolve=True),
            condition_encoders={r: build_encoder(e) for r, e in cfg.spec.condition_encoders.items()},
        ),
        control_key=cfg.spec.control_key,
        covariates=covariates,
        match_context=list(cfg.spec.match_context),
    )
    # The whole model recipe is the `model:` config block — one call, no kwargs unpacking.
    model = FlowMatching.from_config(spec, cfg.model)
    return model, adata


def main() -> None:
    cfg = OmegaConf.create(CONFIG)
    model, adata = build(cfg)
    print(f"[ok] built FlowSpec + FlowMatching from OmegaConf ({adata.n_obs} cells)")

    model.fit(
        adata,
        batch_size=cfg.fit.batch_size,
        chunk_size=1,
        n_train_steps=cfg.fit.n_train_steps,
        device=cfg.fit.device,
    )
    assert model.vf is not None, "fit must build the velocity field"
    print("[ok] fit: categorical(onehot) condition + lookup covariate, through SetEncoder -> OTFM(independent)")

    # predict on a few controls, conditioned on a real perturbed leaf (cols order = match_context+cond+cov).
    # Exercises the predict path: condition resolution -> condition_to_device (integer index -> long) ->
    # SetEncoder onehot/feature -> ODEPredictor.
    import numpy as np

    is_ctrl = adata.obs[cfg.spec.control_key].astype(bool).to_numpy()
    row = adata.obs[~is_ctrl].iloc[0]
    leaf = (str(row["source_split"]), str(row["drugA"]), str(row["drugB"]), str(row["koA"]))
    x = np.asarray(adata[is_ctrl].obsm[cfg.spec.state_rep])[:5]
    pred = model.predict(x, condition=leaf, device="cpu", num_steps=3)
    assert pred.shape == x.shape, (pred.shape, x.shape)
    print(f"[ok] predict through the shared ODEPredictor: {pred.shape}")

    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
