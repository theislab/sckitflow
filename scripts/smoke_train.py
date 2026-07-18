"""Standalone smoke test: FlowMatching (torch-native OTFM/GENOT) fit + predict end-to-end.

Not a pytest test — a runnable script meant to prove the training path works on a given machine
(laptop CPU, cluster CPU/GPU node) outside the test harness, with real console output. Trains on
synthetic data with a known conditional shift (same construction as
``tests/test_flow_matching.py::_conditional_shift_adata``) so success is a real learning signal, not
just "it didn't crash": each drug applies a known, opposite shift to control cells, and the script
asserts the trained model recovers the right direction for both.

Usage
-----
    python scripts/smoke_train.py --objective otfm --device cpu
    python scripts/smoke_train.py --objective genot --device cuda --n-train-steps 500

Exits non-zero (and prints a clear failure) if the model doesn't learn the conditional shift, so it
can be used as a CI-less "did the cluster set up the numeric stack correctly" gate.
"""

from __future__ import annotations

import argparse
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

from sc_flow import FlowMatching
from sc_flow.data import FlowSpec
from sc_flow.data._encoders import lookup
from sc_flow.data.schemas import ConditionDataSchema, StateDataSchema


def make_conditional_shift_adata(*, n_per: int, d: int, delta: float, seed: int) -> tuple[ad.AnnData, dict, dict]:
    """Cells where each drug applies a known, opposite shift — same construction as the test suite."""
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
    adata.uns["drug"] = {dd: rng.standard_normal((1, 4)).astype(np.float32) for dd in obs["drug1"].cat.categories}
    return adata, controls_by_ct, shift


def main() -> int:
    """Parse args, fit a FlowMatching model on the synthetic shift data, and check it learned the shift."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--condition-mode", choices=["deterministic", "stochastic"], default="deterministic")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-per", type=int, default=128, help="cells per (cell_type, condition) leaf")
    p.add_argument("--d", type=int, default=6, help="state feature dim")
    p.add_argument("--delta", type=float, default=5.0, help="magnitude of the per-drug shift")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-train-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"[smoke_train] torch device={args.device!r} objective={args.objective!r} "
          f"condition_mode={args.condition_mode!r}", flush=True)

    adata, controls, shift = make_conditional_shift_adata(
        n_per=args.n_per, d=args.d, delta=args.delta, seed=args.seed
    )
    print(f"[smoke_train] adata: {adata.shape}, drugs={sorted(shift)}", flush=True)

    spec = FlowSpec(
        state=StateDataSchema(sample_rep="X"),
        condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
        control_key="control",
        match_context=["cell_type"],
    )
    model = FlowMatching(
        spec=spec,
        objective=args.objective,
        condition_mode=args.condition_mode,
        condition_embedding_dim=16,
        hidden_dims=(64, 64),
        seed=args.seed,
    )

    t0 = time.perf_counter()
    model.fit(
        adata,
        rep_tables=adata.uns,
        batch_size=args.batch_size,
        n_train_steps=args.n_train_steps,
        lr=args.lr,
        device=args.device,
    )
    fit_s = time.perf_counter() - t0
    print(f"[smoke_train] fit done in {fit_s:.1f}s ({args.n_train_steps} steps)", flush=True)

    x_ctrl = controls["cl_a"]
    t0 = time.perf_counter()
    move_a = float((model.predict(x_ctrl, ("cl_a", "drug_a"), num_steps=20, seed=0, device=args.device) - x_ctrl).mean())
    move_b = float((model.predict(x_ctrl, ("cl_a", "drug_b"), num_steps=20, seed=0, device=args.device) - x_ctrl).mean())
    pred_s = time.perf_counter() - t0
    print(f"[smoke_train] predict done in {pred_s:.2f}s | move_a={move_a:+.2f} move_b={move_b:+.2f} "
          f"(true shifts: drug_a={shift['drug_a']:+.1f} drug_b={shift['drug_b']:+.1f})", flush=True)

    delta = args.delta
    ok = (move_a > 0.3 * delta) and (move_b < -0.3 * delta) and (move_a - move_b > delta)
    if ok:
        print("[smoke_train] PASS: model learned the condition-dependent translation.", flush=True)
        return 0
    print("[smoke_train] FAIL: model did not learn the expected condition-dependent translation.", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
