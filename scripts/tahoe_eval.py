"""Does the model actually learn Tahoe perturbations? Held-in quality eval vs an identity baseline.

Train loss is a weak signal for flow-matching (its floor is the irreducible per-cell velocity variance,
so it stays high/flat even when the model has learned the mean shift). The real question is whether the
predicted perturbed distribution matches the real one BETTER than doing nothing. So: fit on the plates,
then for the top-N most-populated (cell_line, drug) conditions, translate that cell line's controls under
the drug and score r_squared / e_distance against the REAL perturbed cells — next to the identity
baseline (prediction = the untouched controls). Model must beat identity (higher R², lower E-distance).

Usage
-----
    python scripts/tahoe_eval.py --plates '/lustre/.../plate*.zarr' --eval-plate '/lustre/.../plate3_*.zarr' \
        --n-conditions 8 --n-train-steps 4000 --lr 3e-4 --chunk-size 32 --control-in-memory
"""

from __future__ import annotations

import argparse
import glob
import sys
import time

import numpy as np


def _read_obs_col(g, name):
    node = g[name]
    if hasattr(node, "keys") and "codes" in node:
        cats = np.asarray(node["categories"][:], dtype=object)
        return cats[np.asarray(node["codes"][:])]
    return np.asarray(node[:])


def _read_obs_cols(zpath):
    import zarr
    g = zarr.open_group(zpath, mode="r")["obs"]
    return (_read_obs_col(g, "cell_line"), _read_obs_col(g, "drug"),
            _read_obs_col(g, "is_control").astype(bool))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plates", required=True, help="comma-separated glob(s) of training plates")
    p.add_argument("--eval-plate", required=True,
                   help="comma-separated glob(s) of plates to read eval conditions from -- scored "
                        "independently (one fit(), scored against each eval plate in turn) so a single "
                        "training run can compare held-in vs cross-plate generalization")
    p.add_argument("--sample-rep", default="X_pca", help="obsm rep streamed as the state")
    p.add_argument("--n-conditions", type=int, default=8)
    p.add_argument("--eval-cells", type=int, default=512, help="max cells sampled per population for scoring")
    p.add_argument("--device", default="cuda")
    p.add_argument("--objective", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--control-in-memory", action="store_true")
    p.add_argument("--hidden-dims", default="1024,1024,1024")
    p.add_argument("--condition-embedding-dim", type=int, default=64)
    p.add_argument("--match-context", default="cell_line",
                    help="comma-separated match_context columns; add 'plate' to keep OT matching within-plate")
    p.add_argument("--regularization", type=float, default=0.0,
                    help="condition-embedding L2 penalty; default 0.0 (FlowMatching's own default of 1.0 "
                         "collapses drug embeddings -> model degenerates to the identity baseline)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-train-steps", type=int, default=4000)
    p.add_argument("--num-steps", type=int, default=50, help="ODE integration steps for predict")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import zarr

    from sc_flow import FlowMatching
    from sc_flow.core.metrics import EnergyDistance, RSquared
    from sc_flow.core.data import FlowSpec
    from sc_flow.core.data._encoders import one_hot
    from sc_flow.core.data.schemas import ConditionDataSchema, StateDataSchema

    plates = sorted({p for pattern in args.plates.split(",") for p in glob.glob(pattern)})
    eval_plates = sorted({p for pattern in args.eval_plate.split(",") for p in glob.glob(pattern)})
    chunk = args.chunk_size
    min_runs = chunk if chunk > 1 else 0
    hidden = tuple(int(x) for x in args.hidden_dims.split(",") if x)
    print(f"[eval] train_plates={len(plates)} eval_plates={[p.split('/')[-1] for p in eval_plates]} "
          f"rep={args.sample_rep} n_conditions={args.n_conditions} batch={args.batch_size} chunk={chunk} "
          f"match_context={args.match_context} reg={args.regularization} lr={args.lr} "
          f"ctrl_in_mem={args.control_in_memory} steps={args.n_train_steps}", flush=True)

    # --- fit on all training plates ---
    spec = FlowSpec(
        state=StateDataSchema(sample_rep=args.sample_rep),
        condition=ConditionDataSchema(conditions={"drug": ["drug"]}, condition_encoders={"drug": one_hot()}),
        control_key="is_control", match_context=args.match_context.split(","),
    )
    model = FlowMatching(spec=spec, objective=args.objective, condition_embedding_dim=args.condition_embedding_dim,
                         hidden_dims=hidden, pooling={"type": "sc_flow.mean", "version": 1, "config": {}},
                         regularization=args.regularization, seed=args.seed)
    t0 = time.perf_counter()
    model.fit(plates, rep_tables=None, batch_size=args.batch_size, chunk_size=chunk, min_runs_per_leaf=min_runs,
              control_in_memory=args.control_in_memory, n_train_steps=args.n_train_steps, device=args.device, lr=args.lr)
    print(f"[eval] fit {args.n_train_steps} steps in {time.perf_counter()-t0:.0f}s", flush=True)

    # --- score each condition, per eval plate: model vs identity baseline ---
    import torch

    def r2(pred, target):
        m = RSquared(); m.update(torch.as_tensor(pred), torch.as_tensor(target)); return float(m.compute())

    def ed(pred, target):
        m = EnergyDistance(); m.update(torch.as_tensor(pred), torch.as_tensor(target)); return float(m.compute())

    mc_cols = args.match_context.split(",")
    all_win = True
    for eval_plate in eval_plates:
        name = eval_plate.split("/")[-1]
        print(f"[eval] === eval_plate={name} ===", flush=True)

        # top-N (cell_line, drug) by perturbed cell count on THIS eval plate
        cl, dr, ic = _read_obs_cols(eval_plate)
        from collections import Counter
        pert_counts = Counter(zip(cl[~ic].tolist(), dr[~ic].tolist()))
        conditions = [c for c, _ in pert_counts.most_common(args.n_conditions)]
        Xg = zarr.open_group(eval_plate, mode="r")["obsm"][args.sample_rep]

        # predict()'s leaf tuple must match compile_obs's `cols = (*match_context, *cond_cols)` order/length --
        # cond_cols is always just ["drug"] here, so build (value-per-match-context-col..., drug). Every
        # match_context column besides "cell_line" is constant across this single eval plate's rows.
        g_obs = zarr.open_group(eval_plate, mode="r")["obs"]
        mc_const = {c: _read_obs_col(g_obs, c)[0] for c in mc_cols if c != "cell_line"}

        def leaf_for(cl_v, dr_v, mc_const=mc_const):
            return tuple(cl_v if c == "cell_line" else mc_const[c] for c in mc_cols) + (dr_v,)

        def read_rows(mask, cap, Xg=Xg):
            idx = np.flatnonzero(mask)
            if idx.size > cap:
                idx = np.sort(np.random.default_rng(0).choice(idx, cap, replace=False))
            return np.asarray(Xg.oindex[idx, :], dtype=np.float32)

        rows = []
        for cl_v, dr_v in conditions:
            ctrl = read_rows((cl == cl_v) & ic, args.eval_cells)
            real = read_rows((cl == cl_v) & (dr == dr_v) & ~ic, args.eval_cells)
            if len(ctrl) < 16 or len(real) < 16:
                continue
            pred = model.predict(ctrl, leaf_for(cl_v, dr_v), num_steps=args.num_steps, seed=0, device=args.device)
            rows.append((cl_v, dr_v, r2(pred, real), r2(ctrl, real), ed(pred, real), ed(ctrl, real)))
            c = rows[-1]
            print(f"[eval]   {str((cl_v, dr_v)):40.40s} R2 model={c[2]:+.3f} id={c[3]:+.3f} | "
                  f"Edist model={c[4]:.1f} id={c[5]:.1f}", flush=True)

        if not rows:
            print(f"[eval] {name}: no scorable conditions", flush=True)
            all_win = False
            continue
        a = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
        mr2, ir2, med, ied = a.mean(0)
        win = (mr2 > ir2) and (med < ied)
        all_win = all_win and win
        print(f"[eval] {name} MEAN over {len(rows)} conditions: R2 model={mr2:+.3f} vs identity={ir2:+.3f} | "
              f"E-dist model={med:.1f} vs identity={ied:.1f}", flush=True)
        print(f"[eval] {name} {'PASS' if win else 'FAIL'}: model {'BEATS' if win else 'does NOT beat'} the "
              f"no-perturbation baseline (higher R2 + lower E-distance).", flush=True)

    return 0 if all_win else 2


if __name__ == "__main__":
    sys.exit(main())
