"""
VERIFICATION: sc_flow interface input  ──compile──►  dagloader.Scheme  ──run──►  matched batches

Feeds a config written in sc_flow's vocabulary (sample_rep / groups / conditions /
control_values_dict | matched_keys) through a compiler (`lower_to_scheme`) that emits a
REAL `dagloader.Scheme` + `condition_fn`, then:
  A. asserts the compiled Scheme is structurally correct (obs/labels only, no cell reads);
  B. drives the REAL `DAGLoader` and asserts the STREAMED cells obey the structure —
     target/source split + same-group (Bind) matching — proving the compile end-to-end;
  C. compiles the `matched_keys` (one-to-one) mode and asserts the tag+Bind lowering.

Run: cellflow/.venv/bin/python verify_scflow_to_scheme.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import anndata as ad

from dagloader import Bind, Node, Scheme, SamplerConfig, DAGLoader, uniform

Leaf = tuple

# ── the sc_flow-vocabulary INPUT (no bio, no dagloader terms) ───────────────────────────
@dataclass(frozen=True)
class SCFlowSpec:
    sample_rep: str                                   # "X" | obsm key
    conditions: Mapping[str, Sequence[str]]           # {level: obs-cols}  (hierarchy levels)
    groups: Sequence[str] = ()                        # top level + the match key
    control_values_dict: Mapping[str, object] | None = None   # source value per level  -> Bind
    matched_keys: Mapping[Leaf, Leaf] | None = None           # explicit source->target -> tag+Bind
    seed: int = 0


# ── THE COMPILER  (structural passes only; condition_fn = one-hot stand-in for the schemas) ──
def lower_to_scheme(spec: SCFlowSpec, store):
    # Pass 1 · hierarchy -> Node.cols + leaf space (OBS ONLY)
    groups_cols = tuple(spec.groups)
    cond_cols = tuple(c for cols in spec.conditions.values() for c in cols)
    cols = (*groups_cols, *cond_cols)
    obs = store.obs[list(cols)]
    uniq = [tuple(r) for r in obs.drop_duplicates().to_numpy()]

    # Pass 2 · representation -> Node.keys  (a LOCATION, never the array)
    keys = "X" if spec.sample_rep == "X" else f"obsm/{spec.sample_rep}"

    # Pass 3 · partition leaves into target vs source
    if spec.matched_keys is not None:
        return _lower_matched_keys(spec, store, cols, keys)          # -> Scenario C
    cvd = spec.control_values_dict or {}
    def is_source(lf: Leaf) -> bool:
        return all(lf[cols.index(c)] == v for c, v in cvd.items())
    source_leaves = [lf for lf in uniq if is_source(lf)]
    target_leaves = [lf for lf in uniq if not is_source(lf)]
    match_on = groups_cols

    # Pass 4 · weights (the selection IS the weight)
    target_w, source_w = uniform(target_leaves), uniform(source_leaves)

    # Pass 5 · binding
    binds = (Bind("target", "source", common=match_on),)

    # Pass 6 · encoder: leaf -> one-hot(condition value).  (obs/uns only; stands in for schemas)
    drug_col = cond_cols[0]
    drug_pos = cols.index(drug_col)
    vocab = sorted({lf[drug_pos] for lf in uniq})
    idx = {v: i for i, v in enumerate(vocab)}
    def condition_fn(leaf: Leaf) -> np.ndarray:
        oh = np.zeros(len(vocab), np.float32); oh[idx[leaf[drug_pos]]] = 1.0
        return oh

    # Pass 7 · stores (single input here; multi-input would be {"a":A,"b":B})
    scheme = Scheme(
        sources={"data": store},
        nodes={
            "target": Node("data", cols, keys, target_w),
            "source": Node("data", cols, keys, source_w, in_memory=True),
        },
        root="target",
        binds=binds,
        seed=spec.seed,
    )
    meta = dict(cols=cols, keys=keys, target_leaves=set(target_leaves),
                source_leaves=set(source_leaves), match_on=match_on,
                drug_pos=drug_pos, vocab=vocab)
    return scheme, condition_fn, meta


def _lower_matched_keys(spec, store, cols, keys):
    """one-to-one: synthesize a `_match_id` shared column so an arbitrary source->target
    map becomes a Bind on shared column values (no new dagloader primitive)."""
    mk = spec.matched_keys
    # tag obs: each target leaf gets its own id; the matched source leaf gets the SAME id.
    obs = store.obs
    leaf_of = lambda row: tuple(row[c] for c in cols)
    tgt_to_id = {t: f"m{i}" for i, t in enumerate(dict.fromkeys(mk.values()))}
    src_to_id = {s: tgt_to_id[t] for s, t in mk.items()}
    match_id = []
    for _, row in obs.iterrows():
        lf = leaf_of(row)
        match_id.append(tgt_to_id.get(lf, src_to_id.get(lf, "")))
    store.obs["_match_id"] = match_id
    cols2 = (*cols, "_match_id")
    target_leaves = [(*t, tgt_to_id[t]) for t in mk.values()]
    source_leaves = [(*s, src_to_id[s]) for s in mk]
    scheme = Scheme(
        sources={"data": store},
        nodes={"target": Node("data", cols2, keys, uniform(target_leaves)),
               "source": Node("data", cols2, keys, uniform(source_leaves), in_memory=True)},
        root="target", binds=(Bind("target", "source", common=("_match_id",)),), seed=spec.seed,
    )
    meta = dict(cols=cols2, match_on=("_match_id",),
                target_leaves=set(target_leaves), source_leaves=set(source_leaves))
    return scheme, (lambda leaf: np.zeros(1, np.float32)), meta


# ── synthetic AnnData: X_state[:,0]=group id, [:,1]=drug id — so we can read matching back ──
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    groups = ["b0", "b1"]
    drugs = ["DMSO", "D1", "D2"]
    gid = {g: i for i, g in enumerate(groups)}
    did = {d: i for i, d in enumerate(drugs)}
    rows = []
    for g in groups:
        for d in drugs:
            for _ in range(50):
                rows.append((g, d))
    df = pd.DataFrame(rows, columns=["batch", "drug"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    n = len(df)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    X[:, 0] = df["batch"].map(gid).to_numpy()
    X[:, 1] = df["drug"].map(did).to_numpy()
    A = ad.AnnData(X=rng.normal(size=(n, 4)).astype(np.float32), obs=df)
    A.obsm["X_state"] = X
    # sort by hierarchy cols (mirrors the in-memory build path; harmless at chunk_size=1)
    order = A.obs.sort_values(["batch", "drug"], kind="stable").index.to_numpy()
    A = A[order].copy()
    return A, gid, did


# ─────────────────────────────────────────────── RUN ───────────────────────────────────
results = []
def check(name, cond, extra=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {extra}" if extra else ""))


print("═══ Scenario A · control_values_dict → compile ═══")
A, gid, did = make_adata()
spec = SCFlowSpec(
    sample_rep="X_state",
    groups=["batch"],
    conditions={"drug": ["drug"]},
    control_values_dict={"drug": "DMSO"},
    seed=0,
)
scheme, condition_fn, meta = lower_to_scheme(spec, A)

print("── A. structural (labels only, zero cell reads) ──")
tnode, snode = scheme.nodes["target"], scheme.nodes["source"]
check("cols = groups then conditions", tnode.cols == ("batch", "drug"), str(tnode.cols))
check("keys = obsm location string", tnode.keys == ("obsm/X_state",), str(tnode.keys))
check("root is target", scheme.root == "target")
check("bind target→source on groups", scheme.binds == (Bind("target", "source", common=("batch",)),))
exp_target = {("b0", "D1"), ("b0", "D2"), ("b1", "D1"), ("b1", "D2")}
exp_source = {("b0", "DMSO"), ("b1", "DMSO")}
check("target leaves = non-control combos", meta["target_leaves"] == exp_target, str(meta["target_leaves"]))
check("source leaves = control combos", meta["source_leaves"] == exp_source, str(meta["source_leaves"]))
check("target weights uniform over target leaves", set(tnode.weights) == exp_target
      and len(set(tnode.weights.values())) == 1)
check("source weights uniform over source leaves", set(snode.weights) == exp_source)

print("── B. runtime: drive the REAL DAGLoader, check streamed cells obey the structure ──")
cfg = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8, to=None)
loader = DAGLoader(scheme, cfg, condition_fn=condition_fn)
DMSO = did["DMSO"]
n_ok = 0
it = iter(loader)
for _ in range(20):
    b = next(it)
    tgt, src, cond = b["target"], b["source"], b["condition"]
    # target: class-coherent, drug != DMSO
    tgt_drug = np.unique(tgt[:, 1]); tgt_grp = np.unique(tgt[:, 0])
    # source: DMSO, and SAME group as target (the Bind match)
    src_drug = np.unique(src[:, 1]); src_grp = np.unique(src[:, 0])
    # condition one-hot is over the compiler's own (alphabetical) vocab; map its argmax
    # back through `did` to compare against the drug id encoded in the streamed cells.
    cond_drug_id = did[meta["vocab"][int(np.asarray(cond)[0].argmax())]]
    ok = (tgt_drug.size == 1 and tgt_drug[0] != DMSO
          and tgt_grp.size == 1
          and src_drug.tolist() == [DMSO]
          and src_grp.size == 1 and src_grp[0] == tgt_grp[0]           # ← matching
          and cond.shape == (tgt.shape[0], 3)
          and cond_drug_id == int(tgt_drug[0]))                        # ← condition_fn ↔ cells
    n_ok += ok
check("target shape (B,4)", tgt.shape == (8, 4), str(tgt.shape))
check("source shape (B,4)", src.shape == (8, 4), str(src.shape))
check("20/20 batches: target≠DMSO, source=DMSO, source group == target group (Bind match)",
      n_ok == 20, f"{n_ok}/20")
check("condition one-hot matches target's drug in every checked batch", n_ok == 20)

print("\n═══ Scenario C · matched_keys (one-to-one) → compile (structural) ═══")
A2, _, _ = make_adata()
spec2 = SCFlowSpec(
    sample_rep="X_state",
    groups=["batch"],
    conditions={"drug": ["drug"]},
    matched_keys={("b0", "DMSO"): ("b0", "D1"), ("b1", "DMSO"): ("b1", "D2")},
    seed=0,
)
scheme2, _, meta2 = lower_to_scheme(spec2, A2)
check("synthesized _match_id column in cols", "_match_id" in scheme2.nodes["target"].cols)
check("bind is on _match_id", scheme2.binds[0].common == ("_match_id",))
check("target leaves carry match ids", all(len(t) == 3 for t in meta2["target_leaves"]))
check("source & target share match-id values",
      {t[-1] for t in meta2["target_leaves"]} == {s[-1] for s in meta2["source_leaves"]})

print("\n" + "═" * 60)
print(f"RESULT: {sum(results)}/{len(results)} checks passed"
      + ("  ✅ ALL PASS" if all(results) else "  ❌ FAILURES"))
raise SystemExit(0 if all(results) else 1)
