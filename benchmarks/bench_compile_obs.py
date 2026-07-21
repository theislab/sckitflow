"""Scaling benchmark for :func:`sc_flow.data.compile_obs` at 100M-obs scale.

``compile_obs`` is deliberately **obs-only**: it never reads ``.X`` / ``.obsm`` cells (those stream
later via ``binded``). Its cost therefore scales with the number of obs ROWS and the number of label
columns, not with cell feature dims. This benchmark confirms it stays cheap at 100M obs and locates
the superlinear / high-constant hot paths.

WHAT IT MEASURES
----------------
For each ``N`` in a sweep it runs the real :func:`compile_obs` end-to-end (authoritative wall-time +
peak memory), and separately reproduces each hot phase *verbatim* from
``src/sc_flow/data/_compile_obs.py`` on the same obs frame so the per-phase breakdown is faithful:

* ``open+obs_read`` — ``open_source`` + ``obs_columns(source, data_cols)`` (line 144-145): reads the
  label columns for all rows through ``binded._io``.
* ``encoder_fit``   — ``_fit_encoders`` (line 147): ``obs[cols].to_numpy().reshape(-1)`` then
  ``Encoder.fit(union)`` — O(rows) union materialization per realm (+ a scikit-learn ``np.unique``
  fit for one-hot/label; a lookup just binds its table).
* ``ctrl_split``    — ``obs[control_key].to_numpy().astype(bool)`` (line 204).
* ``target_leaves`` — ``Counter(tuple(r) for r in frame[cols].to_numpy())`` (line 195): a pure-Python
  loop building N tuples. The task's prime suspect for the dominant cost.
* ``leaves``        — ``frame[cols].drop_duplicates().to_numpy()`` (line 190): the vectorized path.
* ``target_leaves_vectorized`` — the proposed optimization: ``binded._io.leaf_codes`` (C-level
  factorize, no per-row tuples) + ``np.bincount`` for the same per-leaf counts. Reported as a
  before/after against ``target_leaves``.

DATA
----
100M obs will not fit as one in-memory ``AnnData``, so the benchmark drives an **out-of-core** source
resolved by ``binded._io.open_source`` / ``obs_columns``:

* default: a synthetic obs-only ``annbatch.DatasetCollection`` — a handful of ``dataset_*`` shards
  (~one plate each) of realistic Tahoe-shaped labels (categorical ``cell_line`` + ``drug`` + boolean
  ``is_control``), attached zero-copy. Cached under ``--cache-dir`` keyed by shape, so re-runs skip
  regeneration.
* ``--single-zarr``: one backed zarr AnnData (exercises ``load_backed_adata`` instead).
* ``--data <path>``: a real zarr adata / collection (swap in the real 100M dataset when available);
  the sweep is ignored and that one source is benchmarked.

USAGE
-----
    # quick sweep (default 1e5, 1e6, 1e7)
    .venv/bin/python benchmarks/bench_compile_obs.py

    # full sweep to 100M (writes ~cache), one-hot encoder to expose the sklearn fit
    .venv/bin/python benchmarks/bench_compile_obs.py --sweep 100000,1000000,10000000,100000000

    # against a real store (zarr adata or annbatch collection)
    .venv/bin/python benchmarks/bench_compile_obs.py --data /path/to/tahoe100_grouped.zarr
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import resource
import shutil
import sys
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Real Tahoe plate cardinalities (plate12_filt_Vevo_Tahoe100M...): cell_line=50, drug=95.
DEFAULT_CELL_LINES = 50
DEFAULT_DRUGS = 95
DEFAULT_CONTROL_FRAC = 0.05
DEFAULT_SHARD_ROWS = 10_000_000  # ~one Tahoe plate (10.49M) per collection shard


# --------------------------------------------------------------------------------------------------
# timing / memory helpers
# --------------------------------------------------------------------------------------------------
def _rss_bytes() -> int:
    """Current process peak RSS in bytes (ru_maxrss is bytes on macOS, KiB on Linux)."""
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m if sys.platform == "darwin" else m * 1024


@dataclass
class Phase:
    name: str
    seconds: float
    py_peak_mb: float  # tracemalloc peak (Python-object allocation) during the phase
    result: object = None


def timed(name: str, fn: Callable[[], object]) -> Phase:
    """Run ``fn`` once, timing wall-clock and tracemalloc peak Python allocation."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Phase(name, dt, peak / 1e6, result)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# --------------------------------------------------------------------------------------------------
# synthetic Tahoe-shaped obs generation (out-of-core: sharded collection or single backed zarr)
# --------------------------------------------------------------------------------------------------
def _make_obs(n: int, *, n_cell_lines: int, n_drugs: int, control_frac: float, seed: int) -> pd.DataFrame:
    """One shard of realistic Tahoe-shaped labels as categorical / boolean obs columns.

    ``cell_line`` (match_context) and ``drug`` (condition) are stored categorical — exactly as the real
    Tahoe plate obs is — so the read + ``.to_numpy()`` costs match production. ``control_frac`` rows are
    marked ``is_control`` and get the ``"control"`` drug sentinel (mirrors the DMSO vehicle split).
    """
    rng = np.random.default_rng(seed)
    cell_lines = pd.Categorical.from_codes(
        rng.integers(0, n_cell_lines, size=n, dtype=np.int32),
        categories=[f"cell_line_{i}" for i in range(n_cell_lines)],
    )
    drug_codes = rng.integers(0, n_drugs, size=n, dtype=np.int32)
    is_control = rng.random(n) < control_frac
    drug_cats = [f"drug_{i}" for i in range(n_drugs)] + ["control"]
    drug_int = np.where(is_control, n_drugs, drug_codes).astype(np.int32)  # control -> sentinel category
    drug = pd.Categorical.from_codes(drug_int, categories=drug_cats)
    return pd.DataFrame({"cell_line": cell_lines, "drug": drug, "is_control": is_control})


def _write_shard(path: str, obs: pd.DataFrame) -> None:
    """Write one obs-only AnnData zarr (dummy 1-feature X so it is a valid attachable AnnData store)."""
    import anndata as ad

    X = np.zeros((len(obs), 1), dtype=np.float32)  # cells never read by compile_obs; kept minimal + valid
    adata = ad.AnnData(X=X, obs=obs.reset_index(drop=True))
    adata.var_names = ["g0"]
    adata.write_zarr(path)


def _cache_key(n: int, n_cell_lines: int, n_drugs: int, control_frac: float, shard_rows: int, single: bool) -> str:
    kind = "zarr" if single else "coll"
    return f"bench_{kind}_n{n}_cl{n_cell_lines}_dr{n_drugs}_cf{control_frac:g}_sh{shard_rows}"


def build_synthetic_source(
    n: int,
    *,
    cache_dir: str,
    n_cell_lines: int,
    n_drugs: int,
    control_frac: float,
    shard_rows: int,
    single_zarr: bool,
    seed: int = 0,
) -> str:
    """Build (or reuse a cached) synthetic out-of-core source of ``n`` obs; return its path.

    * collection (default): ``ceil(n / shard_rows)`` obs-only zarr shards attached zero-copy into a
      ``DatasetCollection`` — the realistic multi-plate Tahoe layout, and it bounds generation RAM to
      one shard at a time.
    * ``single_zarr``: one backed zarr AnnData (only sensible for smaller ``n``).
    """
    from annbatch import DatasetCollection

    key = _cache_key(n, n_cell_lines, n_drugs, control_frac, shard_rows, single_zarr)
    root = os.path.join(cache_dir, key)
    done = root + ".done"
    final_path = root + ".zarr" if single_zarr else os.path.join(root, "collection.zarr")
    if os.path.exists(done):
        return final_path
    # rebuild cleanly if a previous run was interrupted mid-write
    for p in (root, root + ".zarr", done):
        if os.path.exists(p):
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)

    if single_zarr:
        _write_shard(final_path, _make_obs(n, n_cell_lines=n_cell_lines, n_drugs=n_drugs,
                                           control_frac=control_frac, seed=seed))
        open(done, "w").close()
        return final_path

    os.makedirs(root, exist_ok=True)
    n_shards = (n + shard_rows - 1) // shard_rows
    shard_paths = []
    for s in range(n_shards):
        rows = min(shard_rows, n - s * shard_rows)
        sp = os.path.join(root, f"_src_shard_{s}.zarr")
        _write_shard(
            sp, _make_obs(rows, n_cell_lines=n_cell_lines, n_drugs=n_drugs, control_frac=control_frac, seed=seed + s)
        )
        shard_paths.append(sp)
    DatasetCollection(final_path, mode="a").attach(shard_paths)
    open(done, "w").close()
    return final_path


# --------------------------------------------------------------------------------------------------
# the benchmark: full compile_obs run + per-phase breakdown
# --------------------------------------------------------------------------------------------------
def _schemas(encoder: str):
    """Tahoe-shaped schema objects for compile_obs: drug=condition, cell_line=match_context."""
    from sc_flow.data._encoders import label, lookup, one_hot
    from sc_flow.data.schemas import ConditionDataSchema, StateDataSchema

    enc = {"lookup": lambda: lookup("drug"), "onehot": one_hot, "label": label}[encoder]()
    state = StateDataSchema(sample_rep="X")
    condition = ConditionDataSchema(conditions={"drug": ["drug"]}, condition_encoders={"drug": enc})
    return state, condition


def _rep_tables(encoder: str, n_drugs: int, dim: int = 16) -> dict:
    """Explicit lookup tables (a DatasetCollection has no in-memory ``.uns``)."""
    if encoder != "lookup":
        return {}
    rng = np.random.default_rng(0)
    cats = [f"drug_{i}" for i in range(n_drugs)] + ["control"]
    return {"drug": {c: rng.standard_normal((1, dim)).astype(np.float32) for c in cats}}


def run_full_compile(source_path: str, *, encoder: str, n_drugs: int, min_runs_per_leaf: int
                     ) -> tuple[float, float, int, tuple[int, int]]:
    """Run the real compile_obs end-to-end; return (wall_s, py_peak_mb, rss_delta_bytes, (n_pert, n_ctrl))."""
    from sc_flow.data import compile_obs

    state, condition = _schemas(encoder)
    rep_tables = _rep_tables(encoder, n_drugs)

    gc.collect()
    rss0 = _rss_bytes()
    tracemalloc.start()
    t0 = time.perf_counter()
    compiled = compile_obs(
        source_path,
        state=state,
        condition=condition,
        control_key="is_control",
        match_context=["cell_line"],
        rep_tables=rep_tables,
        min_runs_per_leaf=min_runs_per_leaf,
    )
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_delta = _rss_bytes() - rss0
    n_pert = len(compiled.scheme.nodes["pert"].weights)
    n_ctrl = len(compiled.scheme.nodes["ctrl"].weights)
    return dt, peak / 1e6, rss_delta, (n_pert, n_ctrl)


def run_phase_breakdown(source_path: str, *, encoder: str, n_drugs: int, min_runs_per_leaf: int) -> list[Phase]:
    """Reproduce each hot phase of compile_obs verbatim on the same obs frame, timed individually."""
    from binded._io import leaf_codes, obs_columns, open_source

    state, condition = _schemas(encoder)
    rep_tables = _rep_tables(encoder, n_drugs)
    cols = ("cell_line", "drug")  # match_context + condition (compile_obs `cols`)
    control_key = "is_control"
    data_cols = [*cols, control_key]

    phases: list[Phase] = []

    # 1) open_source + obs_columns  (_compile_obs.py:144-145)
    p_read = timed("open+obs_read", lambda: obs_columns(open_source(source_path, keys=["X"], cols=data_cols), data_cols))
    obs = p_read.result
    p_read.result = None
    phases.append(p_read)

    # 2) _fit_encoders: union materialization + Encoder.fit  (_compile_obs.py:147-155)
    encoder_obj = condition.condition_encoders["drug"]

    def _fit():
        union = obs[["drug"]].to_numpy().reshape(-1)  # O(rows) union array — the current code
        return copy.deepcopy(encoder_obj).fit(union, uns=rep_tables)

    phases.append(timed("encoder_fit(current)", _fit))

    # 2b) PROPOSED optimization: fit on the UNIQUE values, not all N rows. A one-hot/label/lookup
    # encoder only depends on the category space, so feeding uniques yields an identical encoder at
    # O(cardinality) cost instead of an O(rows) object-array materialization.
    def _fit_uniques():
        col = obs["drug"]
        uniques = col.cat.categories.to_numpy() if isinstance(col.dtype, pd.CategoricalDtype) else col.unique()
        return copy.deepcopy(encoder_obj).fit(np.asarray(uniques).reshape(-1), uns=rep_tables)

    phases.append(timed("encoder_fit(uniques)", _fit_uniques))

    # 3) control split  (_compile_obs.py:204)
    p_ctrl = timed("ctrl_split", lambda: obs[control_key].to_numpy().astype(bool))
    ctrl_flag = p_ctrl.result
    p_ctrl.result = None
    phases.append(p_ctrl)

    pert_obs = obs.loc[~ctrl_flag]
    ctrl_obs = obs.loc[ctrl_flag]

    # 4) _target_leaves: the Counter tuple-loop  (_compile_obs.py:195-199)
    def _target_leaves():
        counts = Counter(tuple(r) for r in pert_obs.loc[:, list(cols)].to_numpy())
        return [leaf for leaf, cnt in counts.items() if cnt >= min_runs_per_leaf]

    p_tl = timed("target_leaves(Counter)", _target_leaves)
    n_pert_leaves = len(p_tl.result)
    p_tl.result = None
    phases.append(p_tl)

    # 5) _leaves: the vectorized drop_duplicates path used for controls  (_compile_obs.py:190)
    p_lv = timed(
        "leaves(drop_duplicates)",
        lambda: [tuple(r) for r in ctrl_obs.loc[:, list(cols)].drop_duplicates().to_numpy()],
    )
    p_lv.result = None
    phases.append(p_lv)

    # 6) PROPOSED optimization: vectorized target-leaf counting via binded.leaf_codes + bincount
    def _target_leaves_vectorized():
        codes, leaves = leaf_codes(pert_obs, list(cols))  # C-level factorize, no per-row python tuples
        counts = np.bincount(codes, minlength=len(leaves))
        return [lf for lf, cnt in zip(leaves, counts, strict=True) if cnt >= min_runs_per_leaf]

    p_vec = timed("target_leaves(leaf_codes)", _target_leaves_vectorized)
    n_vec_leaves = len(p_vec.result)
    p_vec.result = None
    phases.append(p_vec)

    # 6b) alternative vectorized count: pandas groupby.size() (C-level, categorical-aware)
    def _target_leaves_groupby():
        counts = pert_obs.groupby(list(cols), observed=True).size()
        kept = counts[counts >= min_runs_per_leaf]
        return list(kept.index)

    p_gb = timed("target_leaves(groupby)", _target_leaves_groupby)
    n_gb_leaves = len(p_gb.result)
    p_gb.result = None
    phases.append(p_gb)

    # sanity: all target-leaf paths keep the same number of leaves
    if not (n_pert_leaves == n_vec_leaves == n_gb_leaves):
        print(f"  [warn] leaf-count mismatch Counter={n_pert_leaves} leaf_codes={n_vec_leaves} groupby={n_gb_leaves}")

    del obs, pert_obs, ctrl_obs, ctrl_flag
    gc.collect()
    return phases


# --------------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------------
def _row_count(source_path: str) -> int:
    from binded._io import obs_columns, open_source

    return len(obs_columns(open_source(source_path, keys=["X"], cols=["is_control"]), ["is_control"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="100000,1000000,10000000", help="comma-separated N obs values")
    ap.add_argument("--data", default=None, help="benchmark a real zarr adata / collection instead of synthetic")
    ap.add_argument("--encoder", default="lookup", choices=["lookup", "onehot", "label"],
                    help="condition encoder: lookup binds a table (cheap); onehot/label fit sklearn on the union")
    ap.add_argument("--n-cell-lines", type=int, default=DEFAULT_CELL_LINES)
    ap.add_argument("--n-drugs", type=int, default=DEFAULT_DRUGS)
    ap.add_argument("--control-frac", type=float, default=DEFAULT_CONTROL_FRAC)
    ap.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS, help="obs rows per collection shard")
    ap.add_argument("--single-zarr", action="store_true", help="one backed zarr adata instead of a collection")
    ap.add_argument("--min-runs-per-leaf", type=int, default=0)
    ap.add_argument("--cache-dir", default=os.environ.get("BENCH_CACHE_DIR", os.path.join(os.getcwd(), ".bench_cache")))
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"cache-dir: {args.cache_dir}  encoder: {args.encoder}  control_frac: {args.control_frac}")
    print(f"cardinalities: cell_line={args.n_cell_lines} drug={args.n_drugs}  shard_rows={args.shard_rows:,}\n")

    if args.data is not None:
        sources = [(_row_count(args.data), args.data)]
    else:
        sizes = [int(float(x)) for x in args.sweep.split(",")]
        sources = []
        for n in sizes:
            print(f"[gen] preparing synthetic source for N={n:,} ...", flush=True)
            t0 = time.perf_counter()
            path = build_synthetic_source(
                n, cache_dir=args.cache_dir, n_cell_lines=args.n_cell_lines, n_drugs=args.n_drugs,
                control_frac=args.control_frac, shard_rows=args.shard_rows, single_zarr=args.single_zarr,
            )
            print(f"      ready in {time.perf_counter() - t0:.1f}s -> {path}", flush=True)
            sources.append((n, path))

    results = []
    for n, path in sources:
        print(f"\n=== N = {n:,} obs  ({path}) ===", flush=True)
        full_dt, full_peak_mb, rss_delta, (n_pert, n_ctrl) = run_full_compile(
            path, encoder=args.encoder, n_drugs=args.n_drugs, min_runs_per_leaf=args.min_runs_per_leaf
        )
        print(f"  compile_obs total: {full_dt:.3f}s  py_peak={full_peak_mb:,.0f}MB  rssΔ={_fmt_bytes(rss_delta)}"
              f"  (pert_leaves={n_pert}, ctrl_leaves={n_ctrl})", flush=True)

        phases = run_phase_breakdown(
            path, encoder=args.encoder, n_drugs=args.n_drugs, min_runs_per_leaf=args.min_runs_per_leaf
        )
        print(f"  {'phase':<28}{'seconds':>12}{'py_peak_MB':>14}{'µs/1k rows':>14}")
        for ph in phases:
            us_per_1k = ph.seconds / n * 1e6 * 1000
            print(f"  {ph.name:<28}{ph.seconds:>12.4f}{ph.py_peak_mb:>14.1f}{us_per_1k:>14.2f}", flush=True)
        results.append((n, full_dt, full_peak_mb, rss_delta, {ph.name: ph for ph in phases}))

    # ---- scaling summary ----
    print("\n\n================ SCALING SUMMARY ================")
    print(f"{'rows':>13}{'total_s':>10}{'py_peak_MB':>12}{'rss_delta':>12}"
          f"{'read_s':>9}{'fit_s':>9}{'Counter_s':>11}{'gby_s':>9}")
    for n, full_dt, full_peak_mb, rss_delta, ph in results:
        def s(name):
            return ph[name].seconds if name in ph else float("nan")
        print(f"{n:>13,}{full_dt:>10.3f}{full_peak_mb:>12.0f}{_fmt_bytes(rss_delta):>12}"
              f"{s('open+obs_read'):>9.3f}{s('encoder_fit(current)'):>9.3f}"
              f"{s('target_leaves(Counter)'):>11.3f}{s('target_leaves(groupby)'):>9.3f}")

    if len(results) >= 2:
        (n0, dt0, *_), (n1, dt1, *_) = results[0], results[-1]
        ph1 = results[-1][4]

        def spd(cur: str, opt: str) -> None:
            a, b = ph1.get(cur), ph1.get(opt)
            if a and b:
                print(f"  {cur} {a.seconds:.3f}s -> {opt} {b.seconds:.3f}s "
                      f"= {a.seconds / max(b.seconds, 1e-9):.1f}x faster (peak {a.py_peak_mb:.0f}->{b.py_peak_mb:.0f}MB)")

        print(f"\nscaling {n0:,}->{n1:,} ({n1 / n0:.0f}x rows): total {dt1 / max(dt0, 1e-9):.1f}x")
        counter0 = results[0][4].get("target_leaves(Counter)")
        counter1 = ph1.get("target_leaves(Counter)")
        if counter0 and counter1:
            print(f"  Counter tuple-loop {counter1.seconds / max(counter0.seconds, 1e-9):.1f}x scaling "
                  f"({counter0.seconds:.3f}s -> {counter1.seconds:.3f}s)")
        print(f"\nOPTIMIZATIONS at {n1:,} obs:")
        spd("encoder_fit(current)", "encoder_fit(uniques)")
        spd("target_leaves(Counter)", "target_leaves(leaf_codes)")
        spd("target_leaves(Counter)", "target_leaves(groupby)")


if __name__ == "__main__":
    main()
