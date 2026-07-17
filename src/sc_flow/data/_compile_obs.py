"""Obs-only ``compile_obs`` — labels → binded ``Scheme`` + ``condition_fn``.

Replaces the flat ``prepare_data`` blob with the composed schema objects
(:class:`StateDataSchema` / :class:`ConditionDataSchema` / :class:`CovariatesDataSchema`)
and compiles them **off ``obs`` (+ the ``uns`` embedding tables) only — cells are never
read here; they are streamed later by binded. This mirrors cellflow's
``build_annbatch_training`` but the condition encoder is sc_flow's own
:class:`CategoricalData`.

Two condition mechanisms (see the design note):

* **leaf-level** categorical/combinatorial covariates → the returned ``condition_fn``
  (a per-leaf lookup, constant within a class-coherent batch);
* **per-cell** "paired" covariates → extra ``Node`` keys (streamed aligned to the state
  cells), *not* handled here — pass them as additional ``state`` reps.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sc_flow.data._encoders import Encoder
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.schemas._condition_data_schema import ConditionDataSchema
from sc_flow.data.schemas._covariates_data_schema import CovariatesDataSchema
from sc_flow.data.schemas._state_data_schema import StateDataSchema

__all__ = ["compile_obs", "CompiledData"]

Leaf = tuple[Any, ...]
ConditionFn = Callable[[Leaf], dict[str, np.ndarray]]


@dataclass(frozen=True)
class CompiledData:
    """Result of :func:`compile_obs` — everything binded needs, built from labels."""

    scheme: Any  # binded.Scheme
    condition_fn: ConditionFn
    cols: tuple[str, ...]
    data_dim: int | None = None


def _sample_rep_to_key(sample_rep: str) -> str:
    """``sample_rep`` → binded rep key (``"X"`` or ``"obsm/<rep>"``)."""
    return "X" if sample_rep == "X" else f"obsm/{sample_rep}"


def compile_obs(
    adata: Any,
    *,
    state: StateDataSchema,
    condition: ConditionDataSchema,
    covariates: CovariatesDataSchema | None = None,
    control_key: str,
    match_context: Sequence[str] = (),
) -> CompiledData:
    """Compile the composed schemas into a binded ``Scheme`` + ``condition_fn`` from obs only.

    :param adata: Source with ``.obs`` (labels) and ``.uns`` (embedding tables). Cells
        (``.X`` / ``.obsm``) are NOT read here — binded streams them at train time.
    :param state: Which representation to stream (becomes the ``Node`` key).
    :param condition: The leaf-level (categorical/combinatorial) condition covariates.
    :param covariates: Embedded per-sample covariates (each with an encoder); ``None`` = none.
    :param control_key: Boolean/0-1 obs column marking control observations.
    :param match_context: Matching-context columns → the ``Bind.common`` (matching only, not
        embedded). sc-flow's native name for cellflow's ``split_covariates``.
    """
    from binded import Bind, Node, Scheme, uniform

    obs: pd.DataFrame = adata.obs
    uns = getattr(adata, "uns", {}) or {}

    cond_cols = list(condition.all_condition_cols)
    cov_cols = list(covariates.covariates) if covariates is not None else []
    # grouping columns: matching context + condition + embedded covariates,
    # deduped, order-preserving (context first) — matches cellflow's `cols` ordering.
    cols = tuple(dict.fromkeys([*match_context, *cond_cols, *cov_cols]))
    key = _sample_rep_to_key(state.sample_rep)

    def _fit_encoders(encoder_map: Mapping[str, Encoder], realm_to_cols: dict) -> dict[str, Encoder]:
        # Fit a COPY of each realm's encoder (never mutate the schema's objects): a lookup binds its
        # `.uns` table, a data-fit encoder fits ONCE on the union of the realm's columns' values across
        # obs — so a per-leaf single value yields the full category-space encoding, not a dim-1 fit.
        fitted: dict[str, Encoder] = {}
        for realm, encoder in encoder_map.items():
            union = obs[list(realm_to_cols[realm])].to_numpy().reshape(-1)
            fitted[realm] = copy.deepcopy(encoder).fit(union, uns=uns)
        return fitted

    # --- perturbation levels: a level's columns are its combination slots (stacked) ---
    cond_reps_map = condition.categorical_reps_map
    cond_encoders = _fit_encoders(condition.condition_encoders, condition.conditions)
    cond_idx = [cols.index(c) for c in cond_cols]

    # --- embedded covariates: one value per sample, tiled across the max_comb slots (cellflow's np.tile) ---
    cov_reps_map = {c: c for c in cov_cols}  # each covariate is its own realm
    cov_encoders = (
        _fit_encoders(covariates.covariate_encoders, {c: [c] for c in cov_cols}) if covariates is not None else {}
    )
    cov_idx = [cols.index(c) for c in cov_cols]

    # combination length = the (shared) perturbation-level column count; covariates tile to it.
    max_comb = max((len(v) for v in condition.conditions.values()), default=1)

    def condition_fn(leaf: Leaf) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if cond_cols:
            row = pd.DataFrame([{c: leaf[i] for c, i in zip(cond_cols, cond_idx, strict=True)}])
            cat = CategoricalData.from_pandas(row, encoders=cond_encoders, categorical_reps_map=cond_reps_map)
            out.update({k: np.asarray(v, dtype=np.float32) for k, v in cat.extract_reps().mapping.items()})
        if cov_cols:
            srow = pd.DataFrame([{c: leaf[i] for c, i in zip(cov_cols, cov_idx, strict=True)}])
            scat = CategoricalData.from_pandas(srow, encoders=cov_encoders, categorical_reps_map=cov_reps_map)
            for cov, v in scat.extract_reps().mapping.items():
                v = np.asarray(v, dtype=np.float32)  # (1, 1, dim) — one covariate value
                if max_comb > v.shape[1]:
                    v = np.repeat(v, max_comb, axis=1)  # tile across the max_comb combination slots
                out[cov] = v
        return out

    ctrl_flag = obs[control_key].to_numpy().astype(bool)
    pert = [tuple(r) for r in obs.loc[~ctrl_flag, list(cols)].drop_duplicates().to_numpy()]
    ctrl = [tuple(r) for r in obs.loc[ctrl_flag, list(cols)].drop_duplicates().to_numpy()]

    scheme = Scheme(
        sources={"data": adata},
        nodes={
            "pert": Node("data", cols, key, uniform(pert)),
            "ctrl": Node("data", cols, key, uniform(ctrl)),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=tuple(match_context)),),
        seed=0,
    )
    return CompiledData(scheme=scheme, condition_fn=condition_fn, cols=cols)
