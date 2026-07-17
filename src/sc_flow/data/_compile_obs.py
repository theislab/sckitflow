"""Obs-only ``compile_obs`` — labels → dagloader ``Scheme`` + ``condition_fn``.

Replaces the flat ``prepare_data`` blob with the composed schema objects
(:class:`StateDataSchema` / :class:`ConditionDataSchema` / :class:`GroupsDataSchema`)
and compiles them **off ``obs`` (+ the ``uns`` embedding tables) only — cells are never
read here; they are streamed later by dagloader. This mirrors cellflow's
``build_annbatch_training`` but the condition encoder is sc_flow's own
:class:`CategoricalData`.

Two condition mechanisms (see the design note):

* **leaf-level** categorical/combinatorial covariates → the returned ``condition_fn``
  (a per-leaf lookup, constant within a class-coherent batch);
* **per-cell** "paired" covariates → extra ``Node`` keys (streamed aligned to the state
  cells), *not* handled here — pass them as additional ``state`` reps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.schemas._condition_data_schema import ConditionDataSchema
from sc_flow.data.schemas._groups_data_schema import GroupsDataSchema
from sc_flow.data.schemas._state_data_schema import StateDataSchema

__all__ = ["compile_obs", "CompiledData"]

Leaf = tuple[Any, ...]
ConditionFn = Callable[[Leaf], dict[str, np.ndarray]]


@dataclass(frozen=True)
class CompiledData:
    """Result of :func:`compile_obs` — everything dagloader needs, built from labels."""

    scheme: Any  # dagloader.Scheme
    condition_fn: ConditionFn
    cols: tuple[str, ...]
    data_dim: int | None = None


def _sample_rep_to_key(sample_rep: str) -> str:
    """``sample_rep`` → dagloader rep key (``"X"`` or ``"obsm/<rep>"``)."""
    return "X" if sample_rep == "X" else f"obsm/{sample_rep}"


def compile_obs(
    adata: Any,
    *,
    state: StateDataSchema,
    condition: ConditionDataSchema,
    groups: GroupsDataSchema | None = None,
    control_key: str,
    split_covariates: Sequence[str] = (),
) -> CompiledData:
    """Compile the composed schemas into a dagloader ``Scheme`` + ``condition_fn`` from obs only.

    :param adata: Source with ``.obs`` (labels) and ``.uns`` (embedding tables). Cells
        (``.X`` / ``.obsm``) are NOT read here — dagloader streams them at train time.
    :param state: Which representation to stream (becomes the ``Node`` key).
    :param condition: The leaf-level (categorical/combinatorial) condition covariates.
    :param groups: Embedded *sample* covariates (require a rep/encoding); ``None`` = none.
    :param control_key: Boolean/0-1 obs column marking control observations.
    :param split_covariates: Matching-context columns → the ``Bind.common`` (not embedded).
    """
    from dagloader import Bind, Node, Scheme, uniform

    obs: pd.DataFrame = adata.obs
    uns = getattr(adata, "uns", {}) or {}

    cond_cols = list(condition.all_condition_cols)
    group_cols = list(groups.groups) if groups is not None else []
    # grouping columns: matching context (split) + condition + embedded sample covariates,
    # deduped, order-preserving (context first) — matches cellflow's `cols` ordering.
    cols = tuple(dict.fromkeys([*split_covariates, *cond_cols, *group_cols]))
    key = _sample_rep_to_key(state.sample_rep)

    def _fit_encoders(frame_cols: list[str], repr_d: dict, reps_m: dict) -> dict[str, Any]:
        # Fit encoders ONCE on the full (deduped) obs category space, so a per-leaf single value
        # produces the full-width one-hot cellflow makes — not a dim-1 fit. Rep'd realms need none.
        if not frame_cols:
            return {}
        tmpl = CategoricalData.from_pandas(obs[frame_cols].drop_duplicates(), repr_dict=repr_d, categorical_reps_map=reps_m)
        return dict(tmpl.categorical_encoders)

    # --- perturbation levels: a level's columns are its combination slots (stacked) ---
    cond_repr = {level: uns[rep] for level, rep in condition.conditions_reps.items()}
    cond_reps_map = condition.categorical_reps_map
    cond_encoders = _fit_encoders(cond_cols, cond_repr, cond_reps_map)
    cond_idx = [cols.index(c) for c in cond_cols]

    # --- sample covariates: one value per sample, tiled across the max_comb slots (cellflow's np.tile) ---
    group_reps = groups.groups_reps if groups is not None else {}
    samp_repr = {c: uns[group_reps[c]] for c in group_cols if c in group_reps}
    samp_reps_map = {c: c for c in group_cols}  # each sample covariate is its own group
    samp_encoders = _fit_encoders(group_cols, samp_repr, samp_reps_map)
    samp_idx = [cols.index(c) for c in group_cols]

    # combination length = the (shared) perturbation-level column count; sample covariates tile to it.
    max_comb = max((len(v) for v in condition.conditions.values()), default=1)

    def condition_fn(leaf: Leaf) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if cond_cols:
            row = pd.DataFrame([{c: leaf[i] for c, i in zip(cond_cols, cond_idx, strict=True)}])
            cat = CategoricalData.from_pandas(
                row, repr_dict=cond_repr, categorical_encoders=cond_encoders, categorical_reps_map=cond_reps_map
            )
            out.update({k: np.asarray(v, dtype=np.float32) for k, v in cat.extract_reps().mapping.items()})
        if group_cols:
            srow = pd.DataFrame([{c: leaf[i] for c, i in zip(group_cols, samp_idx, strict=True)}])
            scat = CategoricalData.from_pandas(
                srow, repr_dict=samp_repr, categorical_encoders=samp_encoders, categorical_reps_map=samp_reps_map
            )
            for group, v in scat.extract_reps().mapping.items():
                v = np.asarray(v, dtype=np.float32)  # (1, 1, dim) — one sample value
                if max_comb > v.shape[1]:
                    v = np.repeat(v, max_comb, axis=1)  # tile across the max_comb combination slots
                out[group] = v
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
        binds=(Bind("pert", "ctrl", common=tuple(split_covariates)),),
        seed=0,
    )
    return CompiledData(scheme=scheme, condition_fn=condition_fn, cols=cols)
