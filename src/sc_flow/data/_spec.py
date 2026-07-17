"""``FlowSpec`` — the declarative data spec that seeds a binded loader.

Bundles the composed schemas (state / condition / covariates / coupling) plus the matching context and
control key. It is *data-only* and speaks sc-flow's own vocabulary — no ``split_covariates`` /
``sample_covariates`` / ``perturbation_covariates``. ``build_loader(data, ...)`` compiles it against any
:data:`~sc_flow.data._compile_obs.DataInput` (in-memory ``AnnData``, out-of-core ``DatasetCollection``,
or zarr path(s)) and returns a :class:`binded.Loader`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sc_flow.data._compile_obs import CompiledData, compile_obs
from sc_flow.data.schemas._condition_data_schema import ConditionDataSchema
from sc_flow.data.schemas._coupling_data_schema import CouplingDataSchema
from sc_flow.data.schemas._covariates_data_schema import CovariatesDataSchema
from sc_flow.data.schemas._state_data_schema import StateDataSchema

if TYPE_CHECKING:
    from sc_flow.data._compile_obs import DataInput

__all__ = ["FlowSpec"]


@dataclass
class FlowSpec:
    """Declarative, source-agnostic data spec for a flow-matching problem.

    :param state: The streamed cell representation (:class:`StateDataSchema`).
    :param condition: Leaf-level condition covariates (:class:`ConditionDataSchema`).
    :param covariates: Embedded per-sample covariates (:class:`CovariatesDataSchema`); ``None`` = none.
    :param coupling: OT coupling references (:class:`CouplingDataSchema`); ``None`` = reuse state rep.
    :param control_key: Boolean/0-1 obs column marking control observations.
    :param match_context: Columns that define the matching context (source↔target pairing).
    """

    state: StateDataSchema
    condition: ConditionDataSchema
    covariates: CovariatesDataSchema | None = None
    coupling: CouplingDataSchema | None = None
    control_key: str = "control"
    match_context: Sequence[str] = field(default_factory=tuple)

    def compile(
        self,
        data: DataInput,
        *,
        rep_tables: Mapping[str, Mapping] | None = None,
        control_in_memory: bool = False,
        control_path: DataInput | None = None,
        min_runs_per_leaf: int = 0,
    ) -> CompiledData:
        """Compile the spec against ``data`` into a :class:`~sc_flow.data.CompiledData` (labels only)."""
        return compile_obs(
            data,
            state=self.state,
            condition=self.condition,
            covariates=self.covariates,
            coupling=self.coupling,
            control_key=self.control_key,
            match_context=self.match_context,
            rep_tables=rep_tables,
            control_in_memory=control_in_memory,
            control_path=control_path,
            min_runs_per_leaf=min_runs_per_leaf,
        )

    def build_loader(
        self,
        data: DataInput,
        *,
        rep_tables: Mapping[str, Mapping] | None = None,
        batch_size: int = 128,
        chunk_size: int = 1,
        preload_nchunks: int | None = None,
        to: str | None = "jax",
        control_in_memory: bool = False,
        control_path: DataInput | None = None,
        min_runs_per_leaf: int = 0,
    ) -> Any:  # binded.Loader
        """Compile against ``data`` and wrap in a :class:`binded.Loader`.

        ``rep_tables`` supplies the embedding tables for ``lookup`` encoders (see :func:`compile_obs`);
        it defaults to ``data.uns`` for an in-memory ``AnnData`` and is required for a collection/path.
        ``control_path`` (optional) streams controls from a separate source; ``min_runs_per_leaf`` drops
        tiny target leaves. ``preload_nchunks`` defaults to one batch per read window.
        """
        from binded import Loader, SamplerConfig

        compiled = self.compile(
            data,
            rep_tables=rep_tables,
            control_in_memory=control_in_memory,
            control_path=control_path,
            min_runs_per_leaf=min_runs_per_leaf,
        )
        if preload_nchunks is None:
            preload_nchunks = max(1, batch_size // chunk_size)
        cfg = SamplerConfig(batch_size=batch_size, chunk_size=chunk_size, preload_nchunks=preload_nchunks, to=to)
        return Loader(compiled.scheme, cfg, compiled.condition_fn)
