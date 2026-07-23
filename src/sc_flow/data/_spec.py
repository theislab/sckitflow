
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

    state: StateDataSchema
    condition: ConditionDataSchema
    control_key: str
    covariates: CovariatesDataSchema | None = None
    coupling: CouplingDataSchema | None = None
    match_context: Sequence[str] = field(default_factory=tuple)

    def compile(
        self,
        data: DataInput,
        *,
        rep_tables: Mapping[str, Mapping] | None = None,
        control_in_memory: bool = False,
        control_path: DataInput | None = None,
        min_runs_per_leaf: int = 0,
        seed: int = 0,
    ) -> CompiledData:
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
            seed=seed,
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
        seed: int = 0,
    ) -> Any:  # binded.Loader
        from scfit.data import Loader, SamplerConfig

        compiled = self.compile(
            data,
            rep_tables=rep_tables,
            control_in_memory=control_in_memory,
            control_path=control_path,
            min_runs_per_leaf=min_runs_per_leaf,
            seed=seed,
        )
        if preload_nchunks is None:
            preload_nchunks = max(1, batch_size // chunk_size)
        cfg = SamplerConfig(batch_size=batch_size, chunk_size=chunk_size, preload_nchunks=preload_nchunks, to=to)
        return Loader(compiled.scheme, cfg, compiled.condition_lookup)
