from collections.abc import Collection, Mapping
from dataclasses import dataclass

from sc_flow._types import TargetCovariatesEncoding
from sc_flow.data.contracts import ConditionDataContract, StateDataContract, TargetDataContract

__all__ = ["DataManager"]


@dataclass
class DataManager:
    """"""  # noqa

    sample_rep: str | None = None
    conditions: dict[str, Collection[str]] | None = None
    conditions_reps: dict[str, str] | None = None
    conditions_covariates: Collection[str] | None = None
    categorical_target_covariates: Mapping[str, TargetCovariatesEncoding] | None = None
    continuous_target_covariates: Collection[str] | None = None

    def __post_init__(self) -> None:
        """"""  # noqa
        self._state_data_contract = StateDataContract(
            sample_rep=self.sample_rep,
        )
        self._condition_data_contract = ConditionDataContract(
            conditions_reps=self.conditions_reps, conditions_covariates=self.conditions_covariates
        )
        self._target_data_contract = TargetDataContract(
            categorical_target_covariates=self.categorical_target_covariates,
            continuous_target_covariates=self.continuous_target_covariates,
        )

    @property
    def state_data_contract(self) -> StateDataContract:
        """"""  # noqa
        return self._state_data_contract

    @property
    def condition_data_contract(self) -> ConditionDataContract:
        """"""  # noqa
        return self._condition_data_contract

    @property
    def target_data_contract(self) -> ConditionDataContract:
        """"""  # noqa
        return self._condition_data_contract
