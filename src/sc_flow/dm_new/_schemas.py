from collections.abc import Collection
from dataclasses import dataclass


@dataclass
class ConditionSchema:
    """Implements the logic for conditioning."""

    conditions: dict[str, Collection[str]] | None = None
    conditions_reps: dict[str, str] | None = None
    conditions_covariates: Collection[str] | None = None

    @property
    def all_condition_categories(
        self,
    ) -> Collection[str]:
        """"""  # noqa
        if self.conditions is None:
            return ()
        return tuple(cat for condition in self.conditions.values() for cat in condition)

    @property
    def condition_category_to_realm(
        self,
    ) -> dict[str, str]:
        """"""  # noqa
        cat2realm = {}
        for condition, condition_cats in self.conditions.items():
            for cat in condition_cats:
                cat2realm[cat] = condition
        return cat2realm

    @property
    def allows_ot_coupling(
        self,
    ) -> bool:
        """"""  # noqa
        return self.conditions_covariates is None
