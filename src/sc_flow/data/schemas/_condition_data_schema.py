from collections.abc import Collection, Mapping

from anndata import AnnData

from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.core.data._encoders import Encoder, Lookup
from sc_flow.core.data.schemas._base_schema import StrictDataSchema

__all__ = ["ConditionDataSchema"]


class ConditionDataSchema(StrictDataSchema):
    """Data schema for conditioning — combinatorial categorical covariates plus continuous ones.

    Discrete conditioning is organized into **condition levels** (the combinatorial axes). Each level
    maps to a set of ``.obs`` columns (its combination slots) via :param:`conditions`, and to a single
    :class:`~sc_flow.core.data._encoders.Encoder` via :param:`condition_encoders`. After Change 2 there is
    no ``reps`` vs ``encoding`` split — a ``.uns`` lookup is just a
    :func:`~sc_flow.core.data._encoders.lookup` encoder, so every level carries exactly one encoder.

    Continuous "paired" covariates (one dense value per observation, stored in ``.obsm``) are declared
    with :param:`conditions_covariates`; they disable grouping (each observation is its own condition).

    Example::

        >>> from sc_flow.core.data._encoders import lookup, one_hot
        >>> ConditionDataSchema(
        ...     conditions={"drug_perturbation": ["drugA", "drugB"], "genetic_ko": ["koA", "koB"]},
        ...     condition_encoders={"drug_perturbation": lookup("drug"), "genetic_ko": one_hot()},
        ... )

    :param conditions: Mapping ``{level: columns}`` — the combination slots of each condition level.
    :param condition_encoders: Mapping ``{level: Encoder}`` — exactly one encoder per condition level.
    :param conditions_covariates: Continuous ``.obsm`` condition covariates (paired conditioning).
    """

    def __init__(
        self,
        conditions: dict[str, Collection[str]] | None = None,
        condition_encoders: Mapping[str, Encoder] | None = None,
        conditions_covariates: Collection[str] | None = None,
    ) -> None:
        self._conditions = {} if conditions is None else conditions
        self._condition_encoders = {} if condition_encoders is None else condition_encoders
        self._conditions_covariates = () if conditions_covariates is None else conditions_covariates
        super().__init__()

    def _verify_args(self) -> None:
        """Every condition level must declare exactly one encoder (and vice versa)."""
        check_sequence_query_against_reference(
            self._conditions.keys(),
            self._condition_encoders.keys(),
            allow_missing_from_query=False,
            allow_missing_from_reference=False,
        )

    def _verify_categorical_covariates(self, adata: AnnData) -> None:
        """Every condition column must exist in ``obs``; every lookup encoder's table must be in ``uns``."""
        for condition_cols in self._conditions.values():
            for col in condition_cols:
                self._check_key_found_in_adata_field(adata, col, "obs")
        for encoder in self._condition_encoders.values():
            if isinstance(encoder, Lookup):
                self._check_key_found_in_adata_field(adata, encoder.uns_key, "uns")

    def _verify_continuous_covariates(self, adata: AnnData) -> None:
        """Verifies the continuous condition covariates on the input `AnnData`."""
        for covariate in self._conditions_covariates:
            self._check_key_found_in_adata_field(adata, covariate, "obsm")

    def _verify_schema(self, adata: AnnData) -> None:
        """Verifies that input data satisfies the requirements defined by the schema."""
        self._verify_categorical_covariates(adata)
        self._verify_continuous_covariates(adata)

    @property
    def all_condition_cols(self) -> tuple[str]:
        """Identifiers for the condition columns of all levels returned as a tuple."""
        return tuple(cat for condition in self._conditions.values() for cat in condition)

    @property
    def condition_col_to_level(self) -> dict[str, str]:
        """Dictionary mapping each column to the condition level it is associated with."""
        col2level = {}
        for condition, condition_cols in self._conditions.items():
            for cat in condition_cols:
                col2level[cat] = condition
        return col2level

    @property
    def allows_grouping(self) -> bool:
        """Whether the condition schema allows grouping (false when continuous covariates are provided)."""
        return len(self._conditions_covariates) == 0

    @property
    def conditions(self) -> dict[str, Collection[str]]:
        """Exposes to `conditions` parameter set at initialization."""
        return self._conditions

    @property
    def condition_encoders(self) -> Mapping[str, Encoder]:
        """Per-level encoder map (``lookup``/``one_hot``/``label``/``functional``)."""
        return self._condition_encoders

    @property
    def conditions_covariates(self) -> Collection[str]:
        """Exposes to `conditions_covariates` parameter set at initialization."""
        return self._conditions_covariates

    @property
    def has_categorical_covariates(self) -> bool:
        """Whether the condition schema includes categorical covariates."""
        return len(self._conditions) > 0

    @property
    def has_continuous_covariates(self) -> bool:
        """Whether the condition schema includes continuous covariates."""
        return len(self._conditions_covariates) > 0

    @property
    def categorical_reps_map(self) -> dict[str, str]:
        """Dictionary mapping each categorical column to the realm (condition level) of its representation."""
        reps_map = {}
        for realm, cov_list in self.conditions.items():
            for cov in cov_list:
                reps_map[cov] = realm
        return reps_map
