from collections.abc import Collection

import pandas as pd
from anndata import AnnData

from sc_flow._types import MappedArray
from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["ConditionDataSchema"]


class ConditionDataSchema(StrictDataSchema):
    """Data Schema Implementing the logic for conditioning."""

    def __init__(
        self,
        conditions: dict[str, Collection[str]] | None = None,
        conditions_reps: dict[str, str] | None = None,
        conditions_covariates: Collection[str] | None = None,
    ) -> None:
        """Initializes the condition data schema.

        The schema used to define conditioning information entails two types of data:

        * Combinatorial Discrete Covariates.

            ## Condition Levels.
            Condition levels represent the (combinatorial) axes over which the conditioning is applied.
            Each condition level will be determined by a set of columns, whose cardinality is associated
            to the number of combinatorial conditions considered for such conditioning axes. When possible,
            categorical condition covariates define the grouping.

            ## Storage Strategies.
            The unique values over the dataset for each categorical covariate will be associated to
            some pre-computed representation (i.e.: condition embedding).

            Since such categories usually distribute over a finite support with cardinality significantly
            lower than the one of the training data, the embeddings for each unique condition value are stored
            as a lookup table, mapping (hashable) identifiers for each unique values
            (stored as set of columns of `.obs`) to the corresponding representation (stored as in dictionaries in `.uns`).

            ## Practical Examples.
            In the example below we illustrate how to initialize the condition data schema for modeling combinatorial
            condition over two conditioning axes (i.e.: drug perturbations and genetic knockout).
            Each of these axes will be associated to a set of columns, defining the combinatorial nature
            of the conditioning over them. In the following example, we will specify the condition levels/axes
            by setting :param: `conditions`.

            ### Condition Levels

            * Drug Perturbation
                #### Level Covariates for Drug Level
                - Drug A
                - Drug B

            * Genetic KnockOut
                #### Level Covariates for KO Level
                - KO A
                - KO B

            Note that, while the choice of the name of the keys to pass as :param: `conditions` and :param: `conditions_reps`
            is arbitrary (as long as the two dictionaries share the same set of keys), the values of both arguments need
            to appear in the input `AnnData`.

            In particular, the values of :param:`conditions`, should be given by sequences of string identifier
            of columns in `adata.obs`` for each conditioning level.

            On the other hand, the values of :param:`conditions_reps` should be given by a single string identifier,
            specifying the key in `adata.uns` containing the representation for each level.

            .. code-block:: python
                >>> # importing libraries
                >>> from sc_flow.data import schemas, sim
                >>> # annotated data
                >>> adata = sim.get_dummy_adata()
                ... AnnData object with n_obs × n_vars = 60000 × 400
                ... obs: 'drugA', 'drugB', 'koA', 'koB', 'target', 'source_split', 'is_control'
                ... uns: 'drug', 'ko', 'source_split'
                ... obsm: 'drugA_time', 'drugA_dose', 'drugB_time', 'drugB_dose', 'koA_time', 'koA_dose', \
                ...     'koB_time', 'koB_dose', 'paired_condition', 'X_repr', 'target_variable', 'X_src', 'X_tgt'
                >>> condition_schema = schemas.ConditionDataSchema(
                ...     conditions={
                ...         "drug_perturbation":["drugA", "drugB"],
                ...         "genetic_ko":["koA", "koB"],
                ...     },
                ...     conditions_reps={
                ...         "drug_perturbation": "drug",
                ...         "genetic_ko": "ko",
                ...     }
                ... )
                >>> condition_data = condition_schema.get_data(adata)
                >>> type(condition_data)
                ... sc_flow.data._structures.MixedTypeData

        * Multi-Dimensional Continuous Covariates.

            ## Continuous Condition Covariates.
            These are the dense covariates associated to each observation.

            Conditions should be modeled in such a way when each observation is associated to its unique
            and individual condition value. This is the reason why this type of condition covariates
            is referred to as "paired" conditioning.

            In a continuous space, the density at each given point is degenerate, hence
            using such covariates to define the grouping would entail infinitely many different groups.

            For this reason, grouping over conditions when they comprise continuous covariates is not supported.
            In such case, all the conditions will constitute a joint "conditional" groups for sampling.

            ## Storage Strategies.
            We cannot avoid in this case to store the full array as a lookup table, as there will be no duplicate
            conditions. For this reason, continuous conditioning covariates are assumed to be stored as arrays in `.obsm`.

            ## Practical Examples.
            We will continue from the example above and use the data stored in the `"paired_condition"` key of `adata.obsm`
            as dense condition covariates.
            In order to model this type of conditioning is necessary to add the :param: `conditions_covariates` argument
            at initialization, specifying the sequence of key identifier to be used as condition.

            .. code-block:: python
                >>> condition_schema = schemas.ConditionDataSchema(
                ...     conditions={
                ...         "drug_perturbation":["drugA", "drugB"],
                ...         "genetic_ko":["koA", "koB"],
                ...     },
                ...     conditions_reps={
                ...         "drug_perturbation": "drug",
                ...         "genetic_ko": "ko",
                ...     },
                ...     conditions_covariates=["paired_condition"],
                ... )
                >>> condition_data = condition_schema.get_data(adata)
                >>> type(condition_data)
                ... sc_flow.data._structures.MixedTypeData

            Naturally, it is also possible to only use the continuous covariates, when there is no additional information
            available for the conditioning. To achieve this, all it is required is to drop all the arguments associated to
            discrete conditioning covariates, while keeping the arguments for the continuous ones:

            .. code-block:: python
                >>> condition_schema = schemas.ConditionDataSchema(
                ...     conditions_covariates=["paired_condition"],
                ... )
                >>> condition_data = condition_schema.get_data(adata)
                >>> type(condition_data)
                ... sc_flow.data._structures.MixedTypeData

        :param conditions: Mapping from each condition level to the corresponding columns, as described in
            the dedicated section above. Defaults to `None`.
        :type conditions: class: `dict[str, Collection[str]] | None`

        :param conditions_reps: Mapping from each condition level to the corresponding representation, as described in
            the dedicated section above. Defaults to `None`.
        :type conditions_reps: class: `dict[str, str] | None`

        :param conditions_covariates: Sequence of continuous condition covariates, as described in the
            dedicated section above. Defaulst to `None`.
        :type conditions_covariates: class: `Collection[str] | None`
        """
        self._conditions = {} if conditions is None else conditions
        self._conditions_reps = {} if conditions_reps is None else conditions_reps
        self._conditions_covariates = () if conditions_covariates is None else conditions_covariates
        super().__init__()

    def _verify_args(self) -> None:
        """Verifies that :attr:`self.conditions` and :attr:`self.condition_reps` share the same keys."""
        check_sequence_query_against_reference(
            self.conditions_reps.keys(),
            self.conditions.keys(),
            allow_missing_from_query=False,
            allow_missing_from_reference=False,
        )

    def _verify_categorical_covariates(self, adata: AnnData) -> None:
        """Verifies the categorical condition covariates setting on the input `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        for condition_realm, condition_rep in self._conditions_reps.items():
            condition_cols = self._conditions[condition_realm]
            for col in condition_cols:
                self._check_key_found_in_adata_field(adata, col, "obs")
            self._check_key_found_in_adata_field(adata, condition_rep, "uns")

    def _verify_continuous_covariates(self, adata: AnnData) -> None:
        """Verifies the continuous condition covariates on the input `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        for covariate in self._conditions_covariates:
            self._check_key_found_in_adata_field(adata, covariate, "obsm")

    def _get_covariates_df(self, adata: AnnData) -> dict[str, pd.DataFrame]:
        """Extracts the columns for all the condition level from the `.obs` attribute of the input `AnnData`

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        return adata.obs.loc[:, self.all_condition_cols]

    def _get_repr_dict(self, adata: AnnData) -> dict[str, MappedArray]:
        """Retrieves the representations for all condition levels from the input `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        return {
            condition_level: adata.uns[condition_repr]
            for condition_level, condition_repr in self._conditions_reps.items()
        }

    def _get_categorical_covariates(self, adata: AnnData) -> CategoricalData | None:
        """Retrieves the categorical condition covariates from the input `AnnData`.

        Categorical covariates are in this case represented as a lookup table
        using the :class:`pandas.DataFrame` initialized with `self._get_covariates_df`
        as keys  for each observation to look up in the representatoin dictonary
        extracted with `self._get_repr_dict`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        if not self.has_categorical_covariates:
            return None
        covariates_df = self._get_covariates_df(adata)
        repr_dict = self._get_repr_dict(adata)
        return CategoricalData(covariates_df, repr_dict=repr_dict)

    def _get_continuous_covariates(self, adata: AnnData) -> BatchMixin | None:
        """Retrieves the continuous condition covariates from the input `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        if not self.has_continuous_covariates:
            return None
        return BatchMixin(
            {covariate_name: adata.obsm[covariate_name] for covariate_name in self._conditions_covariates}
        )

    def _verify_schema(self, adata: AnnData) -> None:
        """Verifies that input data satisfies the requirements defined by the schema.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        self._verify_categorical_covariates(adata)
        self._verify_continuous_covariates(adata)

    def _get_data(self, adata: AnnData) -> MixedTypeData | None:
        """Retrieves the schema-resolved condition information from the input data.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        categorical_covariates = self._get_categorical_covariates(adata)
        continuous_covariates = self._get_continuous_covariates(adata)
        if categorical_covariates is None and continuous_covariates is None:
            return None
        return MixedTypeData(categorical_covariates=categorical_covariates, continuous_covariates=continuous_covariates)

    @property
    def all_condition_cols(self) -> tuple[str]:
        """Identifiers for the condition columns of all leves returned as a tuple."""
        return tuple(cat for condition in self._conditions.values() for cat in condition)

    @property
    def condition_col_to_level(self) -> dict[str, str]:
        """Dictionary mapping each columns to the condition level it is associated with."""
        col2level = {}
        for condition, condition_cols in self._conditions.items():
            for cat in condition_cols:
                col2level[cat] = condition
        return col2level

    @property
    def allows_grouping(self) -> bool:
        """Whether the condition schema allows grouping.

        This will be false whenever continuous condition covariates are provided.
        """
        return len(self._conditions_covariates) == 0

    @property
    def conditions(self) -> dict[str, Collection[str]]:
        """Exposes to `conditions` parameter set at initialization."""
        return self._conditions

    @property
    def conditions_reps(self) -> dict[str, str]:
        """Exposes to `conditions_reps` parameter set at initialization."""
        return self._conditions_reps

    @property
    def conditions_covariates(self) -> Collection[str]:
        """Exposes to `conditions_covariates` parameter set at initialization."""
        return self._conditions_covariates

    @property
    def has_categorical_covariates(self) -> bool:
        """Whether the condition schema includes categorical covariates.

        This will be `True` whenever :param: `conditions` is set at initializtion
        """
        return len(self._conditions) > 0

    @property
    def has_continuous_covariates(self) -> bool:
        """Whether the condition schema includes categorical covariates.

        This will be `True` whenever :param: `conditions_covariates` is set at initializtion
        """
        return len(self._conditions_covariates) > 0
