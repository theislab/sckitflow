import logging
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from anndata import AnnData

from sc_flow._types import MappedArray
from sc_flow.data._data import (
    CombinationData,
    ConditionData,
)
from sc_flow.dm._unconditional_dm import UnconditionalDataManager

logger = logging.getLogger(__name__)


class ConditionalDataManager(UnconditionalDataManager):
    """"""  # noqa

    def __init__(
        self,
        sample_rep: str | None = None,
        categorical_target_covariates: Sequence[str] | None = None,
        continuous_target_covariates: Sequence[str] | None = None,
        control_key: str | None = None,  # this key is not being checked
        conditions: dict[str, Sequence[str]] | None = None,
        conditions_reps: dict[str, Sequence[str]] | None = None,
        conditions_covariates: Sequence[str] | None = None,
        split_covariates: Sequence[str] | None = None,
    ) -> None:
        """Initializes the conditional data manager.

        :param sample_rep: (Optional) Cell State representation on which the flow will be
            defined. When provided, it has to appear as a key in `.obsm`. Defaults to `None`,
            in which case the `.X` attribute will be used as representation.
        :type sample_rep: class: `str | None`

        :param categorical_target_covariates: (Optional) Dictionary used for definition of target covariates.
            Each key will represent a target covariate identified, that must appear as a column in `.obs`.
            The corresponding values must indicate the type of encoding to be done on them, that can be set to
            either `"label"` or `"one-hot"` for label and one hot encoding respectively. Defaults to `None`.
        :type categorical_target_covariates: class: `dict[str, TargetCovariatesEncoding] | None`

        :param continuous_target_covariates: (Optional) Sequence of string identifier of continuous target
            covariates to regress against. These covariates must appear as keys in `.obsm`. Defaults to `None`.
        :type continuous_target_covariates: class:  `Sequence[str]`

        :param control_key: (Optional) String identifier of the `.obs` column indicating whether
            an observation belongs to the control group. When `None`, it will assume no
            notion of control state is available for the current data. This could be the case
            when all observation have been subjected to some kind of perturbation. Defaults to `None`.
        :type control_key: class: `str | None`

        :param conditions:
        :type conditions:

        :param conditions_reps:
        :type conditions_reps:

        :param conditions_covariates:
        :type conditions_covariates:

        :param split_covariates:
        :type split_covariates:
        """
        super().__init__(
            sample_rep=sample_rep,
            categorical_target_covariates=categorical_target_covariates,
            continuous_target_covariates=continuous_target_covariates,
        )
        self._control_key = control_key
        self._conditions = conditions
        self._conditions_reps = conditions_reps
        self._conditions_covariates = {} if conditions_covariates is None else conditions_covariates
        self._split_covariates = split_covariates

    @property
    def _all_condition_categories(
        self,
    ) -> Sequence[str]:
        """"""  # noqa
        return tuple(cat for condition in self._conditions.values() for cat in condition)

    @property
    def _condition_category_to_realm(
        self,
    ) -> dict[str, str]:
        """"""  # noqa
        cat2realm = {}
        for condition, condition_cats in self._conditions.items():
            for cat in condition_cats:
                cat2realm[cat] = condition
        return cat2realm

    def _obs_columns_to_combination_data(self, adata: AnnData, columns: Sequence[str]) -> CombinationData:
        """"""  # noqa
        for col in columns:
            self._check_key_found_in_adata_field(adata, col, "obs")
        columns_values = adata.obs.loc[:, columns].values
        unique_values = np.unique(columns_values, axis=0)
        comb_id_to_unique_values = {comb_id: unique_values[comb_id] for comb_id in range(unique_values.shape[0])}
        comb_id_to_indices = {
            comb_id: np.argwhere(np.all(columns_values == unique_val, axis=-1))[:, 0]
            for comb_id, unique_val in comb_id_to_unique_values.items()
        }
        return CombinationData(
            columns,
            comb_id_to_unique_values,
            comb_id_to_indices,
        )

    def _validate_conditions(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the condition settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        for condition_id, conditions in self._conditions.items():
            if condition_id not in self._conditions_reps:
                msg = f"Representation not found for {condition_id}."
                raise ValueError(msg)
            for condition in conditions:
                self._check_key_found_in_adata_field(adata, condition, "obs")

    def _validate_conditions_reps(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the condition representations settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        for condition_id, condition_reps in self._conditions.items():
            if condition_id not in self._conditions:
                msg = f"Condition {condition_id} not found."
                raise ValueError(msg)
            for rep in condition_reps:
                self._check_key_found_in_adata_field(adata, rep, "uns")

    def _validate_conditions_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the condition covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        for condition_cat, covariates in self._conditions_covariates.items():
            if condition_cat not in self._all_condition_categories:
                msg = f"Condition category {condition_cat} not found."
                raise KeyError(msg)
            for covariate in covariates:
                self._check_key_found_in_adata_field(adata, covariate, "obsm")

    def _validate_split_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the split covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        if self._split_covariates is not None and self.has_controls:
            # sanity check
            for covariate in self._split_covariates:
                self._check_key_found_in_adata_field(adata, covariate, "obs")

            # retrieving unique source split values
            source_splits = adata[adata.obs[self._control_key]].obs[self._split_covariates].drop_duplicates()
            source_splits = map(tuple, source_splits)

            # retrieving corresponding target split values
            target_splits = adata[~adata.obs[self._control_key]].obs[self._split_covariates].drop_duplicates()
            target_splits = map(tuple, target_splits)

            # check that all source populations have corresponding target
            n_unmatched_source_splits = set(source_splits) - set(target_splits)
            if n_unmatched_source_splits > 0:
                msg = f"There are {n_unmatched_source_splits} source "
                raise ValueError(msg)

    def _get_conditions_reps(
        self,
        adata: AnnData,
    ) -> MappedArray:
        """"""  # noqa
        self._validate_conditions_reps(adata)
        return {condition_id: adata.uns[condition_rep] for condition_id, condition_rep in self._conditions_reps.items()}

    def _get_conditions_covariates(
        self,
        adata: AnnData,
    ) -> MappedArray:
        """"""  # noqa
        self._validate_conditions_covariates(adata)
        data_dict = defaultdict(list)
        for condition_cat, condition_covariates in self._conditions_covariates.items():
            realm = self._condition_category_to_realm[condition_cat]
            current_cat_covariate_data = np.concatenate(
                [adata.obsm[covariate] for covariate in condition_covariates], axis=-1
            )
            data_dict[realm].append(current_cat_covariate_data)
        data_dict = {realm: np.stack(cat_covariates, axis=-2) for realm, cat_covariates in data_dict.items()}
        return data_dict

    def get_condition_data(
        self,
        adata: AnnData,
    ) -> ConditionData:
        """"""  # noqa
        # eraly return if not provided
        # otheriwise performs sanity checks
        if self._conditions is None:
            return None
        self._validate_conditions(adata)
        # retrieving representations and covariates
        reps_dict = self._get_conditions_reps(adata)
        covs_dict = self._get_conditions_covariates(adata)
        # TODO: the keys should be the condition realms
        return ConditionData({"reps": reps_dict, "covariates": covs_dict})

    def get_split_data(
        self,
        adata: AnnData,
    ) -> CombinationData | None:
        """"""  # noqa
        # eraly return if not provided
        # otheriwise performs sanity checks
        if self._split_covariates is None:
            return None
        self._validate_split_covariates(adata)
        return self._obs_columns_to_combination_data(
            adata,
            self._split_covariates,
        )

    @property
    def has_controls(
        self,
    ) -> bool:
        """Whether the data pertains a notion of control states."""
        return self._control_key is not None
