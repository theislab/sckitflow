import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, NewType

import numpy as np
import pandas as pd
from anndata import AnnData

try:
    from tqdm import tqdm
except ImportError:
    from sc_flow._runtime import set_tqdm_import_failed

    set_tqdm_import_failed(True)
    tqdm = NewType("tqdm", Iterable)

from sc_flow._constants import SOURCE_IDXS_KEY
from sc_flow._types import MappedArray
from sc_flow.data._data import (
    CombinationData,
    ConditionalDataset,
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
        cell_covariates: dict[str, str] | None = None,
        control_key: str | None = None,  # this key is not being checked
        conditions: dict[str, Sequence[str]] | None = None,
        conditions_reps: dict[str, str] | None = None,
        conditions_covariates: Sequence[str] | None = None,
        groups_obs_keys: Sequence[str] | None = None,
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

        :param conditions: (Optional) Dictionary mapping each condition realm to the respective categories.
            * "Condition realms" (i.e.: keys of the dictionary) are intended as the high level group the conditions belong to.
            That could be, for example, drugs or genetic perturbations, which can be applied multiple times in combination.
            They can have arbitrary name, with the only restrictions that they should all be present as keys in
            :param: `condition_reps`.

            * "Condition categories" (i.e.: values of the dictionay) instead represent the (possibly multiple)
            application of each condition axes on the observations. They should appear as columns in the `.obs` attribute
            of :class: `anndata.AnnData` objects to annotate. The target populations for generative modeling will be determined
            by the unique combinations over `.obs` of all the columns appearing here as values.

            When this dictionary is not passed, the returned conditions will be `None`, with the behavior of the class falling
            back to the unconditional case. Defaults to `None`.
        :type conditions:

        :param conditions_reps: (Optional) Dictionary mapping each condition realm to the key in `.obs` where their representation
            is stored in. Required when :param: `conditions` is provided. In that case, it should contain the same keys as
            :param: `conditions`.
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
            cell_covariates=cell_covariates,
        )
        self._control_key = control_key
        self._conditions = conditions
        self._conditions_reps = conditions_reps
        self._conditions_covariates = {} if conditions_covariates is None else conditions_covariates
        self._groups_obs_keys = groups_obs_keys

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

    def _obs_columns_to_combination_data(
        self,
        adata: AnnData,
        columns: Sequence[str],
        index: pd.Index | None = None,
    ) -> CombinationData:
        """"""  # noqa
        if index is None:
            index = adata.obs.index
        if not isinstance(columns, list):
            columns = list(columns)
        for col in columns:
            self._check_key_found_in_adata_field(adata, col, "obs")
        groups = adata.obs.groupby(by=columns)
        comb_values_to_indices = {}
        for group_val, obs_group in groups:
            group_idx = obs_group.index
            comb_values_to_indices[tuple(group_val)] = index.get_indexer(group_idx)
        # transforming back into tuple and returning condition data
        return CombinationData(
            columns,
            comb_values_to_indices,
        )

    def _validate_control_key(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        self._check_key_found_in_adata_field(adata, self._control_key, "obs")

    def _validate_conditions(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the condition settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        if self._conditions_reps is None:
            msg = "Conditions representations passed with `conditions_reps` are required, found `None`."
            raise ValueError(msg)
        for condition_realm, condition_categories in self._conditions.items():
            if condition_realm not in self._conditions_reps:
                msg = f"Representation not found for {condition_realm}."
                raise ValueError(msg)
            for condition_category in condition_categories:
                self._check_key_found_in_adata_field(adata, condition_category, "obs")

    def _validate_conditions_reps(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the condition representations settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        for condition_realm, condition_rep in self._conditions_reps.items():
            if condition_realm not in self._conditions:
                msg = f"Condition {condition_realm} not found."
                raise ValueError(msg)
            self._check_key_found_in_adata_field(adata, condition_rep, "uns")

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

    def _validate_groups_obs_keys(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the split covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        # sanity check
        for covariate in self._groups_obs_keys:
            self._check_key_found_in_adata_field(adata, covariate, "obs")

    def _get_conditions_reps(
        self,
        adata: AnnData,
    ) -> dict[str, MappedArray]:
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

    def _get_populations_indices(
        self,
        group_adata: AnnData,
        original_index: pd.Index | None = None,
    ) -> dict[str, np.ndarray | MappedArray]:
        """"""  # noqa
        # constructing results dictionary for current data
        results_dict = {}
        # handling control distriution
        if self.has_controls:
            # computing source mask and slicing adata to only perturbed states
            source_mask = group_adata.obs[self._control_key]
            source_idxs = group_adata.obs.index[source_mask]
            group_adata = group_adata[~source_mask]

            # retrieving source indices and storing them
            results_dict[SOURCE_IDXS_KEY] = source_idxs
        # computing combinations after optional slicing
        populations_data = self._obs_columns_to_combination_data(
            group_adata, self._all_condition_categories, index=original_index
        )
        results_dict.update(populations_data.comb_values_to_indices)
        return results_dict

    def _get_populations_indices_per_group(
        self,
        adata: AnnData,
    ) -> dict[tuple[Any], MappedArray]:
        """"""  # noqa
        if self._groups_obs_keys is None:
            return {None: self._obs_columns_to_combination_data(adata, self._all_condition_categories)}
        self._validate_groups_obs_keys(adata)
        # defining result dictionary
        results_dict = {}
        # retrieving groups data
        groups_data = self._obs_columns_to_combination_data(adata, self._groups_obs_keys)
        # iterating over each group
        for group_val, group_idxs in groups_data.comb_values_to_indices.items():
            # retrieving group adata
            group_adata = adata[group_idxs]
            # storing population indices
            results_dict[group_val] = self._get_populations_indices(group_adata, original_index=adata.obs.index)
        return results_dict

    def _get_condition_data(
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
        return ConditionData(reps_dict, covs_dict)

    def get_dataset(
        self,
        adata: AnnData,
    ) -> ConditionalDataset:
        """"""  # noqa
        # base dataset
        base_dataset = super().get_dataset(adata)

        # retrieving groups and condition data
        if self.has_controls:
            self._validate_control_key(adata)
        condition_data = self._get_condition_data(adata)
        if condition_data is not None:
            group_to_populations_indices = self._get_populations_indices_per_group(adata)
        else:
            group_to_populations_indices = None

        return ConditionalDataset.init_from_unconditional(
            base_dataset,
            group_to_populations_indices=group_to_populations_indices,
            condition_data=condition_data,
        )

    @property
    def has_controls(
        self,
    ) -> bool:
        """Whether the data pertains a notion of control states."""
        return self._control_key is not None
