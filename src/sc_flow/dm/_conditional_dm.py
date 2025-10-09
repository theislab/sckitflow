from collections import defaultdict
from collections.abc import Sequence
from typing import NewType

import numpy as np
from anndata import AnnData

from sc_flow._constants import (
    SOURCE_COUPLING_STATE_LIN,
    SOURCE_COUPLING_STATE_QUAD,
    TARGET_COUPLING_STATE_LIN,
    TARGET_COUPLING_STATE_QUAD,
)
from sc_flow._types import CouplingSpaceReps, MappedArray
from sc_flow.dm._unconditional_dm import UnconditionalDataManager

## TODO: remove when felix writes this stuff
StateData = NewType("StateData", np.array)
ConditionData = NewType("ConditionData", dict[str, MappedArray])
TargetData = NewType("TargetData", MappedArray)
CouplingData = NewType("CouplingData", MappedArray)


class ConditionalDataManager(UnconditionalDataManager):
    """"""  # noqa

    def __init__(
        self,
        sample_rep: str | None = None,
        coupling_reps: dict[CouplingSpaceReps, str] | None = None,
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

        :param coupling_reps: (Optional) Dictionary mapping each term needed for the GENOT
            coupling over incomparable spaces to its representation in `.obsm`.
            It has to contain three keys `("xy", "xx", "yy")` which will be used to compute
            such distances over only partially comparable spaces. Only used when the
            method used dowstream is GENOT, ignored otherwise. Defaults to `None`.
        :type coupling_reps: class: `dict[CouplingSpaceReps, str] | None`

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
        self._coupling_reps = coupling_reps
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

    def _validate_coupling_reps(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the coupling representations settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        if self._coupling_reps is None:
            msg = ""
            raise ValueError(msg)
        if not self.has_controls:
            msg = ""
            raise ValueError(msg)
        for term_id, term_rep in self._coupling_reps.items():
            if ("src" in term_id) or ("tgt" in term_id and term_rep is not None):
                self._check_key_found_in_adata_field(adata, term_rep, "obsm")

    def _validate_conditions(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the condition settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        for conditions in self._conditions.values():
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
        for condition_id in self._conditions.keys():
            if condition_id not in self._conditions:
                msg = f"Condition {condition_id} not found."
                raise ValueError(msg)
            self._check_key_found_in_adata_field(adata, condition_id, "uns")

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

    def get_coupling_data(
        self,
        adata: AnnData,
    ) -> CouplingData | None:
        """"""  # noqa
        self._validate_coupling_reps(adata)
        # source is required
        src_lin = adata.obsm[self._coupling_reps["src_coupling_lin"]]
        src_quad = adata.obsm[self._coupling_reps["src_coupling_quad"]]
        n_shared_dims = src_lin.shape[-1]
        # target is optional (we take it from .X and slice manually otherwise)
        tgt_lin_rep = self._coupling_reps.get("tgt_coupling_lin", None)
        tgt_quad_rep = self._coupling_reps.get("tgt_coupling_quad", None)
        state_data = self.get_state_data(adata)
        if (tgt_lin_rep is not None) and (tgt_quad_rep is not None):
            tgt_lin = adata.obsm[tgt_lin_rep]
            tgt_quad = adata.obsm[tgt_quad_rep]
            print(f"cdm::get_coupling_data::{tgt_lin.shape=}")
            print(f"cdm::get_coupling_data::{tgt_quad.shape=}")
            print(f"cdm::get_coupling_data::{state_data.shape=}")
            if tgt_lin.shape[-1] != n_shared_dims:
                msg = ""
                raise ValueError(msg)
            if tgt_lin.shape[-1] + tgt_quad.shape[-1] != state_data.shape[-1]:
                msg = ""
                raise ValueError(msg)
        else:
            if n_shared_dims >= state_data.shape[-1]:
                msg = ""
                raise ValueError(msg)
            tgt_lin = state_data[..., :n_shared_dims]
            tgt_quad = state_data[..., n_shared_dims:]

        return CouplingData(
            {
                SOURCE_COUPLING_STATE_LIN: src_lin,
                SOURCE_COUPLING_STATE_QUAD: src_quad,
                TARGET_COUPLING_STATE_LIN: tgt_lin,
                TARGET_COUPLING_STATE_QUAD: tgt_quad,
            }
        )

    def get_condition_data(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        self._validate_conditions(adata)
        reps_dict = self._get_conditions_reps(adata)
        covs_dict = self._get_conditions_covariates(adata)
        # TODO: the keys should be the condition realms
        return ConditionData({"reps": reps_dict, "covariates": covs_dict})

    @property
    def has_controls(
        self,
    ) -> bool:
        """Whether the data pertains a notion of control states."""
        return self._control_key is not None
