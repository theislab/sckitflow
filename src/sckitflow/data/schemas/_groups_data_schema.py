from collections.abc import Collection, Mapping
from typing import Any

import pandas as pd
from anndata import AnnData

from sckitflow._types import TargetCovariatesEncoderCls
from sckitflow._utils import check_sequence_query_against_reference
from sckitflow.data._group_encoders import GroupEncoderConfig, GroupEncoderContext
from sckitflow.data._mixins import MappedArray
from sckitflow.data.containers._categorical import CategoricalData
from sckitflow.data.schemas._base_schema import StrictDataSchema

__all__ = ["GroupsDataSchema"]


class GroupsDataSchema(StrictDataSchema):
    """Data Schema implementing the logic for base grouping."""

    def __init__(
        self,
        groups: Collection[str] | None = None,
        groups_reps: dict[str, str] | None = None,
        groups_encoding: dict[str, GroupEncoderConfig] | None = None,
    ) -> None:
        """Initializes the data schema.

        :param groups: Collection of string identifiers indicating the
            columns in `.obs` used to define the grouping. Defaults to `None`
        :type groups: class: `Collection[str] | None`

        :param groups_reps: Dictionary mapping column identifiers in ``groups``
            to the corresponding key of `.uns` used to store the covariate representations.
            The corresponding value in `.uns` need to have keys matching the unique values
            of the corresponding column in `.obs`. Defaults to `None`.
        :type groups_reps: class: `dict[str, str] | None`

        :param groups_encoding: Dictionary mapping each group column to a
            :class:`~sckitflow.data._group_encoders.GroupEncoderConfig` (e.g.
            ``OneHotEncoderConfig(categories=("A549", "K562"))``, ``AffineTransformerConfig(scale=2.0)``).
            Encoders are frozen configs that build their fitted transformer on demand, so they carry no
            callables and stay serializable. Defaults to `None`.
        :type groups_encoding: class: `dict[str, GroupEncoderConfig] | None`
        """
        self._groups = [] if groups is None else groups
        self._groups_reps = {} if groups_reps is None else groups_reps
        self._groups_encoders: dict[str, GroupEncoderConfig] = {} if groups_encoding is None else groups_encoding
        super().__init__()

    def _verify_args(self) -> None:
        """Verifies the validity of the arguments set at initialization."""
        check_sequence_query_against_reference(
            self._groups,
            set(self._groups_reps.keys()).union(self._groups_encoders.keys()),
            allow_missing_from_query=False,
            allow_missing_from_reference=False,
        )
        shared_keys = set(self._groups_reps.keys()).intersection(self._groups_encoders.keys())
        if len(shared_keys):
            msg = "Each group column should have only one representation"
            raise ValueError(msg)

    def _verify_groups(self, adata: AnnData) -> None:
        """Verifies the :attr:`self.groups` attribute on the input data.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        for group in self._groups:
            self._check_key_found_in_adata_field(adata, group, "obs")

    def _verify_groups_reps(self, adata: AnnData) -> None:
        """Verifies the :attr:`self.groups_reps` attribute on the input data.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        for rep in self._groups_reps.values():
            self._check_key_found_in_adata_field(adata, rep, "uns")

    def _verify_schema(self, adata):
        """Verifies the schema on the input data.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        self._verify_groups(adata)
        self._verify_groups_reps(adata)

    def _get_data(self, adata: AnnData) -> CategoricalData:
        """Enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        return self.categorical_data(adata.obs, adata.uns)

    def categorical_data(self, obs: pd.DataFrame, uns: Mapping[str, Any]) -> CategoricalData:
        """The group covariates of ``obs`` rows, looked up in the ``uns`` tables or encoded.

        :param obs: Rows carrying the group columns, e.g. one group's values.
        :param uns: The ``.uns`` the representation tables are read from.
        """
        covs_df_dict: pd.DataFrame = obs.loc[:, list(self._groups)]
        repr_dict: dict[str, MappedArray] = {col: uns[rep] for col, rep in self._groups_reps.items()}
        # The categorical encoders map by their own vocabulary; fitting only checks the column against it.
        encoders_dict: Mapping[str, TargetCovariatesEncoderCls] = {
            col: enc.build(GroupEncoderContext(covs_df_dict.loc[:, col].values))
            for col, enc in self._groups_encoders.items()
        }
        return CategoricalData.from_pandas(
            covs_df_dict,
            repr_dict=repr_dict,
            categorical_encoders=encoders_dict,
            categorical_reps_map=self.categorical_reps_map,
        )

    @property
    def groups(self) -> Collection[str]:
        """Exposes to `groups` parameter set at initialization."""
        return self._groups

    @property
    def groups_reps(self) -> dict[str, str]:
        """Exposes to `groups_reps` parameter set at initialization."""
        return self._groups_reps

    @property
    def groups_encoders(self) -> dict[str, GroupEncoderConfig]:
        """Exposes the `groups_encoding` encoders set at initialization."""
        return self._groups_encoders

    @property
    def categorical_reps_map(self) -> dict[str, str]:
        """Dictionary mapping each categorical column to the corresponding realm for their representation."""
        return {group: group for group in self.groups}
