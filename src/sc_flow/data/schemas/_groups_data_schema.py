from collections.abc import Collection, Mapping

import pandas as pd
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncoderCls
from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.data._group_encoders import GroupEncoder, GroupEncoderContext
from sc_flow.data._mixins import MappedArray
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["GroupsDataSchema"]


class GroupsDataSchema(StrictDataSchema):
    """Data Schema implementing the logic for base grouping."""

    def __init__(
        self,
        groups: Collection[str] | None = None,
        groups_reps: dict[str, str] | None = None,
        groups_encoding: dict[str, GroupEncoder] | None = None,
    ) -> None:
        """Initializes the data schema.

        :param groups: Collection of string identifiers indicating the
            columns in `.obs` used to define the grouping. Defaults to `None`
        :type groups: class: `Collection[str] | None`

        :param groups_reps: Dictionary mapping column identifiers in :param: `groups`
            to the corresponding key of `.uns` used to store the covariate representations.
            The corresponding value in `.uns` need to have keys matching the unique values
            of the corresponding column in `.obs`. Defaults to `None`.
        :type groups_reps: class: `dict[str, str] | None`

        :param groups_encoding: Dictionary mapping each group column to a
            :class:`~sc_flow.data._group_encoders.GroupEncoder` (e.g. ``OneHot()``, ``Label()``,
            ``Affine(scale=2.0)``). Encoders are frozen dataclasses that build their fitted
            transformer on demand, so they carry no callables and stay serializable. Defaults to `None`.
        :type groups_encoding: class: `dict[str, GroupEncoder] | None`
        """
        self._groups = [] if groups is None else groups
        self._groups_reps = {} if groups_reps is None else groups_reps
        self._groups_encoders: dict[str, GroupEncoder] = {} if groups_encoding is None else groups_encoding
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
        for col, enc in self._groups_encoders.items():
            if not isinstance(enc, GroupEncoder):
                msg = f"groups_encoding[{col!r}] must be a GroupEncoder instance, got {type(enc).__name__}."
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

    def _get_covs_df(
        self,
        adata: AnnData,
    ) -> pd.DataFrame:
        """Retrieves the covariates data frame.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        return adata.obs.loc[:, self._groups]

    def _get_covs_repr_dict(
        self,
        adata: AnnData,
    ) -> dict[str, MappedArray]:
        """Retrieves the annotation dictionary for each column.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        return {col: adata.uns[rep] for col, rep in self._groups_reps.items()}

    def _get_data(self, adata: AnnData) -> CategoricalData:
        """Enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        covs_df_dict: pd.DataFrame = self._get_covs_df(adata)
        repr_dict: dict[str, MappedArray] = self._get_covs_repr_dict(adata)
        # Each encoder fits its own transformer from the column values at fit time.
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
    def groups_encoders(self) -> dict[str, GroupEncoder]:
        """Exposes the `groups_encoding` encoders set at initialization."""
        return self._groups_encoders

    @property
    def categorical_reps_map(self) -> dict[str, str]:
        """Dictionary mapping each categorical column to the corresponding realm for their representation."""
        return {group: group for group in self.groups}
