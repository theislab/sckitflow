from collections.abc import Callable, Collection, Mapping

import pandas as pd
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncoderCls, TargetCovariatesEncodingId
from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.data._mixins import MappedArray
from sc_flow.data._utils import get_covariates_encoders_from_dict
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["GroupsDataSchema"]


class GroupsDataSchema(StrictDataSchema):
    """Data Schema implementing the logic for base grouping."""

    def __init__(
        self,
        groups: Collection[str] | None = None,
        groups_reps: dict[str, str] | None = None,
        groups_encoding: dict[str, TargetCovariatesEncodingId] | None = None,
        groups_encoding_transform_fn: dict[str, Callable] | None = None,
        groups_encoding_inverse_transform_fn: dict[str, Callable] | None = None,
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

        :param groups_encoding: Dictionary mapping column identifiers in :param: `groups`
            to the corresponding label indicating the transformation to apply.
            Defaults to `None`.
        :type groups_encoding: class: `dict[str, TargetCovariatesEncodingId] | None`

        :param groups_encoding_transform_fn: Dictionary mapping column identifiers in :param: `groups`
            to the corresponding function used to define functional tranformations.
            This is only used for column identifiers that appear in :param: `groups_encoding` and whose
            corresponding value is `"fuctional"`. Defaults to `None`, in which case it will be set
            to the identity function.
        :type groups_encoding_transform_fn: class: `dict[str, Callable] | None`

        :param groups_encoding_inverse_transform_fn: Dictionary mapping column identifiers in :param: `groups`
            to the corresponding function used to define inverse functional tranformations.
            This is only used for column identifiers that appear in :param: `groups_encoding` and whose
            corresponding value is `"fuctional"`. Defaults to `None`, in which case it will be set
            to the identity function.
        :type groups_encoding_inverse_transform_fn: class: `dict[str, Callable] | None`
        """
        self._groups = [] if groups is None else groups
        self._groups_reps = {} if groups_reps is None else groups_reps
        self._groups_encoding = {} if groups_encoding is None else groups_encoding
        self._groups_encoding_transform_fn = (
            {} if groups_encoding_transform_fn is None else groups_encoding_transform_fn
        )
        self._groups_encoding_inverse_transform_fn = (
            {} if groups_encoding_inverse_transform_fn is None else groups_encoding_inverse_transform_fn
        )
        super().__init__()

    def _verify_args(self) -> None:
        """Verifies the validity of the arguments set at initialization."""
        check_sequence_query_against_reference(
            self._groups,
            set(self._groups_reps.keys()).union(set(self._groups_encoding.keys())),
            allow_missing_from_query=False,
            allow_missing_from_reference=False,
        )
        shared_keys = set(self._groups_reps.keys()).intersection(self._groups_encoding.keys())
        if len(shared_keys):
            msg = "Each group column should have only one representation"
            raise ValueError(msg)
        self._check_is_valid_encoder_id_dict(self._groups_encoding)

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
        encoders_dict: Mapping[str, TargetCovariatesEncoderCls] = get_covariates_encoders_from_dict(
            self._groups_encoding,
            covs_df_dict,
            fn_dict=self._groups_encoding_transform_fn,
            inverse_fn_dict=self._groups_encoding_inverse_transform_fn,
        )
        return CategoricalData.from_pandas(
            covs_df_dict,
            repr_dict=repr_dict,
            categorical_encoders=encoders_dict,
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
    def groups_encoding(self) -> dict[str, TargetCovariatesEncodingId]:
        """Exposes to `groups_encoding` parameter set at initialization."""
        return self._groups_encoding

    @property
    def groups_encoding_transform_fn(self) -> dict[str, Callable]:
        """Exposes to `groups_encoding_transform_fn` parameter set at initialization."""
        return self._groups_encoding_transform_fn

    @property
    def groups_encoding_inverse_transform_fn(self) -> dict[str, Callable]:
        """Exposes to `groups_encoding_inverse_transform_fn` parameter set at initialization."""
        return self._groups_encoding_inverse_transform_fn
