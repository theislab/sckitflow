from collections.abc import Callable, Collection

from anndata import AnnData

from sc_flow._types import TargetCovariatesEncodingId
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["ResponseDataSchema"]


class ResponseDataSchema(StrictDataSchema):
    """Data Schema implementing the logic for response data."""

    def __init__(
        self,
        categorical_covs_dict: dict[str, TargetCovariatesEncodingId] | None = None,
        continuous_covs: Collection[str] | None = None,
        encoding_transform_fn: dict[str, Callable] | None = None,
        encoding_inverse_transform_fn: dict[str, Callable] | None = None,
    ) -> None:
        """Initializes the data schema.

        :param categorical_covs_dict: Dictionary mapping column identifiers in :param: `groups`
            to the corresponding key of `.uns` used to store the covariate representations.
            The corresponding value in `.uns` need to have keys matching the unique values
            of the corresponding column in `.obs`. Defaults to `None`.
        :type categorical_covs_dict: class: `dict[str, TargetCovariatesEncodingId] | None`

        :param continuous_covs: Sequence of continuous condition covariates. They should appear
            as key in `.obsm`. Defaults to `None`.
        :type continuous_covs: class: `Collection[str] | None`

        :param encoding_transform_fn: Dictionary mapping column identifiers in :param: `categorical_covs_dict`
            to the corresponding function used to define functional tranformations.
            This is only used if the corresponding value is `"fuctional"`.
            Defaults to `None`, in which case it will be set to the identity function.
        :type encoding_transform_fn: class: `dict[str, Callable] | None`

        :param encoding_inverse_transform_fn: Dictionary mapping column identifiers in :param: `categorical_covs_dict`
            to the corresponding function used to define inverse functional tranformations.
            This is only used if the corresponding value is `"fuctional"`.
            Defaults to `None`, in which case it will be set to the identity function.
        :type encoding_inverse_transform_fn: class: `dict[str, Callable] | None`
        """
        self._categorical_covs_dict = {} if categorical_covs_dict is None else categorical_covs_dict
        self._continuous_covs = [] if continuous_covs is None else continuous_covs
        self._encoding_transform_fn = {} if encoding_transform_fn is None else encoding_transform_fn
        self._encoding_inverse_transform_fn = (
            {} if encoding_inverse_transform_fn is None else encoding_inverse_transform_fn
        )
        super().__init__()

    def _verify_args(self) -> None:
        """Verifies the validity of the encoder identifiers."""
        self._check_is_valid_encoder_id_dict(self._categorical_covs_dict)

    def _verify_schema_categorical_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the categorical target covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        for target_covariate in self._categorical_covs_dict.keys():
            self._check_key_found_in_adata_field(adata, target_covariate, "obs")

    def _verify_schema_continuous_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the continuous target covariates settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        for target_covariate in self._continuous_covs:
            self._check_key_found_in_adata_field(adata, target_covariate, "obsm")

    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the schema for target and continuous covariates on the input data.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        self._verify_schema_categorical_covariates(adata)
        self._verify_schema_continuous_covariates(adata)

    @property
    def categorical_covariates(self) -> tuple[str]:
        """Returns the keys of :attr: `categorical_covs_dict`."""
        return tuple(self.categorical_covs_dict.keys())

    @property
    def categorical_covs_dict(self) -> dict[str, TargetCovariatesEncodingId]:
        """Exposes to `categorical_covs_dict` parameter set at initialization."""
        return self._categorical_covs_dict

    @property
    def continuous_covs(self) -> Collection[str]:
        """Exposes to `continuous_covs` parameter set at initialization."""
        return self._continuous_covs

    @property
    def has_categorical_covariates(self) -> bool:
        """Whether the response schema includes categorical covariates.

        This will be `True` whenever :param: `categorical_covs_dict` is set at initializtion
        """
        return len(self._categorical_covs_dict) > 0

    @property
    def has_continuous_covariates(self) -> bool:
        """Whether the response schema includes categorical covariates.

        This will be `True` whenever :param: `continuous_covs` is set at initializtion
        """
        return len(self._continuous_covs) > 0

    @property
    def categorical_reps_map(self) -> dict[str, str]:
        """Dictionary mapping each categorical column to the corresponding realm for their representation."""
        return {target_covariate: target_covariate for target_covariate in self._categorical_covs_dict.keys()}
