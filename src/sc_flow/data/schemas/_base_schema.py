import abc
from typing import Literal, get_args

import numpy as np
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncodingId
from sc_flow.data.containers._base import BaseData

__all__ = ["DataSchema", "StrictDataSchema"]


class DataSchema(abc.ABC):  # noqa: B024  # base for StrictDataSchema; _get_data is concrete post-binded
    """Abstract base class for enforcing data configurations on :class:`AnnData` objects.

    The derived classes will need to override the following abstract methods to be instantiated:
        * `_get_data`, to extract the data from the input :class: `sc.AnnData`.
    It would also be preferable to define also an immutable data structure for the returned object.

    :param _IS_STRICT: Whether the schema has to be enforced in a strict way. Defaults to `False`
    :type _IS_STRICT: class: `bool`
    """

    _IS_STRICT: bool = False

    @staticmethod
    def _check_key_found_in_adata_field(
        adata: AnnData,
        identifier: str,
        adata_field_key: Literal["obs", "uns", "obsm"],
    ) -> None:
        """Checks that a given key identifier is present in a selected attribute of the input annotated data.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`

        :param identifier: The string identifier for the key to be searched in :param: `adata`.
        :type identifier: class: `str`

        :param adata_field: The attribute of :param: `adata` checked.
        :type adata_field: class: `Literal["obs", "uns", "obsm"]`
        """
        adata_field = getattr(adata, adata_field_key)

        if identifier not in adata_field:
            available = list(adata_field.keys())
            raise KeyError(f"Key '{identifier}' not found in adata.{adata_field_key}. Available keys: {available}")

    def _extract_array(self, adata: AnnData, repr: str | None = None) -> np.ndarray:
        """Extracts the data array from the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        if self._IS_STRICT and repr is not None:
            self._check_key_found_in_adata_field(adata, repr, "obsm")
        return adata.X if repr is None else adata.obsm[repr]

    def _get_data(
        self,
        adata: AnnData,
    ) -> BaseData | tuple[BaseData | None, ...] | None:
        """Enforce the schema on the input :class:`AnnData`.

        No longer the extraction path: after the ``binded`` migration cell arrays are streamed by
        :class:`binded.Loader` and per-leaf conditions are built by
        :func:`sc_flow.data.compile_obs` from the schemas' declared columns/reps. Schemas are now
        pure declarations; a subclass only overrides this when it still produces a label container
        (e.g. :class:`GroupsDataSchema`). The default raises to flag any stale array-materializing use.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        raise NotImplementedError(
            f"{type(self).__name__} is a declaration schema — array extraction moved to "
            "sc_flow.data.compile_obs (binded streams cells). Read its declared properties instead."
        )

    def extract_array(self, adata: AnnData, repr: str | None = None) -> np.ndarray:
        """Extracts the data array from the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`

        :param repr: The representation to extract. When provided, it should appear as
            key in `.obsm`, otherwise the `.X` attribute will be used. Defaults to `None`.
        :type repr: `str | None`
        """
        return self._extract_array(adata, repr=repr)

    def get_data(
        self,
        adata: AnnData,
    ) -> BaseData | tuple[BaseData, ...] | None:
        """Verifies and enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        return self._get_data(adata)


class StrictDataSchema(DataSchema, abc.ABC):
    """Abstract base class for enforcing and verifying data configurations on :class:`AnnData` objects.

    The derived classes will need to override the following abstract methods to be instantiated:
        * `_verify_args`, to verify the validity of the configurations provided at initialization.
        * `_verify_schema`, to verify the schema defined by the class on the input :class: `sc.AnnData`.
        * `_get_data`, to extract the data from the input :class: `sc.AnnData`.
    It would also be preferable to define also an immutable data structure for the returned object.

    :param _IS_STRICT: Whether the schema has to be enforced in a strict way. Defaults to `True`
    :type _IS_STRICT: class: `bool`
    """

    _IS_STRICT: bool = True

    def __init__(self) -> None:
        """Initializes the class and verifies the arguments."""
        self._verify_args()

    @staticmethod
    def _check_is_valid_encoder_id_dict(encoder_id_dict: dict[str, str]) -> None:
        """Verifies that the provided dictionary of covariate encoder identifiers is provided.

        :param encoder_id_dict: The input dictionary, mapping each covariate to the string
            identifier for its encoder.
        :type encoder_id_dict: class: `encoder_id_dict: dict[str, str]`
        """
        valid_encoder_ids: tuple[str] = get_args(TargetCovariatesEncodingId)
        for encoder_id in encoder_id_dict.values():
            if encoder_id not in valid_encoder_ids:
                msg = (
                    f"Encoder identifier {encoder_id} for target covariate encoding is not supported."
                    f"Possible options are {valid_encoder_ids}"
                )
                raise ValueError(msg)

    @abc.abstractmethod
    def _verify_args(
        self,
    ) -> None:
        """Verifies the validity of arguments set at initialization."""
        raise NotImplementedError

    @abc.abstractmethod
    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the schema on the input :class: `AnnData`.

        Needs to be overriden by derived classes.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        raise NotImplementedError

    def get_data(
        self,
        adata: AnnData,
    ) -> BaseData | tuple[BaseData, ...] | None:
        """Verifies and enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        self._verify_schema(adata)
        return super().get_data(adata)
