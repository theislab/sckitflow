import abc
from typing import Literal, get_args

import numpy as np
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncodingId
from sc_flow.data._structures import BaseData

__all__ = ["BaseDataSchema"]


class BaseDataSchema(abc.ABC):
    """Abstract base class for enforcing and verifying data configurations on :class:`AnnData` objects.

    The derived classes will need to override the following abstract methods to be instantiated:
        * `_verify_args`, to verify the validity of the configurations provided at initialization.
        * `_verify_schema`, to verify the schema defined by the class on the input :class: `sc.AnnData`.
        * `_get_data`, to extract the data from the input :class: `sc.AnnData`.
    It would also be preferable to define also an immutable data structure for the returned object.
    """

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

    @staticmethod
    def _extract_array(adata: AnnData, repr: str | None = None) -> np.ndarray:
        """Extracts the data array from the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        return adata.X if repr is None else adata.obsm[repr]

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

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_data(
        self,
        adata: AnnData,
    ) -> BaseData:
        """Enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        raise NotImplementedError

    def get_data(
        self,
        adata: AnnData,
    ) -> BaseData:
        """Verifies and enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        self._verify_schema(adata)
        return self._get_data(adata)
