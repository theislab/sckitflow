import abc
from dataclasses import dataclass
from typing import Literal

from anndata import AnnData

from sc_flow.data._structures import BaseData

__all__ = ["BaseDataSchema"]


@dataclass
class BaseDataSchema(abc.ABC):
    """"""  # noqa

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

    @abc.abstractmethod
    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        raise NotImplementedError

    @abc.abstractmethod
    def _get_data(
        self,
        adata: AnnData,
    ) -> BaseData:
        """"""  # noqa
        raise NotImplementedError

    def get_data(
        self,
        adata: AnnData,
    ) -> BaseData:
        """"""  # noqa
        self._verify_schema(adata)
        return self._get_data(adata)
