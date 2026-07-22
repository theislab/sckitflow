import abc
from typing import Literal

import numpy as np
from anndata import AnnData

from sc_flow.data.containers._base import BaseData

__all__ = ["DataSchema", "StrictDataSchema"]


class DataSchema(abc.ABC):  # noqa: B024  # base for StrictDataSchema; _get_data is concrete post-binded

    _IS_STRICT: bool = False

    @staticmethod
    def _check_key_found_in_adata_field(
        adata: AnnData,
        identifier: str,
        adata_field_key: Literal["obs", "uns", "obsm"],
    ) -> None:
        adata_field = getattr(adata, adata_field_key)

        if identifier not in adata_field:
            available = list(adata_field.keys())
            raise KeyError(f"Key '{identifier}' not found in adata.{adata_field_key}. Available keys: {available}")

    def _extract_array(self, adata: AnnData, repr: str | None = None) -> np.ndarray:
        if self._IS_STRICT and repr is not None:
            self._check_key_found_in_adata_field(adata, repr, "obsm")
        return adata.X if repr is None else adata.obsm[repr]

    def _get_data(
        self,
        adata: AnnData,
    ) -> BaseData | tuple[BaseData | None, ...] | None:
        raise NotImplementedError(
            f"{type(self).__name__} is a declaration schema — array extraction moved to "
            "sc_flow.data.compile_obs (binded streams cells). Read its declared properties instead."
        )

    def extract_array(self, adata: AnnData, repr: str | None = None) -> np.ndarray:
        return self._extract_array(adata, repr=repr)

    def get_data(
        self,
        adata: AnnData,
    ) -> BaseData | tuple[BaseData, ...] | None:
        return self._get_data(adata)


class StrictDataSchema(DataSchema, abc.ABC):

    _IS_STRICT: bool = True

    def __init__(self) -> None:
        self._verify_args()

    @abc.abstractmethod
    def _verify_args(
        self,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        raise NotImplementedError

    def get_data(
        self,
        adata: AnnData,
    ) -> BaseData | tuple[BaseData, ...] | None:
        self._verify_schema(adata)
        return super().get_data(adata)
