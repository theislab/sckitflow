from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, ClassVar

import numpy as np

__all__ = ["MappedTree", "MappedArray", "BatchMixin"]


@dataclass(frozen=True)
class MappedTree[KeyT: Hashable, ValT]:
    """Recursively mapped tree keyed by ``KeyT`` with leaf values of type ``ValT``.

    Each mapping value is either a leaf ``ValT`` or a nested ``MappedTree[KeyT, ValT]``.
    """

    _REQUIRED_KEY_TYPE: ClassVar[type[Any]] = str
    _REQUIRED_VALUE_TYPE: ClassVar[type[Any]] = object
    mapping: Mapping[Hashable, ValT | MappedTree[KeyT, ValT]] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        """"""  # noqa
        # verifying inputs
        self._verify_inputs()

    def __getitem__(self, key: Hashable) -> ValT | MappedTree[KeyT, ValT]:
        """"""  # noqa
        return self.mapping[key]

    def _verify_inputs(self) -> None:
        """"""  # noqa
        # iterating over each key to check that the type is the same
        for key, value in self.mapping.items():
            if not isinstance(value, self._REQUIRED_VALUE_TYPE | MappedTree):
                msg = f"The values should respect the pre-defined type. Got {type(value)} for {key}, expected {self._REQUIRED_VALUE_TYPE}."
                raise TypeError(msg)
            if not isinstance(key, self._REQUIRED_KEY_TYPE):
                msg = f"The keys should respect the pre-defined type. Got {type(value)} for {key}, expected {self._REQUIRED_VALUE_TYPE}."
                raise TypeError(msg)


@dataclass(frozen=True)
class MappedArray(MappedTree):
    """"""  # noqa

    _REQUIRED_VALUE_TYPE: ClassVar[type[Any]] = np.ndarray | np.generic


@dataclass(frozen=True)
class BatchMixin[KeyT: Hashable, ValT](MappedTree[KeyT, ValT]):
    """"""  # noqa

    _MIN_DIMS: ClassVar[int] = 1

    def __len__(self) -> int:
        """"""  # noqa
        if len(self.reference_dims) != 1:
            msg = f"Continuous covariates should have only one reference dim, found {len(self.reference_dims)}"
            raise ValueError(msg)
        return self.reference_dims[0]

    def _verify_inputs(self) -> None:
        """"""  # noqa
        # calling method of parent class for usual checks
        super()._verify_inputs()

        # iterating over the elements
        for key, value in self.mapping.items():
            # verifying that the required dimensions match
            if isinstance(value, BatchMixin):
                value._verify_inputs()
            else:
                self._verify_shape(key, value)

    def _verify_shape(
        self,
        key: KeyT,
        data: ValT,
    ) -> None:
        """"""  # noqa
        # we need at least self._minimum_dims + 1 dimensions
        if data.ndim <= self._MIN_DIMS:
            msg = (
                f"Data for {key} has insufficient number of dimension."
                f"Expected at least {self._MIN_DIMS} dimensions, but founf {data.ndim}"
            )
            raise ValueError(msg)

        # iterating over the number of required dimensions
        for dim, reference_dim in enumerate(self.reference_dims):
            # raise error if does not match or if dimension
            # is not singleton
            data_dim = data.shape[dim]
            if data_dim != reference_dim and data_dim != 1:
                msg = (
                    f"Shape mismatch for {key}."
                    f"Reference dimension at index {dim}: {reference_dim}"
                    f"Found data of shape{data.shape}"
                )
                raise ValueError(msg)

    @property
    def reference_dims(
        self,
    ) -> Collection[int]:
        """"""  # noqa
        if len(self.mapping):
            reference_array = next(iter(self.mapping.values()))
            return reference_array.shape[: self._MIN_DIMS]
        return []
