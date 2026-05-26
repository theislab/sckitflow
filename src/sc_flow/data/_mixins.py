from collections.abc import Callable, Collection, Hashable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, ClassVar

import numpy as np

from sc_flow.data._abc import DataT, DataTree, DataTreeT

__all__ = ["MappedTree", "MappedLevelIndex", "MappedArray", "BatchMixin"]


@dataclass(frozen=True)
class MappedTree(DataTree):
    """"""  # noqa

    _REQUIRED_KEY_TYPE: ClassVar[type[Any]] = str
    _REQUIRED_VALUE_TYPE: ClassVar[type[Any]] = object
    mapping: Mapping[Hashable, DataT | DataTreeT] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        """"""  # noqa
        # verifying inputs
        self._verify_inputs()

    def __getitem__(self, key: Hashable) -> DataT | DataTreeT:
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

    def _apply_to_level(
        self,
        level_value: DataT | DataTreeT,
        function: Callable[[Any], Any],
        *args,
        output_key_type: type[Any] | None = None,
        output_value_type: type[Any] | None = None,
        **kwargs,
    ) -> DataT:
        """"""  # noqa
        if isinstance(level_value, MappedTree):
            return self._apply(
                level_value,
                function,
                *args,
                output_key_type=output_key_type,
                output_value_type=output_value_type,
            )
        return function(level_value, *args, **kwargs)

    def _apply(
        self,
        mapping: Mapping[Hashable, DataT | DataTreeT],
        function: Callable[[Any], Any],
        *args,
        output_key_type: type[Any] | None = None,
        output_value_type: type[Any] | None = None,
        **kwargs,
    ) -> DataTreeT:
        """"""  # noqa
        out_dict = {}
        if isinstance(mapping, MappedTree):
            mapping = mapping.mapping
        for key, value in mapping.items():
            out_dict[key] = self._apply_to_level(
                value, function, *args, output_key_type=output_key_type, output_value_type=output_value_type, **kwargs
            )
        output_key_type = self._REQUIRED_KEY_TYPE if output_key_type is None else output_key_type
        output_value_type = self._REQUIRED_VALUE_TYPE if output_value_type is None else output_value_type
        return type(
            self.__class__.__name__,
            (self.__class__,),
            {"_REQUIRED_KEY_TYPE": output_key_type, "_REQUIRED_VALUE_TYPE": output_value_type},
        )(out_dict)

    def apply(
        self,
        function: Callable[[Any], Any],
        *args,
        output_key_type: type[Any] | None = None,
        output_value_type: type[Any] | None = None,
        **kwargs,
    ) -> DataTreeT:
        """"""  # noqa
        return self._apply(
            self.mapping,
            function,
            *args,
            output_key_type=output_key_type,
            output_value_type=output_value_type,
            **kwargs,
        )

    @property
    def is_leaf(self) -> bool:
        return all(isinstance(v, self._REQUIRED_VALUE_TYPE) for v in self.mapping.values())

    def flatten(self) -> tuple[DataT]:
        """Flattens itself into a list of nodes."""
        if not self.is_leaf:
            return tuple([v for val in self.mapping.values() for v in val.flatten()])
        return tuple(self.mapping.values())


@dataclass(frozen=True)
class MappedLevelIndex(MappedTree):
    _REQUIRED_KEY_TYPE: ClassVar[type[Any]] = tuple
    _REQUIRED_VALUE_TYPE: ClassVar[type[Any]] = slice


@dataclass(frozen=True)
class MappedArray(MappedTree):
    """"""  # noqa

    _REQUIRED_VALUE_TYPE: ClassVar[type[Any]] = np.ndarray | np.generic


@dataclass(frozen=True)
class BatchMixin(MappedArray):
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
        key: str,
        data: np.ndarray,
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
