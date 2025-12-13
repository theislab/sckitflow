from collections.abc import Callable, Collection, Hashable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

__all__ = ["DataMixin", "ArrayMixin", "BatchMixin"]


T = TypeVar("T")


@dataclass(frozen=True)
class DataMixin(Generic[T]):
    """"""  # noqa

    required_key_type: ClassVar[type[Any]] = str
    required_value_type: ClassVar[type[Any]] = object
    mapping: Mapping[Hashable, "T | DataMixin"] = dc_field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """"""  # noqa
        # verifying inputs
        self._verify_inputs()

    def __getitem__(self, key: Hashable) -> "T | DataMixin[T]":
        """"""  # noqa
        return self.mapping[key]

    def _verify_inputs(self) -> None:
        """"""  # noqa
        # iterating over each key to check that the type is the same
        for key, value in self.mapping.items():
            if not isinstance(value, self.required_value_type):
                msg = f"The values should respect the pre-defined type. Got {type(value)} for {key}, expected {self.data_type}."
                raise TypeError(msg)
            if not isinstance(key, self.required_key_type):
                msg = f"The keys should respect the pre-defined type. Got {type(value)} for {key}, expected {self.data_type}."
                raise TypeError(msg)

    def _apply_to_level(
        self,
        level_value: "T | DataMixin",
        function: Callable[[Any], Any],
        *args,
        output_key_type: type[Any] | None = None,
        output_value_type: type[Any] | None = None,
        **kwargs,
    ) -> "T":
        """"""  # noqa
        if isinstance(level_value, DataMixin):
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
        mapping: Mapping[Hashable, "T | DataMixin"],
        function: Callable[[Any], Any],
        *args,
        output_key_type: type[Any] | None = None,
        output_value_type: type[Any] | None = None,
        **kwargs,
    ) -> "DataMixin":
        """"""  # noqa
        out_dict = {}
        if isinstance(mapping, DataMixin):
            mapping = mapping.mapping
        for key, value in mapping.items():
            out_dict[key] = self._apply_to_level(
                value, function, *args, output_key_type=output_key_type, output_value_type=output_value_type, **kwargs
            )
        output_key_type = self.required_key_type if output_key_type is None else output_key_type
        output_value_type = self.required_value_type if output_value_type is None else output_value_type
        self.__class__.required_key_type = output_key_type
        self.__class__.required_value_type = output_value_type
        return self.__class__(out_dict)

    def apply(
        self,
        function: Callable[[Any], Any],
        *args,
        output_key_type: type[Any] | None = None,
        output_value_type: type[Any] | None = None,
        **kwargs,
    ) -> "DataMixin":
        """"""  # noqa
        return self._apply(
            self.mapping,
            function,
            *args,
            output_key_type=output_key_type,
            output_value_type=output_value_type,
            **kwargs,
        )


@dataclass(frozen=True)
class ArrayMixin(DataMixin):
    """"""  # noqa

    required_type: ClassVar[type[Any]] = np.ndarray | np.generic


@dataclass(frozen=True)
class BatchMixin(ArrayMixin):
    """"""  # noqa

    minimum_dims: ClassVar[int] = 1

    def _verify_inputs(self) -> None:
        """"""  # noqa
        # calling method of parent class for usual checks
        super()._verify_inputs()

        # iterating over the elements
        for key, value in self.mapping.items():
            # verifying that the required dimensions match
            self._verify_shape(key, value)

    def _verify_shape(
        self,
        key: str,
        data: np.ndarray,
    ) -> None:
        """"""  # noqa
        # we need at least self._minimum_dims + 1 dimensions
        if data.ndim <= self.minimum_dims:
            msg = (
                f"Data for {key} has insufficient number of dimension."
                f"Expected at least {self.minimum_dims} dimensions, but founf {data.ndim}"
            )
            raise ValueError(msg)

        # iterating over the number of required dimensions
        for dim, reference_dim in enumerate(self.reference_dims):
            # raise error if does not match
            if data.shape[dim] != reference_dim:
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
            return reference_array.shape[: self.minimum_dims]
        return []
