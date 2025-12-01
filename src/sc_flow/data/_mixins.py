from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from sc_flow._utils import apply_fn_to_mapping

__all__ = ["DataMixin", "ArrayMixin", "BatchMixin"]


@dataclass(frozen=True)
class DataMixin:
    """"""  # noqa

    required_type: ClassVar[type[Any] | None] = object
    mapping: Mapping[str, Any] | None = None

    def __init__(self, required_type: type[Any] | None = None, mapping: Mapping[str, Any] | None = None) -> None:
        """"""  # noqa
        # update data type if provided
        if required_type is not None:
            self.__class__.required_type = required_type

        # initialize dict parent with same items
        super().__init__(mapping)

        # run verification
        self.__post_init__()

    def __post_init__(self) -> None:
        """"""  # noqa
        # creating empty dictionary
        if self.mapping is None:
            self.mapping = {}

        # verifying that the keys are strings
        for key in self.mapping.keys():
            if not isinstance(key, str):
                msg = ""
                raise ValueError(msg)

        # verifying inputs
        self._verify_inputs()

    def _verify_inputs(self) -> None:
        """"""  # noqa
        # checking that we have the required type when specified
        if self.required_type is not None:
            if self.data_type is not self.required_type:
                msg = f"Data is of the wrong type. Got {self.data_type}, expected {self.required_type}."
                raise TypeError(msg)

        # iterating over each key to check that the type is the same
        for key, value in self.mapping.items():
            if not isinstance(value, self.data_type):
                msg = f"The values should share the same type. Got {type(value)} for {key}, expected {self.data_type}."
                raise TypeError(msg)

    def apply(
        self,
        function: Callable[[Any], Any],
        *args,
        fields: None | Collection[str] = None,
        drop_unused_fields: bool = False,
        output_type: type[Any] | None = None,
        **kwargs,
    ):
        """"""  # noqa
        mapping = apply_fn_to_mapping(
            self.mapping,
            function,
            *args,
            fields=fields,
            drop_unused_fields=drop_unused_fields,
            **kwargs,
        )
        output_type = self.data_type if output_type is None else output_type
        self.__class__.required_type = output_type
        return self.__class__(
            mapping=mapping,
        )

    @property
    def data_type(self) -> type[Any]:
        """"""  # noqa
        if len(self.mapping) == 0:
            return self.required_type
        return type(next(iter(self.values())))


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
