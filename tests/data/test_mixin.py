from collections.abc import Callable, Collection, Mapping
from typing import Any

import pytest

from sc_flow.data._mixins import DataMixin

float_data_dict = lambda: {"a": 0.0, "b": 0.0}
string_data_dict = lambda: {"a": "str", "b": "str"}
mixed_data_dict = lambda: {"a": "str", "b": "str"}


def to_int(x: float):
    return int(x)


def double(x: float):
    return 2 * x


class TestMixins:
    @pytest.mark.parametrize("required_type", [None, float, str])
    @pytest.mark.parametrize("mapping_fn", [float_data_dict, string_data_dict])
    def test_data_mixin_init(
        self,
        required_type: type[Any] | None,
        mapping_fn: Callable[..., Mapping[str, Any]],
    ) -> None:
        # mismatch type
        if (required_type is float and mapping_fn.__name__ == "string_data_dict") or (
            required_type is str and mapping_fn.__name__ == "float_data_dict"
        ):
            with pytest.raises(TypeError, match="Data is of the wrong type"):
                mixin = DataMixin(required_type=required_type, mapping=mapping_fn())
        mixin = DataMixin(required_type=required_type, mapping=mapping_fn())
        if mapping_fn.__name__ == "string_data_dict":
            assert mixin.data_type is str
        else:
            assert mixin.data_type is float

    @pytest.mark.parametrize("required_type", [None, float])
    def test_data_mixin_init_fail_with_mixed_type(
        self,
        required_type: type[Any] | None,
    ) -> None:
        mapping = mixed_data_dict()
        if required_type is None:
            with pytest.raises(TypeError, r"(The values should share the same type\.)"):
                _mixin = DataMixin(required_type, mapping)
        with pytest.raises(TypeError, r"(Data is of the wrong type\.)"):
            _mixin = DataMixin(required_type, mapping)

    @pytest.mark.parametrize("function", [to_int, double])
    @pytest.mark.parametrize("fields", [None, ["a"]])
    @pytest.mark.parametrize("drop_unused_fields", [False, True])
    @pytest.mark.parametrize("recursive", [True])
    @pytest.mark.parametrize("output_type", [None, int])
    def test_data_mixin_apply(
        self,
        function: Callable[[Any], Any],
        fields: None | Collection[str],
        drop_unused_fields: bool,
        recursive: bool,
        output_type: type[Any] | None,
    ) -> None:
        # initialize mixin
        mixin = DataMixin(float, float_data_dict())

        # fail cases
        if (function.__name__ == "double" and output_type is int) or (
            function.__name__ == "to_int" and output_type is None
        ):
            with pytest.raises(TypeError, match=r"Data is of the wrong type"):
                mixin = mixin.apply(
                    function,
                    fields,
                    drop_unused_fields=drop_unused_fields,
                    recursive=recursive,
                    output_type=output_type,
                )
        mixin = mixin.apply(
            function, fields, drop_unused_fields=drop_unused_fields, recursive=recursive, output_type=output_type
        )

        # get expected keys
        if drop_unused_fields and fields == ["a"]:
            expected_keys = ["a"]
        elif drop_unused_fields and fields is None:
            expected_keys = ["a", "b"]
        assert list(mixin.mapping) == expected_keys

    def test_batch_mixin_init(self) -> None:
        pass

    def test_batch_mixin_apply(self) -> None:
        pass
