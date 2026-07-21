from collections.abc import Callable
from typing import Any

import pytest

from sc_flow.core.data._mixins import MappedTree

float_data_dict = lambda: {"a": 0.0, "b": 0.0}
string_data_dict = lambda: {"a": "str", "b": "str"}
mixed_data_dict = lambda: {"a": "str", "b": 0.0}


def to_int(x: float):
    return int(x)


def double(x: float):
    return 2 * x


class TestMixins:
    @pytest.mark.parametrize("required_type", [float, str])
    def test_data_mixin_init(
        self,
        required_type: type[Any] | None,
    ) -> None:
        # mismatch type
        CustomMixin = type(
            "CustomMixin",
            (MappedTree,),
            {"_REQUIRED_VALUE_TYPE": required_type},
        )
        if required_type is float:
            data_dict = float_data_dict()
        else:
            data_dict = string_data_dict()
        mixin = CustomMixin(mapping=data_dict)
        for value in mixin.mapping.values():
            assert isinstance(value, required_type | MappedTree)

    @pytest.mark.parametrize("required_type", [float, str])
    def test_data_mixin_init_fails(
        self,
        required_type: type[Any] | None,
    ):
        CustomMixin = type(
            "CustomMixin",
            (MappedTree,),
            {"_REQUIRED_VALUE_TYPE": required_type},
        )
        if required_type is str:
            data_dict = float_data_dict()
        else:
            data_dict = string_data_dict()
        with pytest.raises(TypeError, match="The values should respect the pre-defined type"):
            _mixin = CustomMixin(mapping=data_dict)
        return None

    @pytest.mark.parametrize("required_type", [str, float])
    def test_data_mixin_init_fail_with_mixed_type(
        self,
        required_type: type[Any],
    ) -> None:
        CustomMixin = type(
            "CustomMixin",
            (MappedTree,),
            {"_REQUIRED_VALUE_TYPE": required_type},
        )
        mapping = mixed_data_dict()
        with pytest.raises(TypeError, match=r"The values should respect the pre-defined type\."):
            _mixin = CustomMixin(mapping)
        return None

    @pytest.mark.parametrize("function", [to_int, double])
    @pytest.mark.parametrize("output_value_type", [None, int])
    def test_data_mixin_apply(
        self,
        function: Callable[[Any], Any],
        output_value_type: type[Any] | None,
    ) -> None:
        # initialize mixin
        MappedTree._REQUIRED_VALUE_TYPE = float
        mixin_orig = MappedTree(float_data_dict())

        # fail cases
        if (function.__name__ == "double" and output_value_type is int) or (
            function.__name__ == "to_int" and output_value_type is None
        ):
            with pytest.raises(TypeError, match=r"The values should respect the pre-defined type"):
                mixin_mapped = mixin_orig.apply(
                    function,
                    output_value_type=output_value_type,
                )
            return None
        mixin_mapped = mixin_orig.apply(function, output_value_type=output_value_type)

        assert set(mixin_orig.mapping.keys()) == set(mixin_mapped.mapping.keys())
        for key, val in mixin_orig.mapping.items():
            val_mapped = mixin_mapped[key]
            assert val_mapped == 2 * val

    def test_batch_mixin_init(self) -> None:
        pass

    def test_batch_mixin_apply(self) -> None:
        pass
