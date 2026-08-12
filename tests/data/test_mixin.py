from collections.abc import Callable
from typing import Any

import pytest

from sckitflow.data._mixins import MappedTree

float_data_dict = lambda: {"a": 0.0, "b": 0.0}
string_data_dict = lambda: {"a": "str", "b": "str"}
mixed_data_dict = lambda: {"a": "str", "b": 0.0}


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

    def test_batch_mixin_init(self) -> None:
        pass
