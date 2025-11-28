from collections.abc import Mapping
from typing import Any

import pytest

from sc_flow.data_new._mixins import DataMixin


class TestMixins:
    @pytest.mark.parametrize(
        "required_type",
        [None, float, "str"],
        "mapping",
        [
            {
                "a": 0.0,
                "b": 0.0,
            },
            {
                "a": "str",
                "b": "str",
            },
            {
                "a": 0.0,
                "b": "str",
            },
        ],
    )
    def test_data_mixin_init(
        self,
        required_type: type[Any] | None,
        mapping: Mapping[str, Any],
    ) -> None:
        # when we dont specify the required type the only possible
        # reason of failure is the mismatch between the elements' types
        if required_type is None and type(mapping["a"]) is not type(mapping["b"]):
            with pytest.raises(TypeError, r"Data is of the wrong type\."):
                mixin = DataMixin(required_type, mapping)
        # when required type is passed need to respect it
        if required_type is not None and (
            not isinstance(mapping["a"], required_type) or not isinstance(mapping["b"], required_type)
        ):
            mixin = DataMixin(required_type, mapping)
        # initializing data mixin
        mixin = DataMixin(required_type, mapping)
        assert mixin.data_type in [str, float]

    def test_data_mixin_apply(self) -> None:
        raise NotImplementedError

    def test_batch_mixin_init(self) -> None:
        raise NotImplementedError

    def test_batch_mixin_apply(self) -> None:
        raise NotImplementedError
