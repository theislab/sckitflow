from collections.abc import Callable

import pytest

from sckitflow._utils import (
    get_fn_args_names_and_types,
    get_fn_kwargs_names_and_types,
    verify_fn_args,
)

Tfn1 = {"template": Callable[[float, float, float], float], "is_valid": True}
Tfn2 = {"template": Callable[[float, float], float], "is_valid": False}
Tfn3 = {"template": Callable[[float, int, float], float], "is_valid": False}


def fn_with_args(
    t: float,
    x0: float,
    x1: float,
) -> float:
    return (1 - t) * x1 - t * x0


def fn_with_kwargs(
    t: float,
    x0: float,
    x1: float,
    kwarg1: int = 0,
    kwarg2: float = 0.0,
) -> float:
    return (1 - t) * x1 - t * x0


class TestUtils:
    @pytest.mark.parametrize("fn", [fn_with_args, fn_with_kwargs])
    def test_get_fn_args_names_and_types(
        self,
        fn: dict[str, dict[str, type]],
    ) -> None:
        arg_types_dict = get_fn_args_names_and_types(fn)
        assert arg_types_dict == {"t": float, "x0": float, "x1": float}

    @pytest.mark.parametrize("fn", [fn_with_args, fn_with_kwargs])
    def test_get_fn_kwargs_names_and_types(
        self,
        fn: dict[str, Callable],
    ) -> None:
        if fn.__name__ == "fn_with_args":
            expected_kwargs = {}
        elif fn.__name__ == "fn_with_kwargs":
            expected_kwargs = {"kwarg1": int, "kwarg2": float}
        arg_types_dict = get_fn_kwargs_names_and_types(fn)
        assert arg_types_dict == expected_kwargs

    @pytest.mark.parametrize("Tfn_dict", [Tfn1, Tfn2, Tfn3])
    def test_verify_fn_args(
        self,
        Tfn_dict: dict[int, type[Callable]] | int,
    ) -> None:
        if not Tfn_dict["is_valid"]:
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                verify_fn_args(fn_with_args, Tfn_dict["template"])
            return None
        else:
            verify_fn_args(fn_with_args, Tfn_dict["template"])
