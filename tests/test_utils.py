from collections.abc import Callable

import pytest

from sc_flow._utils import (
    get_fn_args_names_and_types,
    get_fn_kwargs_names_and_types,
    verify_fn_args,
)

correct_template_id = "correct_template"
missing_arg_template_id = "wrong_template_missing_arg"
wrong_type_template_id = "wrong_template_wrong_type"
Tfn1 = Callable[[float, float, float], float]
Tfn2 = Callable[[float, float], float]
Tfn3 = Callable[[float, int, float], float]

fn_with_args_id = "fn_with_args"


def fn_with_args(
    t: float,
    x0: float,
    x1: float,
) -> float:
    return (1 - t) * x1 - t * x0


fn_with_kwargs_id = "fn_with_kwargs"


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

    @pytest.mark.parametrize("fn_dict", [{fn_with_args_id: fn_with_args}, {fn_with_kwargs_id: fn_with_kwargs}])
    def test_get_fn_kwargs_names_and_types(
        self,
        fn_dict: dict[str, Callable],
    ) -> None:
        fn_id, fn = list(fn_dict.items())[0]
        if fn_id == fn_with_args_id:
            expected_kwargs = {}
        elif fn_id == fn_with_kwargs_id:
            expected_kwargs = {"kwarg1": int, "kwarg2": float}
        arg_types_dict = get_fn_kwargs_names_and_types(fn)
        assert arg_types_dict == expected_kwargs

    @pytest.mark.parametrize(
        "Tfn_dict", [{correct_template_id: Tfn1}, {missing_arg_template_id: Tfn2}, {wrong_type_template_id: Tfn3}]
    )
    def test_verify_fn_args(
        self,
        Tfn_dict: dict[str, type[Callable]],
    ) -> None:
        template_name, template_fn = list(Tfn_dict.items())[0]
        if template_name == wrong_type_template_id:
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                verify_fn_args(fn_with_args, template_fn)
                return None
        elif template_name == missing_arg_template_id:
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                verify_fn_args(fn_with_args, template_fn)
                return None
        elif template_name == correct_template_id:
            verify_fn_args(fn_with_args, template_fn)
