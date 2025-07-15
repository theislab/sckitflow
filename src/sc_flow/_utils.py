import inspect
from collections.abc import Callable, Sequence
from typing import Any, get_args

__all__ = [
    "get_fn_args_names_and_types",
    "get_fn_kwargs_names_and_types",
    "verify_fn_args",
    "verify_fn_kwargs_dictionary",
    "verify_fn_signature",
]


def get_fn_args_names_and_types(
    fn: Callable,
    omitted_args: Sequence[str] | None = None,
) -> dict[str, type[object]]:
    """Retrieves the type annotation for the positional arguments of a function and their respective names.

    :param fn: The function whose positional arguments annotation we want to retrieve.
    :type fn: class: `Callable`

    :param omitted_args: Optional positional arguments to be omitted from the verification.
        This is to be used only when calling this function on a class method, as we do not want to consider
        `self` as a positional argument. Defaults to `None`.
    :type omitted_args: class: `Sequence[str] | None`

    :return: Dictionary mapping each positional argument name to the respective type annotation.
    :rtype: class: `dict[str, type[object]]`
    """
    omitted_args = [] if omitted_args is None else omitted_args
    signature = inspect.signature(fn)
    return {
        name: param.annotation
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and param.default is inspect.Parameter.empty
        and name not in omitted_args
    }


def get_fn_kwargs_names_and_types(
    fn: Callable,
) -> dict[str, type[object]]:
    """Retrieves the type annotation for the key-word arguments of a function and their respective names.

    :param fn: The function whose key-word arguments annotation we want to retrieve.
    :type fn: class: `Callable`

    :return: Dictionary mapping each key-word argument name to the respective type annotation.
    :rtype: class: `dict[str, type[object]]`
    """
    signature = inspect.signature(fn)
    return {
        name: param.annotation
        for name, param in signature.parameters.items()
        if (
            param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and param.default is not inspect.Parameter.empty
        )
    }


def verify_fn_args(
    fn: Callable,
    Tfn: type[Callable],
    omitted_args: Sequence[str] | None = None,
) -> None:
    """Verifies the positional arguments of a function against a template.

    Checks that the annotations of the input function :param: `fn` match the template specified in :param: `Tfn`.
    Raises :class: `TypeError` when the input function has a different number of positional
    arguments than the template or when the type annotations for the inputs do not match.

    :param fn: The input function whose arguments we want to verify against the template.
    :type fn: class: `Callable`

    :param Tfn: The reference template function against which to verify the inputs.
    :type Tfn: class: `type[Callable]`

    :param omitted_args: Optional positional arguments to be omitted from the verification.
        This is to be used only when calling this function on a class method, as we do not want to consider
        `self` as a positional argument. Defaults to `None`.
    :type omitted_args: class: `Sequence[str] | None`
    """
    input_types, output_types = get_args(Tfn)
    positional_args = get_fn_args_names_and_types(fn, omitted_args=omitted_args)
    if len(positional_args) != len(input_types):
        msg = (
            f"The input function {fn} and the reference template {Tfn} have different number of"
            f"positional arguments. Found {len(positional_args)}, expected {len(input_types)}"
        )
        raise TypeError(msg)
    for idx, (arg_name, arg_type) in enumerate(positional_args.items()):
        if arg_type != input_types[idx]:
            msg = (
                f"The input function {fn} and the reference template {Tfn} have a different annotation for"
                f"argument {arg_name}. Found {arg_type}, expected {input_types[idx]}"
            )
            raise TypeError(msg)


def verify_fn_kwargs_dictionary(
    fn: Callable,
    kwargs_dict: dict[str, Any],
) -> None:
    """Verifies that the input key-word arguments dictionary matches the signature of the function.

    :param fn: The function whose key-word arguments annotation we want to verify the input dictionary against.
    :type fn: class: `Callable`

    :param kwargs_dict: The input dictionary containing the optional key-word arguments.
    :type kwargs_dict: class: `dict[str, Any]`
    """
    kwargs_names_and_types = get_fn_kwargs_names_and_types(fn)
    for name, param in kwargs_dict.items():
        if name not in kwargs_names_and_types.keys():
            msg = f"Argument name {name} not found in key-word arguments of function {fn}."
            raise TypeError(msg)
        if not isinstance(param, kwargs_names_and_types[name]):
            msg = (
                f"Argument name {name} of function {fn} is of the wrong type."
                f"Expected {kwargs_names_and_types[name]}, found {type(param)}."
            )
            raise TypeError(msg)


def verify_fn_signature(
    fn: Callable,
    Tfn: type[Callable],
    kwargs_dict: dict[str, Any],
    omitted_args: Sequence[str] | None = None,
) -> None:
    """Verifies the signature of the input function against a template and optional key-word arguments.

    Verifies that the function positional argument match the template and that the key-word dictionary contains the
    correct argument name and type annotations.

    :param fn: The function whose key-word arguments annotation we want to verify the input dictionary against.
    :type fn: class: `Callable`

    :param Tfn: The reference template function against which to verify the inputs.
    :type Tfn: class: `type[Callable]`

    :param kwargs_dict: The input dictionary containing the optional key-word arguments.
    :type kwargs_dict: class: `dict[str, Any]`

    :param omitted_args: Optional positional arguments to be omitted from the verification.
        This is to be used only when calling this function on a class method, as we do not want to consider
        `self` as a positional argument. Defaults to `None`.
    :type omitted_args: class: `Sequence[str] | None`
    """
    if not isinstance(fn, Callable):
        msg = f"Input `fn` is expected to be a `Callable`, found {type(fn)}"
        raise TypeError(msg)
    verify_fn_args(fn, Tfn, omitted_args=omitted_args)
    verify_fn_kwargs_dictionary(fn, kwargs_dict)
