import inspect
from collections.abc import Callable, Collection
from typing import Any, get_args, get_origin

__all__ = [
    "get_fn_args_names_and_types",
    "get_fn_kwargs_names_and_types",
    "verify_fn_args",
    "verify_fn_kwargs_dictionary",
    "verify_fn_signature",
    "check_sequence_query_against_reference",
]


def check_type_against_generic(
    input_type: type,
    target_type: type,
) -> bool:
    """Checks whether the input type matches the target type, generic or union.

    :param input_type: The input type to be verified.
    :type input_type: class: `type`

    :param target_type: The target type, generic or union which to verify against.
    :type target_type: class: `type`
    """
    origin = get_origin(target_type)
    args = get_args(target_type)
    if origin is not None:
        return any(check_type_against_generic(input_type, t) for t in args)
    if origin:
        return input_type is origin
    return input_type is target_type


def get_fn_args_names_and_types(
    fn: Callable,
    omitted_args: Collection[str] | None = None,
) -> dict[str, type[object]]:
    """Retrieves the type annotation for the positional arguments of a function and their respective names.

    :param fn: The function whose positional arguments annotation we want to retrieve.
    :type fn: class: `Callable`

    :param omitted_args: Optional positional arguments to be omitted from the verification.
        This is to be used only when calling this function on a class method, as we do not want to consider
        `self` as a positional argument. Defaults to `None`.
    :type omitted_args: class: `Collection[str] | None`

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
    omitted_args: Collection[str] | None = None,
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
    :type omitted_args: class: `Collection[str] | None`
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
        if not check_type_against_generic(arg_type, input_types[idx]):
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
    omitted_args: Collection[str] | None = None,
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
    :type omitted_args: class: `Collection[str] | None`
    """
    if not isinstance(fn, Callable):
        msg = f"Input `fn` is expected to be a `Callable`, found {type(fn)}"
        raise TypeError(msg)
    verify_fn_args(fn, Tfn, omitted_args=omitted_args)
    verify_fn_kwargs_dictionary(fn, kwargs_dict)


def check_sequence_query_against_reference(
    query: Collection[str],
    reference: Collection[str],
    allow_missing_from_query: bool = True,
    allow_missing_from_reference: bool = False,
    *,
    query_name: str = "query",
    reference_name: str = "reference",
) -> None:
    """Checks a collection of names against a reference collection.

    :param query: The names to check.
    :type query: class: `Collection[str]`

    :param reference: The names to check against.
    :type reference: class: `Collection[str]`

    :param allow_missing_from_query: Whether the query may omit reference names. Defaults to `True`.
    :type allow_missing_from_query: class: `bool`

    :param allow_missing_from_reference: Whether the query may contain names absent from the reference
        (i.e. whether it need not be a subset). Defaults to `False`.
    :type allow_missing_from_reference: class: `bool`

    :param query_name: What the query is called in the error message -- pass the caller's own parameter
        name (e.g. `"always_train_keys"`) so the message points at the argument to fix. Defaults to
        `"query"`.
    :type query_name: class: `str`

    :param reference_name: What the reference is called in the error message. Defaults to `"reference"`.
    :type reference_name: class: `str`

    :raises ValueError: If either side is missing names the corresponding flag does not allow.
    """
    # set logic for overlap
    union = set(query).union(set(reference))
    missing_from_reference = union - set(reference)
    missing_from_query = union - set(query)

    # strictly checking that we have all reference columns
    if len(missing_from_query) and not allow_missing_from_query:
        # sorted, so the message is stable across runs (set iteration order is not)
        msg = f"{reference_name} entries missing from {query_name}: {sorted(missing_from_query)}."
        raise ValueError(msg)

    # query value not found in reference set
    if len(missing_from_reference) and not allow_missing_from_reference:
        msg = (
            f"{query_name} entries not found in {reference_name}: {sorted(missing_from_reference)}. "
            f"{query_name} must be a subset of {reference_name} ({sorted(reference)})."
        )
        raise ValueError(msg)
