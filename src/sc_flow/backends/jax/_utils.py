from collections.abc import Sequence

import jax.numpy as jnp

from sc_flow.backends.jax._types import ArrayLike

__all__ = [
    "broadcast_to_target_shape",
    "ensure_2d_tensor_with_singleton_trailing_dim",
    "make_concatenation_possible",
]


def broadcast_to_target_shape(
    input_array: ArrayLike,
    target_shape: Sequence[int],
) -> ArrayLike:
    """Broadcasts the input tensor to the target shape.

    This is done according to the following reshaping policy:
        * When the :param: `input_array` is of the correct shape, simply returns it.
        * When the input tensor has more dimensions than the target shape, a :class: `ValueError` is raised.
        * When the input tensor has at most the same number of dimensions as the target shape, they should be either
            matching the respective dimension in the target shape or be singleton, in which case they will be
            expanded to match the corresponding target dimension. In case of mismatch a :class: `ValueError` is raised.

    :param input_array: The input tensor whose to broadcast.
    :type input_array: :class:`~sc_flow.backends.jax._types.ArrayLike`

    :param target_shape: The target shape which we want to broadcast the input to.
    :type target_shape: Sequence[int]
    """
    # tensor already in the correct shape, do nothing
    if input_array.shape == target_shape:
        return input_array

    # retrieving number of dimensions
    num_target_dims = len(target_shape)

    # sanity check (there should be at most the same dimensions in input_array)
    if input_array.ndim > num_target_dims:
        msg = (
            f"The input tensor has more dimensions that the target shape. (i.e.: {input_array.shape=}, {target_shape=})"
        )
        raise ValueError(msg)

    # shapes should match across the shared dimensions
    dims_to_expand = []
    for dim_idx, target_dim in enumerate(target_shape):
        # check that they match on the shared dimensions or that input array has only singleton dimension
        if dim_idx < input_array.ndim:
            current_dim = input_array.shape[dim_idx]
            if target_dim != current_dim:
                if current_dim != 1:
                    msg = f"Mismatch in shape. (i.e.: {input_array.shape=}, {target_shape=})"
                    raise ValueError(msg)
                dims_to_expand.append(target_dim)
            else:
                dims_to_expand.append(1)
        else:
            input_array = jnp.expand_dims(input_array, -1)
            dims_to_expand.append(target_dim)
    return jnp.broadcast_to(input_array, target_shape)


def make_concatenation_possible(
    input_array: ArrayLike,
    target_array: ArrayLike,
    concat_dims: int = -1,
) -> tuple[ArrayLike, ArrayLike]:
    """"""  # noqa

    dims_to_match = [d for d in target_array.shape[:concat_dims]]
    dims_to_retain = [d for d in input_array.shape[concat_dims:]]
    for idx in range(len(dims_to_match)):
        if idx + 1 > input_array.ndim - len(dims_to_retain):
            input_array = jnp.expand_dims(input_array, idx)
    return broadcast_to_target_shape(input_array, dims_to_match + dims_to_retain)


def ensure_2d_tensor_with_singleton_trailing_dim(
    input_array: ArrayLike,
):
    """"""  # noqa

    if len(input_array.shape) == 0:
        input_array = jnp.expand_dims(input_array, axis=0)
    return broadcast_to_target_shape(input_array, (input_array.shape[0], 1))
