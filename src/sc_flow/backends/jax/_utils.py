from collections.abc import Sequence

import jax.numpy as jnp

from sc_flow.backends.jax._types import ArrayLike


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
    :type input_array: class: `torch.Tensor`

    :param target_shape: The target shape which we want to broadcast the input to.
    :type target_shape: class: `ShapeLike`
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
    return jnp.tile(input_array, jnp.array(dims_to_expand))
