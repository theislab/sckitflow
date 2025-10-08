import torch

from sc_flow.backends.torch._types import ShapeLike

__all__ = [
    "broadcast_to_target_shape",
    "ensure_2d_tensor_with_singleton_trailing_dim",
    "make_concatenation_possible",
]


def broadcast_to_target_shape(
    input_tensor: torch.Tensor,
    target_shape: ShapeLike,
) -> torch.Tensor:
    """Broadcasts the input tensor to the target shape.

    This is done according to the following reshaping policy:
        * When the :param: `input_tensor` is of the correct shape, simply returns it.
        * When the input tensor has more dimensions than the target shape, a :class: `ValueError` is raised.
        * When the input tensor has at most the same number of dimensions as the target shape, they should be either
            matching the respective dimension in the target shape or be singleton, in which case they will be
            expanded to match the corresponding target dimension. In case of mismatch a :class: `ValueError` is raised.

    :param input_tensor: The input tensor whose to broadcast.
    :type input_tensor: class: `torch.Tensor`

    :param target_shape: The target shape which we want to broadcast the input to.
    :type target_shape: class: `ShapeLike`
    """
    # tensor already in the correct shape, do nothing
    if input_tensor.shape == target_shape:
        return input_tensor

    # retrieving number of dimensions
    num_target_dims = len(target_shape)

    # sanity check (there should be at most the same dimensions in input_tensor)
    if input_tensor.ndim > num_target_dims:
        msg = f"The input tensor has more dimensions that the target shape. (i.e.: {input_tensor.shape=}, {target_shape=})"
        raise ValueError(msg)

    # shapes should match across the shared dimensions
    dims_to_expand = []
    for dim_idx, target_dim in enumerate(target_shape):
        # check that they match on the shared dimensions or that input array has only singleton dimension
        if dim_idx < input_tensor.ndim:
            current_dim = input_tensor.shape[dim_idx]
            if target_dim != current_dim:
                if current_dim != 1:
                    msg = f"Mismatch in shape. (i.e.: {input_tensor.shape=}, {target_shape=})"
                    raise ValueError(msg)
                dims_to_expand.append(target_dim)
            else:
                dims_to_expand.append(-1)
        else:
            input_tensor = input_tensor.unsqueeze(-1)
            dims_to_expand.append(target_dim)
    return input_tensor.expand(*dims_to_expand)


def ensure_2d_tensor_with_singleton_trailing_dim(
    input_tensor: torch.Tensor,
):
    """"""  # noqa

    if len(input_tensor.shape) == 0:
        input_tensor = input_tensor.unsqueeze(0)
    return broadcast_to_target_shape(input_tensor, (input_tensor.shape[0], 1))


def make_concatenation_possible(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    concat_dims: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO."""  # noqa

    dims_to_match = list(target_tensor.shape[:concat_dims])
    dims_to_retain = list(input_tensor.shape[concat_dims:])
    for idx in range(len(dims_to_match)):
        if idx + 1 > input_tensor.ndim - len(dims_to_retain):
            input_tensor = input_tensor.unsqueeze(idx)
    return broadcast_to_target_shape(input_tensor, dims_to_match + dims_to_retain)
