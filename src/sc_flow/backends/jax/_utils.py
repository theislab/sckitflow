import warnings
from collections.abc import Sequence

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

from sc_flow.backends.jax._types import ArrayLike, JaxArray, JaxDevice, NumpyArray, TDevice

__all__ = [
    "broadcast_to_target_shape",
    "ensure_2d_tensor_with_singleton_trailing_dim",
    "make_concatenation_possible",
    "get_jax_device",
    "get_ode_solver",
    "get_sde_solver",
    "to_jax_array",
]

_ODE_SOLVER_REGISTRY = {
    "euler": dfx.Euler(),
    "heun": dfx.Heun(),
    "midpoint": dfx.Midpoint(),
    "ralston": dfx.Ralston(),
    "bosh3": dfx.Bosh3(),
    "tsit5": dfx.Tsit5(),
    "dopri5": dfx.Dopri5(),
    "dopri8": dfx.Dopri8(),
    "implicit_euler": dfx.ImplicitEuler(),
    "kvaerno3": dfx.Kvaerno3(),
    "kvaerno4": dfx.Kvaerno4(),
    "kvaerno5": dfx.Kvaerno5(),
}

_SDE_SOLVER_REGISTRY = {
    "euler": dfx.Euler(),
    "heun": dfx.Heun(),
    "midpoint": dfx.Midpoint(),
    "ralston": dfx.Ralston(),
    "euler_heun": dfx.EulerHeun(),
    "ito_milstein": dfx.ItoMilstein(),
    "stratonovich_milstein": dfx.StratonovichMilstein(),
    "sea": dfx.SEA(),
    "sra1": dfx.SRA1(),
    "shark": dfx.ShARK(),
    "general_shark": dfx.GeneralShARK(),
    "slow_rk": dfx.SlowRK(),
    "spark": dfx.SPaRK(),
}


def to_jax_array(x: ArrayLike | NumpyArray) -> ArrayLike:
    """Convert a NumPy array to a JAX array if needed."""
    if isinstance(x, np.ndarray):
        return jnp.array(x)
    if x is not None and not isinstance(x, JaxArray):
        msg = f"Invalid type found {type(x)}"
        raise TypeError(msg)
    return x


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

    dims_to_match = list(target_array.shape[:concat_dims])
    dims_to_retain = list(input_array.shape[concat_dims:])
    for idx in range(len(dims_to_match)):
        if idx + 1 > input_array.ndim - len(dims_to_retain):
            input_array = jnp.expand_dims(input_array, idx)
    return broadcast_to_target_shape(input_array, dims_to_match + dims_to_retain)


def ensure_2d_tensor_with_singleton_trailing_dim(
    input_array: ArrayLike,
):
    """"""  # noqa
    if len(input_array.shape) == 0:
        return jnp.expand_dims(input_array, axis=0)
    return broadcast_to_target_shape(input_array, (input_array.shape[0], 1))


def get_jax_device(dev: TDevice) -> JaxDevice:
    """Validate the JAX device passed as input and return the corresponding jax.Device object.

    If the requested device is not found, falls back to CPU with a warning.
    """
    if isinstance(dev, str):
        candidates = [d for d in jax.devices() if d.platform == dev]
        if not candidates:
            warnings.warn(
                f"No available device found for platform '{dev}'. Falling back to CPU.", UserWarning, stacklevel=2
            )
            cpu_devices = [d for d in jax.devices() if d.platform == "cpu"]
            if not cpu_devices:
                raise RuntimeError("No CPU device available as fallback")
            return cpu_devices[0]
        return candidates[0]
    else:
        return dev


def get_ode_solver(method: str | dfx.AbstractSolver | None) -> dfx.AbstractSolver:
    """Retrieve the diffrax ODE solver corresponding to the given method name.

    :param method: The name of the ODE solver method or a diffrax solver instance.
    :type method: str | dfx.AbstractSolver | None

    :returns: The corresponding diffrax ODE solver instance.
    :rtype: dfx.AbstractSolver
    """
    if method is None:
        return dfx.Euler()
    elif isinstance(method, dfx.AbstractSolver):
        return method

    method_key = method.lower()
    if method_key not in _ODE_SOLVER_REGISTRY:
        raise ValueError(f"ODE solver '{method}' not found.")

    return _ODE_SOLVER_REGISTRY[method_key]


def get_sde_solver(method: str | dfx.AbstractSolver | None) -> dfx.AbstractSolver:
    """Retrieve the diffrax SDE solver corresponding to the given method name.

    :param method: The name of the SDE solver method or a diffrax solver instance.
    :type method: str | dfx.AbstractSolver | None

    :returns: The corresponding diffrax SDE solver instance.
    :rtype: dfx.AbstractSolver
    """
    if method is None:
        return dfx.Euler()
    elif isinstance(method, dfx.AbstractSolver):
        return method

    method_key = method.lower()
    if method_key not in _SDE_SOLVER_REGISTRY:
        raise ValueError(f"SDE solver '{method}' not found.")

    return _SDE_SOLVER_REGISTRY[method_key]
