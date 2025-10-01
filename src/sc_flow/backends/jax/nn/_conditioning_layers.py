import abc
from typing import Any

import jax.numpy as jnp

from sc_flow._constants import DEFAULT_CONDITIONING_LAYER
from sc_flow._types import ConditioningLayersId
from sc_flow._utils import verify_fn_kwargs_dictionary
from sc_flow.backends.jax._types import ArrayLike, TConditioningFn
from sc_flow.backends.jax._utils import make_concatenation_possible
from sc_flow.backends.jax.nn._modules import BaseModule, Resnet1d

__all__ = [
    "BaseConditioningLayer",
    "ConcatConditioning",
    "Resnet1dConditioning",
    "make_custom_conditioning_layer",
    "get_conditioning_layer",
]


class BaseConditioningLayer(BaseModule):
    """Base class for conditioning layers.
    
    :param latent_state_dim: The latent dimensionality of the input states.
    :type latent_state_dim: class: `int`

    :param latent_time_dim: The latent dimensionality of the input time index.
    :type latent_time_dim: class: `int`

    :param latent_condition_dim: (Optional) The dimensionality of extra conditioning argument to be concatenated to the input.
    :type latent_condition_dim: class: `int | None`
    """

    latent_state_dim: int
    latent_time_dim: int
    latent_condition_dim: int | None = None

    @property
    @abc.abstractmethod
    def output_dim(
        self,
    ) -> int:
        """Returns the dimensionality of the conditioned output."""

    @abc.abstractmethod
    def __call__(
        self,
        encoded_t: ArrayLike,
        encoded_state: ArrayLike,
        encoded_condition: ArrayLike | None = None,
    ) -> ArrayLike:
        """Retrieves the encoded concatenation to be fed as input to the velocity field decoder.

        :param encoded_t: The time index at which the velocity field is computed.
        :type encoded_t: class: `sc_flow.backends.jax._types.ArrayLike`

        :param encoded_state: The state at which the velocity fiel is computed.
        :type encoded_state: class: `sc_flow.backends.jax._types.ArrayLike`s

        :param encoded_condition: Optional extra conditioning argument to be concatenated to the input.
            Its trailing dimension should match the corresponding one specified in the :attr: `self._latent_condition_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `sc_flow.backends.jax._types.ArrayLike` | `None`
        """

    def _verify_inputs_shape(
        self,
        encoded_t: ArrayLike,
        encoded_state: ArrayLike,
        encoded_condition: ArrayLike | None = None,
    ) -> None:
        """Verifies that the inputs match their respective expected shapes."""
        # sanity checks
        if encoded_state.shape[-1] != self.latent_state_dim:
            msg = (
                "The encoded state has the wrong trailing dimension."
                f"Found {encoded_state.shape[-1]}, expected {self.latent_state_dim}"
            )
            raise RuntimeError(msg)
        if encoded_t.shape[-1] != self.latent_time_dim:
            msg = (
                "The encoded time has the wrong trailing dimension."
                f"Found {encoded_t.shape[-1]}, expected {self.latent_time_dim}"
            )
            raise RuntimeError(msg)
        if encoded_condition is not None:
            if encoded_condition.shape[-1] != self.latent_condition_dim:
                msg = (
                    f"Incorrect trailing dimension for additional condition input."
                    f"Found {encoded_condition.shape[-1]}, expected {self.latent_condition_dim}"
                )
                raise RuntimeError(msg)
        else:
            if self.latent_condition_dim is not None:
                msg = (
                    "When the latent condition dimension is specified,"
                    "the corresponding input should be passed, found `None`."
                )
                raise RuntimeError(msg)


class ConcatConditioning(BaseConditioningLayer):
    """Class for concatenation based conditioning layers.
    
    :param latent_state_dim: The latent dimensionality of the input states.
    :type latent_state_dim: class: `int`

    :param latent_time_dim: The latent dimensionality of the input time index.
    :type latent_time_dim: class: `int`

    :param latent_condition_dim: (Optional) The dimensionality of extra conditioning argument to be concatenated to the input.
    :type latent_condition_dim: class: `int | None`
    """

    latent_state_dim: int
    latent_time_dim: int
    latent_condition_dim: int | None = None

    @property
    def output_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""
        out_dim = self.latent_state_dim + self.latent_time_dim
        if self.latent_condition_dim is not None:
            out_dim = out_dim + self.latent_condition_dim
        return out_dim

    def __call__(
        self,
        encoded_t: ArrayLike,
        encoded_state: ArrayLike,
        encoded_condition: ArrayLike | None = None,
    ) -> ArrayLike:
        """Forward pass on the conditioning layer, done by concatenating the inputs.

        :param encoded_t: The encoded representation for the current time index.
            Its trailing dimension should match what specified in the :attr: `self._latent_time_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_t: class: `sc_flow.backends.jax._types.ArrayLike`

        :param encoded_state: The encoded representation for the current states.
            Its trailing dimension should match what specified in the :attr: `self._latent_state_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_state: class: `sc_flow.backends.jax._types.ArrayLike`

        :param encoded_condition: Optional extra conditioning argument to be concatenated to the input.
            Its trailing dimension should match the corresponding one specified in the :attr: `self._latent_condition_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `sc_flow.backends.jax._types.ArrayLike`
        """
        # sanity checks
        self._verify_inputs_shape(
            encoded_t,
            encoded_state,
            encoded_condition,
        )

        # concatenating input
        to_concat = (encoded_state, make_concatenation_possible(encoded_t, encoded_state, -1))
        if encoded_condition is not None:
            to_concat = to_concat + (make_concatenation_possible(encoded_condition, encoded_state, -1),)
        return jnp.concatenate(to_concat, axis=-1)

class Resnet1dConditioning(BaseConditioningLayer):
    """Class for residual network based conditioning layers.

    :param latent_state_dim: The latent dimensionality of the input states.
    :type latent_state_dim: class: `int`

    :param latent_time_dim: The latent dimensionality of the input time index.
    :type latent_time_dim: class: `int`

    :param latent_condition_dim: (Optional) The dimensionality of extra conditioning argument to be concatenated to the input.
    :type latent_condition_dim: class: `int | None`

    :param resnet_kwargs: Additional key-word arguments used to initialize the :class: `Resnet1d` object.
    :type resnet_kwargs: class: `dict[str, Any]`
    """

    latent_state_dim: int
    latent_time_dim: int
    latent_condition_dim: int | None = None
    resnet_kwargs: dict[str, Any] | None = None

    def setup(self):
        """Initializes the resnet based conditioning."""

        resnet_kwargs = {} if self.resnet_kwargs is None else self.resnet_kwargs
    
        verify_fn_kwargs_dictionary(Resnet1d.__init__, resnet_kwargs)

        self.resnet = Resnet1d(
            self.latent_state_dim,
            self.embedding_dim,
            **resnet_kwargs,
        )

    def __call__(
        self,
        encoded_t: ArrayLike,
        encoded_state: ArrayLike,
        encoded_condition: ArrayLike | None = None,
    ) -> ArrayLike:
        """Forward pass on the conditioning layer, done by concatenating the inputs.

        :param encoded_t: The encoded representation for the current time index.
            Its trailing dimension should match what specified in the :attr: `self._latent_time_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_t: class: `sc_flow.backends.jax._types.ArrayLike`

        :param encoded_state: The encoded representation for the current states.
            Its trailing dimension should match what specified in the :attr: `self._latent_state_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_state: class: `sc_flow.backends.jax._types.ArrayLike`

        :param encoded_condition: Optional extra conditioning argument to be concatenated to the input.
            Its trailing dimension should match the corresponding one specified in the :attr: `self._latent_condition_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `sc_flow.backends.jax._types.ArrayLike`
        """
        # sanity checks
        self._verify_inputs_shape(
            encoded_t,
            encoded_state,
            encoded_condition,
        )

        # concatenating input
        to_concat = (make_concatenation_possible(encoded_t, encoded_state, -1),)
        if encoded_condition is not None:
            to_concat = to_concat + (make_concatenation_possible(encoded_condition, encoded_state, -1),)

        conditions = jnp.concatenate(to_concat, axis=-1)
        return self.resnet(encoded_state, conditions)

    @property
    def output_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""
        return self.resnet.output_dim

    @property
    def embedding_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""
        embedding_dim = self.latent_time_dim
        if self.latent_condition_dim is not None:
            embedding_dim = embedding_dim + self.latent_condition_dim
        return embedding_dim

def make_custom_conditioning_layer(
    conditioning_fn: TConditioningFn,
    conditioning_kwargs: dict[str, Any] | None = None,
) -> BaseConditioningLayer:
    """"""  # noqa
    raise NotImplementedError


def get_conditioning_layer(
    latent_state_dim: int,
    latent_time_dim: int,
    latent_condition_dim: int | None = None,
    conditioning_id: ConditioningLayersId | None = None,
    conditioning_fn: TConditioningFn | None = None,
    conditioning_kwargs: dict[str, Any] | None = None,
) -> BaseConditioningLayer:
    """"""  # noqa

    if conditioning_fn is not None:
        conditioning_kwargs = {} if conditioning_kwargs is None else conditioning_kwargs
        return make_custom_conditioning_layer(conditioning_fn, conditioning_kwargs)

    conditioning_id = DEFAULT_CONDITIONING_LAYER if conditioning_id is None else conditioning_id

    if conditioning_id == "concat":
        return ConcatConditioning(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim=latent_condition_dim,
        )

    elif conditioning_id == "resnet1d":
        conditioning_kwargs = {} if conditioning_kwargs is None else conditioning_kwargs
        return Resnet1dConditioning(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim=latent_condition_dim,
            resnet_kwargs=conditioning_kwargs,
        )
    else:
        msg = f'Conditioning layer {conditioning_id} is not available, possible options are `["concat", "resnet1d"]`'
        raise ValueError(msg)
