import abc
from typing import Any

import torch

from sc_flow._constants import DEFAULT_CONDITIONING_LAYER
from sc_flow._types import ConditioningLayersId
from sc_flow._utils import verify_fn_kwargs_dictionary
from sc_flow.backends.torch._types import TConditioningFn
from sc_flow.backends.torch._utils import make_concatenation_possible
from sc_flow.backends.torch.nn._modules import BaseModule, Resnet1d

__all__ = [
    "BaseConditioningLayer",
    "ConcatConditioning",
    "Resnet1dConditioning",
    "make_custom_conditioning_layer",
    "get_conditioning_layer",
]


class BaseConditioningLayer(BaseModule):
    """Base class for conditioning layers."""


    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        *additional_dims,
    ) -> None:
        """Initializes the concatenation based conditioning.
        
        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param additional_dims: The dimensionality of optional extra arguments to be concatenated to the input.
        :type additional_dims: class: `int`
        """

        super().__init__()
        self._latent_state_dim = latent_state_dim
        self._latent_time_dim = latent_time_dim
        self._additional_dims = additional_dims

    @property
    @abc.abstractmethod
    def output_dim(
        self,
    ) -> int:
        """Returns the dimensionality of the conditioned output."""

    @abc.abstractmethod
    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        *additional_inputs,
    ) -> torch.Tensor:
        """Retrieves the encoded concatenation to be fed as input to the velocity field decoder.

        :param encoded_t: The time index at which the velocity field is computed.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The state at which the velocity fiel is computed.
        :type encoded_state: class: `torch.Tensor`

        :param additional_inputs: Optional extra arguments to be concatenated to the input.
            This should have the same length as the :attr: `self._args` attribute.
            Their trailing dimensions should match the corresponding one specified in the :attr: `self._args`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type additional_inputs: class: `tuple[torch.Tensor]`
        """

    def _verify_inputs_shape(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        *additional_inputs,
    ) -> None:
        """Verifies that the inputs match their respective expected shapes."""

        # sanity checks
        if encoded_state.shape[-1] != self._latent_state_dim:
            msg = (
                "The encoded state has the wrong trailing dimension."
                f"Found {encoded_state.shape[-1]}, expected {self._latent_state_dim}"
            )
            raise RuntimeError(msg)
        if encoded_t.shape[-1] != self._latent_time_dim:
            msg = (
                "The encoded time has the wrong trailing dimension."
                f"Found {encoded_t.shape[-1]}, expected {self._latent_time_dim}"
            )
            raise RuntimeError(msg)
        if len(additional_inputs) != len(self._additional_dims):
            msg = (
                "The number of additional arguments does not match what given at initialization."
                f"The conditioning block expects {len(self._additional_dims)} extra arguments, but {len(additional_inputs)} were given."
            )
            raise RuntimeError(msg)
        for idx, input_tensor in enumerate(additional_inputs):
            if input_tensor.shape[-1] != self._additional_dims[idx]:
                msg = (
                    f"Incorrect trailing dimension for additional input at position {idx}."
                    f"Found {input_tensor.shape[-1]}, expected {self._additional_dims[idx]}"
                )
                raise RuntimeError(msg)


class ConcatConditioning(BaseConditioningLayer):
    """Class for concatenation based conditioning layers."""

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        *additional_dims,
    ) -> None:
        """Initializes the concatenation based conditioning.
        
        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param additional_dims: The dimensionality of optional extra arguments to be concatenated to the input.
        :type additional_dims: class: `int`
        """

        super().__init__(
            latent_state_dim,
            latent_time_dim,
            *additional_dims,
        )

        self._identity = self._make_modules()

    def _make_modules(
        self,
    ) -> torch.nn.Module:
        """Initializes the module."""

        return torch.nn.Identity()

    @property
    def output_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""

        out_dim = self._latent_state_dim + self._latent_time_dim
        for dim in self._additional_dims:
            out_dim += dim
        return out_dim

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        *additional_inputs,
    ) -> torch.Tensor:
        """Forward pass on the conditioning layer, done by concatenating the inputs.

        :param encoded_t: The encoded representation for the current time index.
            Its trailing dimension should match what specified in the :attr: `self._latent_time_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The encoded representation for the current states.
            Its trailing dimension should match what specified in the :attr: `self._latent_state_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_state: class: `torch.Tensor`
        
        :param additional_inputs: Optional extra arguments to be concatenated to the input.
            This should have the same length as the :attr: `self._args` attribute.
            Their trailing dimensions should match the corresponding one specified in the :attr: `self._args`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type additional_inputs: class: `tuple[torch.Tensor]`
        """

        # sanity checks
        self._verify_inputs_shape(
            encoded_t,
            encoded_state,
            *additional_inputs,
        )

        concat_input = torch.concatenate(
                (
                encoded_state,
                make_concatenation_possible(encoded_t, encoded_state, -1),
                *(make_concatenation_possible(arg, encoded_state, -1) for arg in additional_inputs),
            ),
            dim=-1
        )
        return self._identity(concat_input)


class Resnet1dConditioning(BaseConditioningLayer):
    """Class for residual network based conditioning layers."""

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        *additional_dims,
        resnet_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the resnet based conditioning.
        
        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param additional_dims: The dimensionality of optional extra arguments to be concatenated to the input.
        :type additional_dims: class: `int`

        :param resnet_kwargs: Additional key-word arguments used to initialize the :class: `Resnet1d` object.
        :type resnet_kwargs: class: `dict[str, Any]`
        """

        super().__init__(
            latent_state_dim,
            latent_time_dim,
            *additional_dims,
        )
        self._resnet_kwargs = {} if resnet_kwargs is None else resnet_kwargs

        self._resnet = self._make_modules()
    
    def _make_modules(
        self,
    ) -> torch.nn.Module:
        """Initializes the module."""

        verify_fn_kwargs_dictionary(Resnet1d.__init__, self._resnet_kwargs)
        return Resnet1d(
            self._latent_state_dim,
            self.embedding_dim,
            **self._resnet_kwargs,
        )

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        *additional_inputs,
    ) -> torch.Tensor:
        """Forward pass on the conditioning layer, done by concatenating the inputs.

        :param encoded_t: The encoded representation for the current time index.
            Its trailing dimension should match what specified in the :attr: `self._latent_time_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The encoded representation for the current states.
            Its trailing dimension should match what specified in the :attr: `self._latent_state_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_state: class: `torch.Tensor`
        
        :param additional_inputs: Optional extra arguments to be concatenated to the input.
            This should have the same length as the :attr: `self._args` attribute.
            Their trailing dimensions should match the corresponding one specified in the :attr: `self._args`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type additional_inputs: class: `tuple[torch.Tensor]`
        """

        # sanity checks
        self._verify_inputs_shape(
            encoded_t,
            encoded_state,
            *additional_inputs,
        )

        conditions = torch.concatenate(
                (
                make_concatenation_possible(encoded_t, encoded_state, -1),
                *(make_concatenation_possible(input_tensor, encoded_state, -1) for input_tensor in additional_inputs),
            ),
            dim=-1
        )
        return self._resnet(encoded_state, conditions)

    @property
    def output_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""

        return self._resnet.output_dim

    @property
    def embedding_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""

        embedding_dim = self._latent_time_dim
        for dim in self._additional_dims:
            embedding_dim += dim
        return embedding_dim


def make_custom_conditioning_layer(
    conditioning_fn: TConditioningFn,
    conditioning_kwargs: dict[str, Any] | None = None,
) -> BaseConditioningLayer:
    """"""
    raise NotImplementedError


def get_conditioning_layer(
    *dims,
    conditioning_id: ConditioningLayersId | None = None,
    conditioning_fn: TConditioningFn | None = None,
    conditioning_kwargs: dict[str, Any] | None = None,
) -> BaseConditioningLayer:
    """"""

    if conditioning_fn is not None:
        conditioning_kwargs = {} if conditioning_kwargs is None else conditioning_kwargs
        return make_custom_conditioning_layer(conditioning_fn, conditioning_kwargs)
    
    conditioning_id = DEFAULT_CONDITIONING_LAYER if conditioning_id is None else conditioning_id
    
    # sanity checks
    if len(dims) < 2:
        msg = (
            "When initializing predefined conditioning layers, by pasing `conditioning_id`"
            "at least two positional arguments should be given as they are needed to specify"
            f"the input dimensionality of the latent time and of the latent states. Found {len(dims)}"
        )
        raise TypeError(msg)
    for idx, arg in enumerate(dims):
        if not isinstance(arg, int):
            msg = f"The additional positional arguments should be integers, found {type(arg)} at position {idx}"
            raise TypeError(msg)

    if conditioning_id == "concat":
        return ConcatConditioning(*dims)

    elif conditioning_id == "resnet1d":
        conditioning_kwargs = {} if conditioning_kwargs is None else conditioning_kwargs
        return Resnet1dConditioning(
            *dims,
            resnet_kwargs=conditioning_kwargs,
        )
    else:
        msg = f"Conditioning layer {conditioning_id} is not available, possible options are `[\"concat\", \"resnet1d\"]`"
        raise ValueError(msg)
