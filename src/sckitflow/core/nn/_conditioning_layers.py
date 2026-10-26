import abc
from typing import Any

import torch

from sckitflow._constants import DEFAULT_CONDITIONING_LAYER
from sckitflow._types import ConditioningLayersId
from sckitflow._utils import verify_fn_kwargs_dictionary
from sckitflow.core._utils import make_concatenation_possible
from sckitflow.core.nn._modules import Resnet1d

__all__ = [
    "AdaLNZeroConditioning",
    "BaseConditioningLayer",
    "ConcatConditioning",
    "FiLMConditioning",
    "Resnet1dConditioning",
    "get_conditioning_layer",
]


class BaseConditioningLayer(torch.nn.Module):
    """Base class for conditioning layers."""

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        """Initializes the concatenation based conditioning.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) The dimensionality of extra conditioning argument to be concatenated to the input.
        :type latent_condition_dim: class: `int | None`
        """
        super().__init__()
        self._latent_state_dim = latent_state_dim
        self._latent_time_dim = latent_time_dim
        self._latent_condition_dim = latent_condition_dim

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
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Retrieves the encoded concatenation to be fed as input to the velocity field decoder.

        :param encoded_t: The time index at which the velocity field is computed.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The state at which the velocity fiel is computed.
        :type encoded_state: class: `torch.Tensor`

        :param encoded_condition: Optional extra conditioning argument to be concatenated to the input.
            Its trailing dimension should match the corresponding one specified in the :attr: `self._latent_condition_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `torch.Tensor`
        """

    @property
    def embedding_dim(self) -> int:
        """Width of the conditioning signal: the time embedding plus the optional condition."""
        return self._latent_time_dim + (self._latent_condition_dim or 0)

    def _conditions(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        """Checks the inputs, then broadcasts time and condition to the state's leading dims and concatenates them."""
        self._verify_inputs_shape(encoded_t, encoded_state, encoded_condition)
        parts = [encoded_t] if encoded_condition is None else [encoded_t, encoded_condition]
        return torch.cat([make_concatenation_possible(p, encoded_state, -1) for p in parts], dim=-1)

    def _verify_inputs_shape(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
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
        if encoded_condition is not None:
            if encoded_condition.shape[-1] != self._latent_condition_dim:
                msg = (
                    f"Incorrect trailing dimension for additional condition input."
                    f"Found {encoded_condition.shape[-1]}, expected {self._latent_condition_dim}"
                )
                raise RuntimeError(msg)
        else:
            if self._latent_condition_dim is not None:
                msg = (
                    "When the latent condition dimension is specified,"
                    "the corresponding input should be passed, found `None`."
                )
                raise RuntimeError(msg)


class ConcatConditioning(BaseConditioningLayer):
    """Class for concatenation based conditioning layers."""

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        """Initializes the concatenation based conditioning.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) The dimensionality of extra conditioning argument to be concatenated to the input.
        :type latent_condition_dim: class: `int | None`
        """
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )

    @property
    def output_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""
        out_dim = self._latent_state_dim + self._latent_time_dim
        if self._latent_condition_dim is not None:
            out_dim = out_dim + self._latent_condition_dim
        return out_dim

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
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

        :param encoded_condition: Optional extra conditioning argument to be concatenated to the input.
            Its trailing dimension should match the corresponding one specified in the :attr: `self._latent_condition_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `torch.Tensor`
        """
        return torch.cat([encoded_state, self._conditions(encoded_t, encoded_state, encoded_condition)], dim=-1)


class Resnet1dConditioning(BaseConditioningLayer):
    """Class for residual network based conditioning layers."""

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
        resnet_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the resnet based conditioning.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) The dimensionality of extra conditioning argument to be concatenated to the input.
        :type latent_condition_dim: class: `int | None`

        :param resnet_kwargs: Additional key-word arguments used to initialize the :class: `Resnet1d` object.
        :type resnet_kwargs: class: `dict[str, Any]`
        """
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )
        self._resnet_kwargs = {} if resnet_kwargs is None else resnet_kwargs

        verify_fn_kwargs_dictionary(Resnet1d.__init__, self._resnet_kwargs)
        self._resnet = Resnet1d(
            self._latent_state_dim,
            self.embedding_dim,
            **self._resnet_kwargs,
        )

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
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

        :param encoded_condition: Optional extra conditioning argument to be concatenated to the input.
            Its trailing dimension should match the corresponding one specified in the :attr: `self._latent_condition_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `torch.Tensor`
        """
        return self._resnet(encoded_state, self._conditions(encoded_t, encoded_state, encoded_condition))

    @property
    def output_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""
        return self._resnet.output_dim


class FiLMConditioning(BaseConditioningLayer):
    """Feature-wise linear modulation (FiLM, Perez et al. 2018) of the state by time and condition.

    A linear map of the conditioning signal gives a per-feature scale ``gamma`` and shift ``beta``, and the state
    becomes ``(1 + gamma) * encoded_state + beta``: ``gamma`` is an offset from 1, as in the paper's code. The
    projection starts at zero, as DiT's modulation does, so the layer starts as the identity and a random
    projection cannot rescale the state before training. The modulation is bilinear in condition and state,
    which concatenation cannot express. ``output_dim`` is ``latent_state_dim``.
    """

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        """Initializes the FiLM based conditioning.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) The dimensionality of extra conditioning
            argument to be used in the FiLM modulation.
        :type latent_condition_dim: class: `int | None`
        """
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )
        # one projection for both: its output splits into (gamma, beta); zero, so the layer starts as the identity
        self._linear = torch.nn.Linear(self.embedding_dim, 2 * self._latent_state_dim)
        torch.nn.init.zeros_(self._linear.weight)
        torch.nn.init.zeros_(self._linear.bias)

    @property
    def output_dim(self) -> int:
        """Return the dimensionality of the conditioned output."""
        return self._latent_state_dim

    def _modulate(self, conditions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Scales and shifts ``state`` per feature by the ``(gamma, beta)`` projected from ``conditions``."""
        gamma, beta = self._linear(conditions).chunk(2, dim=-1)
        return (1 + gamma) * state + beta

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass applying FiLM modulation to the encoded state.

        :param encoded_t: The encoded representation for the current time index.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The encoded representation for the current states.
        :type encoded_state: class: `torch.Tensor`

        :param encoded_condition: Optional extra conditioning argument used
            to compute the FiLM parameters.
        :type encoded_condition: class: `torch.Tensor`
        """
        return self._modulate(self._conditions(encoded_t, encoded_state, encoded_condition), encoded_state)


class AdaLNZeroConditioning(FiLMConditioning):
    """adaLN-Zero (Peebles & Xie 2023): FiLM of the layer-normed state, with SiLU on the condition.

    The state is normalized without a learned scale or shift of its own and the conditioning signal goes through a
    SiLU before the projection; the projection starts at zero as in FiLM, so the layer starts as ``LayerNorm(state)``.
    DiT also gates a residual branch; this layer feeds the decoder directly, so there is no branch to gate.
    """

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        """Initializes the adaLN-Zero based conditioning.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) The dimensionality of extra conditioning
            argument to be used in the modulation.
        :type latent_condition_dim: class: `int | None`
        """
        super().__init__(latent_state_dim, latent_time_dim, latent_condition_dim)
        self._norm = torch.nn.LayerNorm(self._latent_state_dim, elementwise_affine=False)

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass modulating the layer-normed state.

        :param encoded_t: The encoded representation for the current time index.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The encoded representation for the current states.
        :type encoded_state: class: `torch.Tensor`

        :param encoded_condition: Optional extra conditioning argument used
            to compute the modulation.
        :type encoded_condition: class: `torch.Tensor`
        """
        conditions = torch.nn.functional.silu(self._conditions(encoded_t, encoded_state, encoded_condition))
        return self._modulate(conditions, self._norm(encoded_state))


_CONDITIONING_LAYERS: dict[str, type[BaseConditioningLayer]] = {
    "concat": ConcatConditioning,
    "resnet1d": Resnet1dConditioning,
    "film": FiLMConditioning,
    "adaln_zero": AdaLNZeroConditioning,
}


def get_conditioning_layer(
    latent_state_dim: int,
    latent_time_dim: int,
    *,
    latent_condition_dim: int | None = None,
    conditioning_id: ConditioningLayersId | None = None,
    conditioning_kwargs: dict[str, Any] | None = None,
) -> BaseConditioningLayer:
    """Builds the conditioning layer named ``conditioning_id`` (default ``DEFAULT_CONDITIONING_LAYER``).

    ``conditioning_kwargs`` go to the :class:`Resnet1d` of ``"resnet1d"``; the other layers take none.
    """
    conditioning_id = DEFAULT_CONDITIONING_LAYER if conditioning_id is None else conditioning_id
    if conditioning_id not in _CONDITIONING_LAYERS:
        msg = f"Conditioning layer {conditioning_id!r} is not available, possible options are {list(_CONDITIONING_LAYERS)}"
        raise ValueError(msg)
    kwargs = {"resnet_kwargs": conditioning_kwargs} if conditioning_id == "resnet1d" else {}
    return _CONDITIONING_LAYERS[conditioning_id](
        latent_state_dim, latent_time_dim, latent_condition_dim=latent_condition_dim, **kwargs
    )
