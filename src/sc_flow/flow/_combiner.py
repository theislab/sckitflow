import abc
from dataclasses import dataclass
from typing import Any

import torch

from scfit._component import ComponentRegistry, ComponentSpec
from scfit.nn import NET_REGISTRY, NetContext, NetSpec, Resnet1d
from sc_flow.flow._torch_utils import make_concatenation_possible

__all__ = [
    "COMBINER_REGISTRY",
    "BaseCombiner",
    "CombinerContext",
    "CombinerSpec",
    "ConcatCombiner",
    "Resnet1dCombiner",
    "build_combiner",
    "validate_combiner_spec",
]

#: A combiner spec is a :class:`~scfit.ComponentSpec` — the slot (``combiner``) fixes the family.
CombinerSpec = ComponentSpec


@dataclass(frozen=True)
class CombinerContext:

    latent_state_dim: int
    latent_time_dim: int
    latent_condition_dim: int | None = None


class BaseCombiner(abc.ABC, torch.nn.Module):

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        super().__init__()
        self._latent_state_dim = latent_state_dim
        self._latent_time_dim = latent_time_dim
        self._latent_condition_dim = latent_condition_dim

    @property
    @abc.abstractmethod
    def output_dim(
        self,
    ) -> int:
        ...

    @abc.abstractmethod
    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...

    def _verify_inputs_shape(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> None:
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


class ConcatCombiner(BaseCombiner):

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )

    @property
    def output_dim(
        self,
    ) -> int:
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
        return torch.concatenate(to_concat, dim=-1)


class Resnet1dCombiner(BaseCombiner):

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
        *,
        resnet: Resnet1d,
    ) -> None:
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )
        self._resnet = resnet

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # sanity checks
        self._verify_inputs_shape(
            encoded_t,
            encoded_state,
            encoded_condition,
        )

        # build the [time (+ condition)] conditioning vector the resnet is conditioned on
        to_concat = (make_concatenation_possible(encoded_t, encoded_state, -1),)
        if encoded_condition is not None:
            to_concat = to_concat + (make_concatenation_possible(encoded_condition, encoded_state, -1),)

        conditions = torch.concatenate(to_concat, dim=-1)
        return self._resnet(encoded_state, conditions)

    @property
    def output_dim(
        self,
    ) -> int:
        return self._resnet.output_dim

    @property
    def embedding_dim(
        self,
    ) -> int:
        embedding_dim = self._latent_time_dim
        if self._latent_condition_dim is not None:
            embedding_dim = embedding_dim + self._latent_condition_dim
        return embedding_dim


#: Closed built-in combiner family on the shared registry machinery. ``concat`` is parameter-free; the
#: resnet combiner carries weights and a different output dim, which is why a combiner is a discriminated
#: spec (like pooling) rather than a flat enum. A custom :class:`BaseCombiner` instance is the runtime-only
#: research escape hatch (see :class:`~sc_flow.flow._vf.MLPVelocity`), not a registered type.
COMBINER_REGISTRY: ComponentRegistry[CombinerContext] = ComponentRegistry("combiner")


@COMBINER_REGISTRY.register("sc_flow.concat")
@dataclass(frozen=True)
class ConcatCombinerConfig:
    def build(self, context: CombinerContext) -> BaseCombiner:
        return ConcatCombiner(
            context.latent_state_dim,
            context.latent_time_dim,
            latent_condition_dim=context.latent_condition_dim,
        )


@COMBINER_REGISTRY.register("sc_flow.resnet1d")
@dataclass
class Resnet1dCombinerConfig:
    resnet: NetSpec

    def __post_init__(self) -> None:
        try:
            self.resnet = NET_REGISTRY.validate(self.resnet)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid sc_flow.resnet1d config slot 'resnet': {e}") from e
        if self.resnet["type"] != "scfit.resnet":
            raise ValueError(
                "sc_flow.resnet1d config slot 'resnet' requires a 'scfit.resnet' net spec, "
                f"found {self.resnet['type']!r}."
            )

    def build(self, context: CombinerContext) -> BaseCombiner:
        condition_dim = context.latent_time_dim
        if context.latent_condition_dim is not None:
            condition_dim += context.latent_condition_dim
        resnet = NET_REGISTRY.build(
            self.resnet,
            NetContext(
                input_dim=context.latent_state_dim,
                output_dim=context.latent_state_dim,
                condition_dim=condition_dim,
            ),
        )
        if not isinstance(resnet, Resnet1d):  # registry/slot invariant; fail loudly if registration changes
            raise TypeError(
                "sc_flow.resnet1d config slot 'resnet' must build scfit.nn.Resnet1d, "
                f"found {type(resnet).__name__}."
            )
        return Resnet1dCombiner(
            context.latent_state_dim,
            context.latent_time_dim,
            latent_condition_dim=context.latent_condition_dim,
            resnet=resnet,
        )


def validate_combiner_spec(spec: CombinerSpec | dict[str, Any]) -> CombinerSpec:
    return COMBINER_REGISTRY.validate(spec)


def build_combiner(
    spec: CombinerSpec,
    *,
    latent_state_dim: int,
    latent_time_dim: int,
    latent_condition_dim: int | None = None,
) -> BaseCombiner:
    context = CombinerContext(
        latent_state_dim=latent_state_dim,
        latent_time_dim=latent_time_dim,
        latent_condition_dim=latent_condition_dim,
    )
    return COMBINER_REGISTRY.build(spec, context)
