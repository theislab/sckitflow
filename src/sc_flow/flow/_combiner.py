import abc
from dataclasses import dataclass
from typing import Any

import torch

from sc_flow._component import ComponentRegistry, ComponentSpec
from sc_flow.nn import NET_REGISTRY, AdaLNZero1d, NetContext, NetSpec, Resnet1d
from sc_flow.flow._torch_utils import make_concatenation_possible

__all__ = [
    "COMBINER_REGISTRY",
    "BaseCombiner",
    "CombinerContext",
    "CombinerSpec",
    "ConcatCombiner",
    "AdaLNCombiner",
    "Resnet1dCombiner",
    "build_combiner",
    "validate_combiner_spec",
]

#: A combiner spec is a :class:`~sc_flow.ComponentSpec` — the slot (``combiner``) fixes the family.
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


class _InputLayerNorms(torch.nn.Module):
    """Optional per-stream LayerNorm applied to (state, time, condition) before they are combined.

    Off by default (identity, zero params) so it never perturbs the legacy concat path. When on it
    matches CellFlow2's ``layer_norm_before_concatenation`` — a LayerNorm on each stream so, e.g., time
    doesn't dominate the condition in the combined signal (matters most for adaLN's modulation vector)."""

    def __init__(self, latent_state_dim, latent_time_dim, latent_condition_dim, *, enabled):
        super().__init__()
        self._enabled = bool(enabled)
        if self._enabled:
            self.state = torch.nn.LayerNorm(latent_state_dim)
            self.time = torch.nn.LayerNorm(latent_time_dim)
            self.condition = None if latent_condition_dim is None else torch.nn.LayerNorm(latent_condition_dim)

    def __call__(self, encoded_t, encoded_state, encoded_condition):
        if not self._enabled:
            return encoded_t, encoded_state, encoded_condition
        encoded_condition = None if encoded_condition is None else self.condition(encoded_condition)
        return self.time(encoded_t), self.state(encoded_state), encoded_condition


class ConcatCombiner(BaseCombiner):

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
        *,
        layer_norm: bool = False,
    ) -> None:
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )
        self._norms = _InputLayerNorms(
            latent_state_dim, latent_time_dim, latent_condition_dim, enabled=layer_norm
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

        encoded_t, encoded_state, encoded_condition = self._norms(encoded_t, encoded_state, encoded_condition)

        # concatenating input
        to_concat = (encoded_state, make_concatenation_possible(encoded_t, encoded_state, -1))
        if encoded_condition is not None:
            to_concat = to_concat + (make_concatenation_possible(encoded_condition, encoded_state, -1),)
        return torch.cat(to_concat, dim=-1)


class AdaLNCombiner(BaseCombiner):
    """adaLN-zero conditioning (DiT-style). The **state** stream is the trunk input; ``[time (+ condition)]``
    is the modulation signal that drives each block's zero-initialized scale/shift/gate. Contrast
    :class:`ConcatCombiner`, which concatenates the condition into the decoder input — here the condition
    never enters the trunk stream, it only *modulates* it. Output width == ``latent_state_dim`` (the
    :class:`~sc_flow.nn.AdaLNZero1d` trunk preserves width), which the ``vf_decoder`` then projects to
    ``state_dim``."""

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
        *,
        net: AdaLNZero1d,
        layer_norm: bool = False,
    ) -> None:
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )
        self._net = net
        self._norms = _InputLayerNorms(
            latent_state_dim, latent_time_dim, latent_condition_dim, enabled=layer_norm
        )

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._verify_inputs_shape(
            encoded_t,
            encoded_state,
            encoded_condition,
        )

        encoded_t, encoded_state, encoded_condition = self._norms(encoded_t, encoded_state, encoded_condition)

        # modulation signal = [time (+ condition)]; the state stream is the trunk input it modulates.
        to_concat = (make_concatenation_possible(encoded_t, encoded_state, -1),)
        if encoded_condition is not None:
            to_concat = to_concat + (make_concatenation_possible(encoded_condition, encoded_state, -1),)
        modulation = torch.cat(to_concat, dim=-1)
        return self._net(encoded_state, modulation)

    @property
    def output_dim(
        self,
    ) -> int:
        return self._net.output_dim


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

        conditions = torch.cat(to_concat, dim=-1)
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
    #: LayerNorm each stream (state/time/condition) before concatenating (CellFlow2's
    #: ``layer_norm_before_concatenation``). Default off => byte-for-byte the legacy concat path.
    layer_norm: bool = False

    def build(self, context: CombinerContext) -> BaseCombiner:
        return ConcatCombiner(
            context.latent_state_dim,
            context.latent_time_dim,
            latent_condition_dim=context.latent_condition_dim,
            layer_norm=self.layer_norm,
        )


@COMBINER_REGISTRY.register("sc_flow.adaln")
@dataclass(frozen=True)
class AdaLNCombinerConfig:
    #: One :class:`~sc_flow.nn.AdaLNZero1d` block per unit (DiT-style trunk depth).
    num_blocks: int
    #: Per-block MLP hidden width = round(latent_state_dim * mlp_ratio).
    mlp_ratio: float = 4.0
    dropout_p: float = 0.0
    activation: str = "silu"
    #: LayerNorm the modulation streams (time/condition) + the state stream before the trunk.
    layer_norm: bool = False

    def build(self, context: CombinerContext) -> BaseCombiner:
        cond_dim = context.latent_time_dim + (context.latent_condition_dim or 0)
        net = AdaLNZero1d(
            dim=context.latent_state_dim,
            cond_dim=cond_dim,
            num_blocks=self.num_blocks,
            mlp_ratio=self.mlp_ratio,
            dropout_p=self.dropout_p,
            activation_cls=self.activation,
        )
        return AdaLNCombiner(
            context.latent_state_dim,
            context.latent_time_dim,
            latent_condition_dim=context.latent_condition_dim,
            net=net,
            layer_norm=self.layer_norm,
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
        if self.resnet["type"] != "sc_flow.resnet":
            raise ValueError(
                "sc_flow.resnet1d config slot 'resnet' requires a 'sc_flow.resnet' net spec, "
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
                "sc_flow.resnet1d config slot 'resnet' must build sc_flow.nn.Resnet1d, "
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
