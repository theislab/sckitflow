from __future__ import annotations

import json
from dataclasses import dataclass, field

from sc_flow._component import ComponentRegistry, ComponentSpec, JsonValue
from sc_flow.nn._activation import ActivationId, resolve_activation
from sc_flow.nn._modules import AdaLNZero, Resnet1d

__all__ = [
    "NET_REGISTRY",
    "AdaLNZeroConfig",
    "NetContext",
    "NetSpec",
    "ResnetConfig",
]

#: A net spec is a :class:`~sc_flow.ComponentSpec`; the consuming slot supplies tensor dimensions.
NetSpec = ComponentSpec


@dataclass(frozen=True)
class NetContext:
    """Dimensions supplied by the slot that owns a generic network.

    Authored net specs contain architecture hyperparameters only. Observed or parent-derived tensor widths
    stay in the build context, so the same spec can be built for another compatible input schema.
    """

    input_dim: int
    output_dim: int
    condition_dim: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("input_dim", self.input_dim), ("output_dim", self.output_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, found {value!r}.")
        if self.condition_dim is not None and (
            not isinstance(self.condition_dim, int) or isinstance(self.condition_dim, bool) or self.condition_dim <= 0
        ):
            raise ValueError(f"condition_dim must be a positive integer or None, found {self.condition_dim!r}.")


NET_REGISTRY: ComponentRegistry[NetContext] = ComponentRegistry("net")


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, found {type(value).__name__}.")


def _validate_positive(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive, found {value!r}.")


@NET_REGISTRY.register("sc_flow.resnet")
@dataclass(frozen=True)
class ResnetConfig:
    """Portable hyperparameters for a conditioned residual MLP.

    ``input_dim``, ``output_dim``, and the conditioning width are deliberately absent: the consuming slot
    provides them through :class:`NetContext`.
    """

    num_layers: int
    activation: ActivationId = "silu"
    use_batchnorm: bool = False
    batchnorm_eps: float = 1e-3
    batchnorm_momentum: float = 1e-2
    batchnorm_affine: bool = True
    batchnorm_track_running_stats: bool = True
    use_layernorm: bool = False
    layernorm_eps: float = 1e-5
    layernorm_elementwise_affine: bool = False
    layernorm_bias: bool = False
    dropout_p: float = 0.0
    dropout_inplace: bool = False
    activation_kwargs: dict[str, JsonValue] = field(default_factory=dict)
    bias: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.num_layers, int) or isinstance(self.num_layers, bool) or self.num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, found {self.num_layers!r}.")
        if not isinstance(self.activation, str):
            raise TypeError(f"activation must be a string id, found {type(self.activation).__name__}.")
        resolve_activation(self.activation, "silu")
        for name in (
            "use_batchnorm",
            "batchnorm_affine",
            "batchnorm_track_running_stats",
            "use_layernorm",
            "layernorm_elementwise_affine",
            "layernorm_bias",
            "dropout_inplace",
            "bias",
        ):
            _validate_bool(name, getattr(self, name))
        _validate_positive("batchnorm_eps", self.batchnorm_eps)
        _validate_positive("layernorm_eps", self.layernorm_eps)
        if (
            not isinstance(self.batchnorm_momentum, (int, float))
            or isinstance(self.batchnorm_momentum, bool)
            or not 0.0 <= self.batchnorm_momentum <= 1.0
        ):
            raise ValueError(f"batchnorm_momentum must be in [0, 1], found {self.batchnorm_momentum!r}.")
        if (
            not isinstance(self.dropout_p, (int, float))
            or isinstance(self.dropout_p, bool)
            or not 0.0 <= self.dropout_p <= 1.0
        ):
            raise ValueError(f"dropout_p must be in [0, 1], found {self.dropout_p!r}.")
        if not isinstance(self.activation_kwargs, dict):
            raise TypeError(f"activation_kwargs must be a dictionary, found {type(self.activation_kwargs).__name__}.")
        try:
            json.dumps(self.activation_kwargs)
        except (TypeError, ValueError) as e:
            raise TypeError("activation_kwargs must contain JSON-serializable values only.") from e

    def build(self, context: NetContext) -> Resnet1d:
        if context.condition_dim is None:
            raise ValueError("sc_flow.resnet requires NetContext.condition_dim.")
        return Resnet1d(
            input_dim=context.input_dim,
            embedding_dim=context.condition_dim,
            num_resnet_layers=self.num_layers,
            output_dim=context.output_dim,
            activation_cls=self.activation,
            use_batchnorm=self.use_batchnorm,
            batchnorm_eps=self.batchnorm_eps,
            batchnorm_momentum=self.batchnorm_momentum,
            batchnorm_affine=self.batchnorm_affine,
            batchnorm_track_running_stats=self.batchnorm_track_running_stats,
            use_layernorm=self.use_layernorm,
            layernorm_eps=self.layernorm_eps,
            layernorm_elementwise_affine=self.layernorm_elementwise_affine,
            layernorm_bias=self.layernorm_bias,
            dropout_p=self.dropout_p,
            dropout_inplace=self.dropout_inplace,
            activation_cls_kwargs=dict(self.activation_kwargs),
            bias=self.bias,
        )


@NET_REGISTRY.register("sc_flow.adaln_zero")
@dataclass(frozen=True)
class AdaLNZeroConfig:
    """Portable hyperparameters for an adaLN-Zero stack (:class:`~sc_flow.nn.AdaLNZero`).

    Like :class:`ResnetConfig`, the tensor widths come from :class:`NetContext`, not from here. Blocks are
    residual so they preserve width; ``num_layers`` sets depth and ``mlp_ratio`` the internal expansion
    (4.0 is the transformer/DiT convention).
    """

    num_layers: int = 1
    mlp_ratio: float = 4.0
    activation: ActivationId = "silu"
    dropout_p: float = 0.0
    layernorm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if not isinstance(self.num_layers, int) or isinstance(self.num_layers, bool) or self.num_layers < 1:
            raise ValueError(f"num_layers must be a positive integer, found {self.num_layers!r}.")
        _validate_positive("mlp_ratio", self.mlp_ratio)
        if self.dropout_p < 0.0 or self.dropout_p >= 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), found {self.dropout_p!r}.")

    def build(self, context: NetContext) -> AdaLNZero:
        if context.condition_dim is None:
            raise ValueError("sc_flow.adaln_zero requires NetContext.condition_dim.")
        return AdaLNZero(
            input_dim=context.input_dim,
            embedding_dim=context.condition_dim,
            num_layers=self.num_layers,
            output_dim=context.output_dim,
            mlp_ratio=self.mlp_ratio,
            activation_cls=self.activation,
            dropout_p=self.dropout_p,
            layernorm_eps=self.layernorm_eps,
        )
